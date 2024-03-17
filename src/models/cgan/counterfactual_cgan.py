from functools import partial
import itertools

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.losses import CARL, kl_divergence, loss_hinge_dis, loss_hinge_gen
from src.models.cgan.discriminator import ResBlocksDiscriminator
from src.models.cgan.generator import ResBlocksEncoder, ResBlocksGenerator
from src.models.classifier import build_classifier
from src.utils.grad_norm import grad_norm

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


@torch.no_grad()
def posterior2bin(posterior_pred: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Given classifier predictions in range [0; 1] and the number of condition bins, returns the condition labels.

    Args:
        posterior_pred (torch.Tensor): classifier predictions
        num_bins (int): number of conditions

    Returns:
        torch.Tensor: resulting condition labels
    """
    posterior_pred = posterior_pred.cpu().numpy()
    bin_step = 1 / num_bins
    bins = np.arange(0, 1, bin_step)
    bin_ids = np.digitize(posterior_pred, bins=bins) - 1
    return Variable(LongTensor(bin_ids), requires_grad=False)


class CounterfactualCGAN(nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.opt = opt
        self.img_size = img_size
        self.explain_class_idx = opt.explain_class_idx  # class id to be explained
        # generator and discriminator are conditioned against a discrete bin index which is computed
        # from the classifier probability for the explanation class using `posterior2bin` function
        self.num_bins = opt.num_bins  # number of bins for explanation
        self.ptb_based = opt.get('ptb_based', False)
        
        self.enc = ResBlocksEncoder(opt.in_channels, **opt.get('enc_params', {}))
        self.gen = ResBlocksGenerator(self.num_bins, in_channels=self.enc.out_channels, **opt.get('gen_params', {}))
        self.disc = ResBlocksDiscriminator(self.num_bins, opt.in_channels, **opt.get('disc_params', {}))

        # black box classifier
        self.n_classes = opt.n_classes
        self.classifier_f = build_classifier(opt.classifier_kind, opt.n_classes, pretrained=False, restore_ckpt=opt.classifier_ckpt)
        self.classifier_f.eval()

        # Loss functions
        self.adv_loss = opt.get('adv_loss', 'mse')
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss() if self.adv_loss else torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.rec_kind = opt.get('rec_kind', 'carl')
        self.lambda_adv = opt.get('lambda_adv', 1.0)
        self.lambda_kl = opt.get('lambda_kl', 1.0)
        self.lambda_rec = opt.get('lambda_rec', 1.0)
        self.lambda_minc = opt.get('lambda_minc', 1.0)
        
        self.eps = opt.get('eps', 1e-8)
        
        self.kl_clamp = self.eps
        # by default update both generator and discriminator on each training step
        self.gen_update_freq = opt.get('gen_update_freq', 1)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain.from_iterable((self.enc.parameters(), self.gen.parameters())),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
            eps=self.eps,
        )
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=self.eps)
        self.norms = {'E': None, 'G': None, 'D': None}
        self.gen_loss_logs, self.disc_loss_logs = {}, {}
        self.fabric_setup = False

    @property
    def classifier_kind(self):
        return self.opt.classifier_kind

    def prepare_fabric(self) -> None:
        self.fabric = L.Fabric(precision=self.opt.get('precision', '32'))
        self.fabric.launch()
        self.enc = self.fabric.setup_module(self.enc)
        self.gen = self.fabric.setup_module(self.gen)
        self.disc = self.fabric.setup_module(self.disc)
        self.optimizer_D, self.optimizer_G = self.fabric.setup_optimizers(self.optimizer_D, self.optimizer_G)

    def train(self, mode: bool = True) -> torch.nn.Module:
        if not self.fabric_setup:
            self.prepare_fabric()
            self.fabric_setup = True
        ret = super().train(mode)
        # black-box classifier remains fixed throughout the whole training
        for mod in self.classifier_f.modules():
            mod.requires_grad_(False)
        self.classifier_f.eval()
        return ret

    def posterior_prob(self, x):
        assert self.classifier_f.training is False, 'Classifier is not set to evaluation mode'
        # f(x)[k] - classifier prediction at class k
        f_x = self.classifier_f(x)
        f_x = f_x.softmax(dim=1) if self.n_classes > 1 else f_x.sigmoid()
        f_x = f_x[:, [self.explain_class_idx]]
        f_x_discrete = posterior2bin(f_x, self.num_bins)
        # the posterior probabilities `c` we would like to obtain after the explanation image is fed into the classifier
        f_x_desired = Variable(1.0 - f_x.detach(), requires_grad=False)
        f_x_desired_discrete = posterior2bin(f_x_desired, self.num_bins)
        return f_x, f_x_discrete, f_x_desired, f_x_desired_discrete

    def explanation_function(self, x, f_x_discrete, z=None, ret_features=False):
        """Computes I_f(x, c)"""
        if z is None:
            # get embedding of the input image
            z = self.enc(x)
        # reconstruct explanation images
        gen_imgs = self.gen(z, f_x_discrete, x=x if self.ptb_based else None, ret_features=ret_features)
        return gen_imgs

    def reconstruction_loss(self, real_imgs, gen_imgs, masks, f_x_discrete, f_x_desired_discrete, z=None):
        """
        Computes a reconstruction loss L_rec(E, G) that enforces self-consistency loss
        Formula 9 https://arxiv.org/pdf/2101.04230v3.pdf#page=7&zoom=100,30,412
        
        real_imgs - input images that are explained
        gen_imgs - generated images with condition label 1 - f(x) (i.e computation of I_f(x, c))
        masks - semantic segmentation masks to be used for CARL loss to enforce local consistency for each label
        f_x_discrete - bin index for posterior probability f(x)
        f_x_desired_discrete - bin index for posterior probability 1 - f(x) (also known as desired probability)
        """
        rec_fn = partial(CARL, masks=masks) if self.rec_kind.lower() == 'carl' else self.l1 
        # I_f(x, f(x))
        ifx_fx = self.explanation_function(real_imgs, f_x_discrete, z=z)
        # L_rec(x, I_f(x, f(x)))
        forward_term = rec_fn(real_imgs, ifx_fx)

        # I_f(I_f(x, c), f(x))
        ifxc_fx = self.explanation_function(
            x=gen_imgs,  # I_f(x, c)
            f_x_discrete=f_x_discrete,
        )
        # L_rec(x, I_f(I_f(x, c), f(x)))
        cyclic_term = rec_fn(real_imgs, ifxc_fx)
        return forward_term + cyclic_term

    def forward(self, batch, training=False, validation=False, compute_norms=False, global_step=None):
        assert training and not validation or validation and not training

        # imgs are in [-1, 1] range
        imgs, labels, masks = batch['image'], batch['label'], batch['masks']
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        masks = Variable(masks.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Classifier predictions and desired outputs for the explanation function
        with torch.no_grad():
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.posterior_prob(real_imgs)

        # -----------------
        #  Train Generator
        # -----------------
        if training:
            self.optimizer_G.zero_grad()

        # `real_imgs` and `gen_imgs` are in [-1, 1] range
        # E(x)
        z = self.enc(real_imgs)
        # G(z, c) = I_f(x, c)
        gen_imgs = self.gen(z, real_f_x_desired_discrete, x=real_imgs if self.ptb_based else None)

        update_generator = global_step is not None and global_step % self.gen_update_freq == 0
        
        # data consistency loss for generator
        if update_generator or validation:
            dis_fake = self.disc(gen_imgs, real_f_x_desired_discrete)
            if self.adv_loss == 'hinge':
                g_adv_loss = self.lambda_adv * loss_hinge_gen(dis_fake)
            else:
                g_adv_loss = self.lambda_adv * self.adversarial_loss(dis_fake, valid)

            # classifier consistency loss for generator
            # f(I_f(x, c)) ≈ c
            gen_f_x, _, _, _ = self.posterior_prob(gen_imgs)
            # both y_pred and y_target are single-value probs for class k
            g_kl = (
                self.lambda_kl * kl_divergence(gen_f_x, real_f_x_desired)
                if self.lambda_kl != 0 else torch.tensor(0.0, requires_grad=True)
            )
            # reconstruction loss for generator
            g_rec_loss = (
                self.lambda_rec * self.reconstruction_loss(real_imgs, gen_imgs, masks, real_f_x_discrete, real_f_x_desired_discrete, z=z)
                if self.lambda_rec != 0 else torch.tensor(0.0, requires_grad=True)
            )
            # total generator loss
            g_loss = g_adv_loss + g_kl + g_rec_loss

            # update generator
            if update_generator:
                self.fabric.backward(g_loss)
                if compute_norms:
                    self.norms['E'] = grad_norm(self.enc)
                    self.norms['G'] = grad_norm(self.gen)
                self.optimizer_G.step()

            self.gen_loss_logs['g_adv'] = g_adv_loss.item()
            self.gen_loss_logs['g_kl'] = g_kl.item()
            self.gen_loss_logs['g_rec_loss'] = g_rec_loss.item()
            self.gen_loss_logs['g_loss'] = g_loss.item()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if training:
            self.optimizer_D.zero_grad()

        dis_real = self.disc(real_imgs, real_f_x_discrete) # changed from real_f_x_desired_discrete to real_f_x_discrete
        dis_fake = self.disc(gen_imgs.detach(), real_f_x_desired_discrete)

        # data consistency loss for discriminator (real and fake images)
        if self.adv_loss == 'hinge':
            d_real_loss, d_fake_loss = loss_hinge_dis(dis_fake, dis_real)
        else:
            d_real_loss = self.adversarial_loss(dis_real, valid)
            d_fake_loss = self.adversarial_loss(dis_fake, fake)
        
        # total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            self.fabric.backward(d_loss)
            if compute_norms:
                self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()

        self.disc_loss_logs['d_real_loss'] = d_real_loss.item()
        self.disc_loss_logs['d_fake_loss'] = d_fake_loss.item()
        self.disc_loss_logs['d_loss'] = d_loss.item()

        outs = {
            'loss': {**self.gen_loss_logs, **self.disc_loss_logs},
            'gen_imgs': gen_imgs,
        }
        return outs

    def generate_counterfactual(self, real_imgs:torch.Tensor) -> torch.Tensor:
        real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.posterior_prob(real_imgs)
        gen_cf_c = self.explanation_function(real_imgs, real_f_x_desired_discrete)
        diff = (real_imgs - gen_cf_c).abs()
        return real_f_x_discrete, real_f_x_desired_discrete, gen_cf_c, diff
