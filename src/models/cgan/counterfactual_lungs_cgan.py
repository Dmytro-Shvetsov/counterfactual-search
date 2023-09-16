import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from src.classifier import build_classifier
from src.losses import CARL, kl_divergence
from src.models.cgan.discriminator import ResBlocksDiscriminator
from src.models.cgan.generator import ResBlocksEncoder, ResBlocksGenerator
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


class CounterfactualLungsCGAN(nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.opt = opt
        self.latent_dim = opt.get('noise_dim', 1024)
        self.explain_class_idx = opt.explain_class_idx  # class id to be explained
        self.num_bins = opt.num_bins  # number of bins for explanation
        self.enc = ResBlocksEncoder(img_size, opt.in_channels)
        # generator and discriminator are conditioned against a discrete bin index which is computed
        # from the classifier probability for the explanation class using `posterior2bin` function
        self.gen = ResBlocksGenerator(self.num_bins, in_channels=self.latent_dim)
        self.disc = ResBlocksDiscriminator(img_size, self.num_bins, opt.in_channels)

        # black box classifier
        self.classifier_f = build_classifier(opt.n_classes, pretrained=False, restore_ckpt=opt.classifier_ckpt)
        self.classifier_f.eval()

        # Loss functions
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss() if opt.get('adv_loss', 'mse') == 'bce' else torch.nn.MSELoss()
        self.lambda_adv = opt.get('lambda_adv', 1.0)
        self.lambda_kl = opt.get('lambda_kl', 1.0)
        self.lambda_rec = opt.get('lambda_rec', 1.0)
        self.kl_clamp = 1e-8
        # by default update both generator and discriminator on each training step
        self.gen_update_freq = opt.get('gen_update_freq', 1)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain.from_iterable((self.enc.parameters(), self.gen.parameters())),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
        )
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.norms = {'E': None, 'G': None, 'D': None}

    def train(self, mode: bool = True) -> torch.nn.Module:
        ret = super().train(mode)
        # black-box classifier remains fixed throughout the whole training
        for mod in self.classifier_f.modules():
            mod.requires_grad_(False)
        self.classifier_f.eval()
        return ret

    def posterior_prob(self, x):
        assert self.classifier_f.training is False, 'Classifier is not set to evaluation mode'
        # f(x)[k] - classifier prediction at class k
        f_x = self.classifier_f(x).softmax(dim=1)[:, [self.explain_class_idx]]
        f_x_discrete = posterior2bin(f_x, self.num_bins)
        # the posterior probabilities `c` we would like to obtain after the explanation image is fed into the classifier
        f_x_desired = Variable(1.0 - f_x.detach(), requires_grad=False)
        f_x_desired_discrete = posterior2bin(f_x_desired, self.num_bins)
        return f_x, f_x_discrete, f_x_desired, f_x_desired_discrete

    def explanation_function(self, x, f_x_discrete, z=None):
        """Computes I_f(x, c)"""
        if z is None:
            # get embedding of the input image
            z = self.enc(x)
        # reconstruct explanation images
        gen_imgs = self.gen(z, f_x_discrete)
        return gen_imgs

    def reconstruction_loss(self, real_imgs, masks, f_x_discrete, f_x_desired_discrete, z=None):
        """
        Computes a reconstruction loss L_rec(E, G) that enforces self-consistency loss
        Formula 9 https://arxiv.org/pdf/2101.04230v3.pdf#page=7&zoom=100,30,412

        f_x_discrete - bin index for posterior probability f(x)
        f_x_desired_discrete - bin index for posterior probability 1 - f(x) (also known as desired probability)
        """
        # I_f(x, f(x))
        ifx_fx = self.explanation_function(real_imgs, f_x_discrete, z=z)
        # L_rec(x, I_f(x, f(x)))
        forward_term = CARL(real_imgs, ifx_fx, masks)

        # I_f(I_f(x, c), f(x))
        ifxc_fx = self.explanation_function(
            x=self.explanation_function(real_imgs, f_x_desired_discrete, z=z),  # I_f(x, c)
            f_x_discrete=f_x_discrete,
        )
        cyclic_term = CARL(real_imgs, ifxc_fx, masks)
        return forward_term + cyclic_term

    def forward(self, batch, training=False, compute_norms=False, global_step=None):
        imgs, labels, masks = batch['image'], batch['label'], batch['mask']
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

        # E(x)
        z = self.enc(real_imgs)
        # G(z, c)
        gen_imgs = self.gen(z, real_f_x_desired_discrete)

        update_generator = global_step is not None and global_step % self.gen_update_freq == 0
        # data consistency loss for generator
        validity = self.disc(gen_imgs, real_f_x_desired_discrete)
        g_adv_loss = self.lambda_adv * self.adversarial_loss(validity, valid)

        # classifier consistency loss for generator
        # f(I_f(x, c)) ≈ c
        gen_f_x, _, _, _ = self.posterior_prob(gen_imgs)
        # both y_pred and y_target are single-value probs for class k
        g_kl = self.lambda_kl * kl_divergence(gen_f_x, real_f_x_desired)
        # reconstruction loss for generator
        g_rec_loss = self.lambda_rec * self.reconstruction_loss(real_imgs, masks, real_f_x_discrete, real_f_x_desired_discrete, z=z)

        # total generator loss
        g_loss = g_adv_loss + g_kl + g_rec_loss

        if training and update_generator:
            g_loss.backward()
            if compute_norms:
                self.norms['E'] = grad_norm(self.enc)
                self.norms['G'] = grad_norm(self.gen)
            self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if training:
            self.optimizer_D.zero_grad()

        # data consistency loss for discriminator (real images)
        validity_real = self.disc(real_imgs, real_f_x_desired_discrete)
        d_real_loss = self.adversarial_loss(validity_real, valid)

        # data consistency loss for discriminator (fake images)
        validity_fake = self.disc(gen_imgs.detach(), real_f_x_desired_discrete)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)

        # total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            d_loss.backward()
            if compute_norms:
                self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()

        outs = {
            'loss': {
                # generator
                'g_adv': g_adv_loss.item(),
                'g_kl': g_kl.item(),
                'g_rec_loss': g_rec_loss.item(),
                'g_loss': g_loss.item(),
                # discriminator
                'd_real_loss': d_real_loss.item(),
                'd_fake_loss': d_fake_loss.item(),
                'd_loss': d_loss.item(),
            },
            'gen_imgs': gen_imgs,
        }
        return outs
