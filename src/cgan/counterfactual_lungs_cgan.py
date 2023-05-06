import itertools
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from src.utils.grad_norm import grad_norm

from src.cgan.discriminator import ResBlocksDiscriminator
from src.cgan.generator import ResBlocksEncoder, ResBlocksGenerator
from src.classifier import build_classifier
from src.losses import CARL

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


@torch.no_grad()
def posterior2bin(posterior_pred:torch.Tensor, num_bins:int) -> torch.Tensor:
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
    return LongTensor(bin_ids)


class CounterfactualLungsCGAN(nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.opt = opt
        self.latent_dim = opt.get('noise_dim', 1024)
        self.explain_class_idx = opt.explain_class_idx # class id to be explained
        self.num_bins = opt.num_bins # number of bins for explanation
        self.enc = ResBlocksEncoder(img_size, opt.in_channels)
        # generator and discriminator are conditioned against a discrete bin index which is computed
        # from the classifier probability for the explanation class using `posterior2bin` function
        self.gen = ResBlocksGenerator(self.num_bins, in_channels=self.latent_dim)
        self.disc = ResBlocksDiscriminator(img_size, self.num_bins, opt.in_channels)

        # black box classifier
        self.classifier_f = build_classifier(opt.n_classes, pretrained=False, restore_ckpt=opt.classifier_ckpt)
        self.classifier_f.eval()

        # Loss functions
        self.adversarial_loss = torch.nn.MSELoss()
        # self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.lambda_adv = 1.0
        self.lambda_rec = 1.0

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
            x=self.explanation_function(real_imgs, f_x_desired_discrete, z=z), # I_f(x, c)
            f_x_discrete=f_x_discrete,
        )
        cyclic_term = CARL(real_imgs, ifxc_fx, masks)
        return forward_term + cyclic_term

    def forward(self, batch, training=False, compute_norms=False):
        # NOTE: images are expected to be sampled (TODO)
        imgs, labels, masks = batch['image'], batch['label'], batch['mask']
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        masks = Variable(masks.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        with torch.no_grad():
            assert self.classifier_f.training is False, 'Classifier is not set to evaluation mode'
            # f(x)
            f_x = self.classifier_f(real_imgs).softmax(dim=1)[:, [self.explain_class_idx]]
            f_x_discrete = posterior2bin(f_x, self.num_bins)
            # the posterior probabilities `c` we would like to obtain after the explanation image is fed into the classifier
            f_x_desired = 1.0 - f_x
            f_x_desired_discrete = posterior2bin(f_x_desired, self.num_bins)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        if training: self.optimizer_G.zero_grad()

        # E(x) 
        z = self.enc(real_imgs)
        # G(z, c) 
        gen_imgs = self.gen(z, f_x_desired_discrete)

        # data consistency loss for generator
        validity = self.disc(gen_imgs, f_x_desired_discrete)
        g_adv_loss = self.adversarial_loss(validity, valid)
        # reconstruction loss for generator
        g_rec_loss = self.reconstruction_loss(real_imgs, masks, f_x_discrete, f_x_desired_discrete, z=z)

        # total generator loss
        g_loss = self.lambda_adv * g_adv_loss + self.lambda_rec * g_rec_loss

        if training: 
            g_loss.backward()
            if compute_norms:
                self.norms['E'] = grad_norm(self.enc) 
                self.norms['G'] = grad_norm(self.gen)
            self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if training: self.optimizer_D.zero_grad()

        # data consistency loss for discriminator (real images)
        validity_real = self.disc(real_imgs, f_x_desired_discrete)
        d_real_loss = self.adversarial_loss(validity_real, valid)

        # data consistency loss for discriminator (fake images)
        validity_fake = self.disc(gen_imgs.detach(), f_x_desired_discrete)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)

        # total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            d_loss.backward()
            if compute_norms: self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()

        return {
            'loss': {
                'g_loss': g_loss.item(),
                'g_adv': g_adv_loss.item(),
                'g_loss': g_loss.item(),
                'g_rec_loss': g_rec_loss,
                'd_real_loss': d_real_loss.item(),
                'd_loss': d_loss.item(),
                'd_fake_loss': d_fake_loss.item(),
            },
            'gen_imgs': gen_imgs,
        }

    def sample_image(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = self.gen(z, labels)
        return gen_imgs
