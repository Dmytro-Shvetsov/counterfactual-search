import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from src.cgan.discriminator import ResBlocksDiscriminator
from src.cgan.generator import ResBlocksEncoder, ResBlocksGenerator

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class LungsCGAN(nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.opt = opt
        self.latent_dim = opt.get('noise_dim', 1024)
        self.enc = ResBlocksEncoder(img_size, opt.in_channels)
        self.gen = ResBlocksGenerator(opt.n_classes, in_channels=self.latent_dim)
        self.disc = ResBlocksDiscriminator(img_size, opt.n_classes, opt.in_channels)
        
        # Loss functions
        self.adversarial_loss = torch.nn.MSELoss()

        self.optimizer_G = torch.optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    def forward(self, batch, training=False):
        imgs, labels = batch['image'], batch['label']
        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        


        # -----------------
        #  Train Generator
        # -----------------

        if training: self.optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim, 8, 8))))
        gen_labels = Variable(LongTensor(np.random.randint(0, self.opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = self.gen(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = self.disc(gen_imgs, gen_labels)
        g_loss = self.adversarial_loss(validity, valid)

        if training: 
            g_loss.backward()
            self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if training: self.optimizer_D.zero_grad()

        # Loss for real images
        validity_real = self.disc(real_imgs, labels)
        d_real_loss = self.adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = self.disc(gen_imgs.detach(), gen_labels)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            d_loss.backward()
            self.optimizer_D.step()

        return {
            'loss': {
                'g_loss': g_loss.item(),
                'd_real_loss': d_real_loss.item(),
                'd_loss': d_loss.item(),
                'd_fake_loss': d_fake_loss.item(),
            },
            'gen_imgs': gen_imgs,
            'gen_labels': gen_labels,
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
