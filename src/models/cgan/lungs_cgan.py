import torch
import torch.nn as nn
from torch.autograd import Variable

from src.models.cgan.discriminator import ResBlocksDiscriminator
from src.models.cgan.generator import ResBlocksEncoder, ResBlocksGenerator
from src.utils.grad_norm import grad_norm

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
        self.pixel_loss = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.norms = {'G': None, 'D': None}

    def forward(self, batch, training=False, compute_norms=False):
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

        if training:
            self.optimizer_G.zero_grad()

        z = self.enc(real_imgs)
        gen_imgs = self.gen(z, labels)

        # Loss measures generator's ability to fool the discriminator
        validity = self.disc(gen_imgs, labels)
        g_pixel_loss = self.pixel_loss(gen_imgs, real_imgs)
        g_adv_loss = self.adversarial_loss(validity, valid) + g_pixel_loss
        g_loss = g_pixel_loss + g_adv_loss

        if training:
            g_loss.backward()
            if compute_norms:
                self.norms['G'] = grad_norm(self.gen)
            self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if training:
            self.optimizer_D.zero_grad()

        # Loss for real images
        validity_real = self.disc(real_imgs, labels)
        d_real_loss = self.adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = self.disc(gen_imgs.detach(), labels)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            d_loss.backward()
            if compute_norms:
                self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()

        return {
            'loss': {
                'g_pixel_loss': g_pixel_loss.item(),
                'g_adv': g_adv_loss.item(),
                'g_loss': g_loss.item(),
                'pixel_loss': g_pixel_loss,
                'd_real_loss': d_real_loss.item(),
                'd_loss': d_loss.item(),
                'd_fake_loss': d_fake_loss.item(),
            },
            'gen_imgs': gen_imgs,
            # 'gen_labels': gen_labels,
        }
