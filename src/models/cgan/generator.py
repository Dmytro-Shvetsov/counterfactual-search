import numpy as np
import torch
import torch.nn as nn

from src.models.cgan import blocks


class Generator(nn.Module):
    def __init__(self, n_classes, img_shape, latent_dim=100):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class ResBlocksEncoder(nn.Module):
    """Table 5(a) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(self, img_shape, in_channels=1):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        )
        self.blocks = nn.ModuleList(
            [
                blocks.EncoderResBlock([d // 2 for d in img_shape], in_channels, out_channels=64),
                blocks.EncoderResBlock([d // 4 for d in img_shape], 64, out_channels=128),
                blocks.EncoderResBlock([d // 8 for d in img_shape], 128, out_channels=256),
                blocks.EncoderResBlock([d // 16 for d in img_shape], 256, out_channels=512),
                blocks.EncoderResBlock([d // 32 for d in img_shape], 512, out_channels=1024),
            ]
        )

    def forward(self, imgs):
        outs = imgs
        for b in self.blocks:
            outs = b(outs)
        return outs


class ResBlocksGenerator(nn.Module):
    """Table 5(b) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(self, n_classes, in_channels=1024):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                blocks.GeneratorResBlock(n_classes, in_channels, in_channels, scale_factor=2),
                blocks.GeneratorResBlock(n_classes, in_channels, 512, scale_factor=2),
                blocks.GeneratorResBlock(n_classes, 512, 256, scale_factor=2),
                blocks.GeneratorResBlock(n_classes, 256, 128, scale_factor=2),
                blocks.GeneratorResBlock(n_classes, 128, 64, scale_factor=2),
            ]
        )
        self.last_block = nn.Sequential(
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1), 
            nn.Tanh(),
        )

    def forward(self, z, labels):
        outs = z
        for b in self.blocks:
            outs = b(outs, labels)
        outs = self.last_block(outs)
        # (B, 1024, 8, 8) for img_shape=(256, 256)
        return outs


if __name__ == '__main__':
    enc = ResBlocksEncoder((256, 256), 1)
    print('enc', sum(np.prod(p.shape) for p in enc.parameters() if p.requires_grad))
    imgs = torch.zeros((16, 1, 256, 256))
    z = enc(imgs)
    print(z.shape)

    gen = ResBlocksGenerator(4, 1024)
    print('gen', sum(np.prod(p.shape) for p in gen.parameters()))
    z = torch.zeros((16, 1024, 8, 8))
    labels = torch.zeros((16,)).long()
    print(gen(z, labels).shape)
