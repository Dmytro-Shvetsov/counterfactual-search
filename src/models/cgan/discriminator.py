import numpy as np
import torch
import torch.nn as nn

from src.models.cgan import blocks
from src.models.cgan.blocks import snlinear


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class ResBlocksDiscriminator(nn.Module):
    """Table 5(c) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(self, img_shape, n_classes, in_channels=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                blocks.DiscriminatorResBlock([d // 2 for d in img_shape], in_channels, out_channels=64),
                blocks.DiscriminatorResBlock([d // 4 for d in img_shape], 64, out_channels=128),
                blocks.DiscriminatorResBlock([d // 8 for d in img_shape], 128, out_channels=256),
                blocks.DiscriminatorResBlock([d // 16 for d in img_shape], 256, out_channels=512),
                blocks.DiscriminatorResBlock([d // 32 for d in img_shape], 512, out_channels=1024),
                blocks.DiscriminatorResBlock([d // 32 for d in img_shape], 1024, out_channels=1024),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(1),  # global sum pooling (GSP)
            ]
        )
        self.sn_dense = snlinear.SNLinear(1024, 1)
        self.embd = nn.Embedding(n_classes, 1024)

    def forward(self, imgs, labels):
        outs = imgs
        labels = labels.view(-1)
        for b in self.blocks:
            outs = b(outs)
        gsp = outs.view(*outs.shape[:2])  # (B, 1024, 1, 1)
        # print('GSP', gsp.shape)

        sndense = self.sn_dense(gsp)  # (B, 1)

        # embed the labels (B, 1) -> (B, 1024)
        embed = self.embd(labels)
        # print('EMBED', embed.shape)
        inner_prod = (gsp * embed).sum(dim=1, keepdims=True)  # (B, 1)
        # print('INNER', inner_prod.shape)

        final_add = sndense + inner_prod
        return final_add


if __name__ == '__main__':
    disc = ResBlocksDiscriminator((256, 256), 4, 1)
    print('disc', sum(np.prod(p.shape) for p in disc.parameters()))
    imgs = torch.zeros((16, 1, 256, 256))
    labels = torch.zeros((16,)).long()
    print(disc(imgs, labels).shape)
