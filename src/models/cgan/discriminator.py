import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(
        self, 
        n_classes, 
        in_channels=1, 
        downsample_scales=[2, 2, 2, 2, 2, 1], 
        out_channels=[64, 128, 256, 512, 1024, 1024],
        output_logits=True,
    ):
        super().__init__()
        assert len(downsample_scales) == len(out_channels)
        
        self.n_blocks = len(out_channels)
        
        self.blocks = nn.ModuleList(
            [
                blocks.DiscriminatorResBlock(in_channels, out_channels[0], downsample_scales[0], first_block=True),
                *( 
                    blocks.DiscriminatorResBlock(out_channels[i-1], out_channels[i], downsample_scales[i]) 
                    for i in range(1, self.n_blocks)
                ),
                nn.ReLU(),
                # nn.AvgPool2d(img_shape[0] // downsample_scales[-1], stride=1, divisor_override=1)  # global sum pooling (GSP)
                # nn.AdaptiveAvgPool2d(1)  # replacement of GSP
            ]
        )
        self.sn_dense = snlinear.SNLinear(out_channels[-1], 1)
        self.embd = nn.Embedding(n_classes, out_channels[-1])
        self.output_logits = output_logits

        self.initialize()
        # print('Initialized discriminator')
        # print(self)

    def initialize(self):
        gain = torch.tensor(1.0)
        nn.init.xavier_uniform_(self.sn_dense.weight, gain)
        nn.init.xavier_uniform_(self.embd.weight, gain)

    def forward(self, imgs, labels):
        outs = imgs
        labels = labels.view(-1)
        for b in self.blocks:
            outs = b(outs)
        # gsp = outs.view(*outs.shape[:2])  # (B, 1024, 1, 1)
        # gsp = outs.view(-1, outs.shape[1])  # (B, 1024)
        gsp = outs.sum(dim=(2, 3))
        # print('GSP', gsp.shape)

        sndense = self.sn_dense(gsp)  # (B, 1)

        # embed the labels (B, 1) -> (B, 1024)
        embed = self.embd(labels)
        # print('EMBED', embed.shape)
        inner_prod = (gsp * embed).sum(dim=1, keepdims=True)  # (B, 1)
        # print('INNER', inner_prod.shape)

        final_add = sndense + inner_prod
        return final_add if self.output_logits else F.logsigmoid(final_add).exp()


if __name__ == '__main__':
    disc = ResBlocksDiscriminator((256, 256), 4, 1)
    print('disc', sum(np.prod(p.shape) for p in disc.parameters()))
    imgs = torch.zeros((16, 1, 256, 256))
    labels = torch.zeros((16,)).long()
    print(disc(imgs, labels).shape)
