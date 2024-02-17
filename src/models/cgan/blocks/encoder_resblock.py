import torch
from torch import nn

from src.models.cgan.blocks import snconv2d


class EncoderResBlock(nn.Module):
    """Table 16 (a) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(self, in_channels=1, out_channels=64, downsample_scale=1, use_snconv=True):
        super().__init__()

        downsample = nn.AvgPool2d(downsample_scale, downsample_scale) if downsample_scale > 1 else nn.Identity()
        ds_conv = snconv2d.SNConv2d if use_snconv else nn.Conv2d
        self.left_branch = nn.Sequential(
            downsample,
            ds_conv(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )

        self.right_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            downsample,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.initialize()

    def initialize(self):
        gain = torch.tensor(1.0)
        for m in self.left_branch.modules():
            if isinstance(m, (snconv2d.SNConv2d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain)
        gain = torch.tensor(2.0).sqrt()
        for m in self.right_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain)

    def forward(self, x):
        left = self.left_branch(x)
        right = self.right_branch(x)
        return left + right


if __name__ == '__main__':
    gen_up_1024 = EncoderResBlock((128, 128), 1, 128)

    x = torch.zeros((16, 1, 256, 256))
    print(gen_up_1024(x).shape)
