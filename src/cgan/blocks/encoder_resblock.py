import torch
from torch import nn

from src.cgan.blocks import snconv2d

class EncoderResBlock(nn.Module):
  """ Table 16 (a) - https://arxiv.org/pdf/2101.04230v3.pdf"""

  def __init__(self, out_size, in_channels=1, out_channels=64):
    super().__init__()

    self.left_branch = nn.Sequential(
        nn.AdaptiveAvgPool2d(out_size),
        snconv2d.SNConv2d(in_channels, out_channels, kernel_size=1, padding=0),
    )

    self.right_branch = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(out_size),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )

  def forward(self, x):
    left = self.left_branch(x)
    right = self.right_branch(x)
    return left + right


if __name__ == '__main__':
    gen_up_1024 = EncoderResBlock((128, 128), 1, 128)

    x = torch.zeros((16, 1, 256, 256))
    print(gen_up_1024(x).shape)
