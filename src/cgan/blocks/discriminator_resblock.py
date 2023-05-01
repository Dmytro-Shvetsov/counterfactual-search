import torch
from torch import nn

from src.cgan.blocks import snconv2d

class DiscriminatorResBlock(nn.Module):
  """ Table 16 (c) - https://arxiv.org/pdf/2101.04230v3.pdf"""

  def __init__(self, out_size, in_channels=1, out_channels=64):
    super().__init__()
    self.out_size = out_size
    self.left_branch = nn.Sequential(
        snconv2d.SNConv2d(in_channels, out_channels, kernel_size=1),
        nn.AdaptiveAvgPool2d(out_size),
    )

    self.right_branch = nn.Sequential(
        nn.ReLU(),
        snconv2d.SNConv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        snconv2d.SNConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(out_size),
    )

  def forward(self, x):
    left = self.left_branch(x)
    right = self.right_branch(x)
    # print(left.shape, right.shape, self.out_size)
    return left + right


if __name__ == '__main__':
    disk_down_64 = DiscriminatorResBlock(out_size=(64, 64), in_channels=1, out_channels=64)

    x = torch.zeros((16, 1, 256, 256))
    print(disk_down_64(x).shape)
