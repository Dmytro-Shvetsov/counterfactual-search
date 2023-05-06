import torch
from torch import nn

from src.cgan.blocks import snconv2d, cbn

class GeneratorResBlock(nn.Module):
  """ Table 16 (b) - https://arxiv.org/pdf/2101.04230v3.pdf"""

  def __init__(self, num_classes, in_channels=1024, out_channels=512, scale_factor=2,):
    super().__init__()

    self.left_branch = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=scale_factor),
        snconv2d.SNConv2d(in_channels, out_channels, kernel_size=1, padding=0),
    )

    self.cbn_relu_first = cbn.ConditionalBatchNorm2d(in_channels, num_classes, act=nn.ReLU())
    self.cbn_relu_second = cbn.ConditionalBatchNorm2d(out_channels, num_classes, act=nn.ReLU())
    self.upsample_conv = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=scale_factor),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    )
    self.last_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x, labels):
    labels = labels.view(-1)
    left = self.left_branch(x)

    right = self.cbn_relu_first(x, labels)
    right = self.upsample_conv(right)
    right = self.cbn_relu_second(right, labels)
    right = self.last_conv(right)

    return left + right


if __name__ == '__main__':
    gen_up_1024 = GeneratorResBlock(4, 1024, 512, scale_factor=2)

    x = torch.zeros((16, 1024, 8, 8))
    labels = torch.zeros((16,)).long()
    print(gen_up_1024(x, labels).shape)
