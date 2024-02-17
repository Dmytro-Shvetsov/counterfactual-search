import torch
from torch import functional as F
from torch import nn

from src.models.cgan.blocks import snconv2d


class DiscriminatorResBlock(nn.Module):
    """Table 16 (c) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(self, in_channels=1, out_channels=64, downsample_scale=1, first_block=False):
        super().__init__()
        downsample = nn.AvgPool2d(downsample_scale, downsample_scale) if downsample_scale > 1 else nn.Identity()
        if first_block:
            self.left_branch = nn.Sequential(
                downsample,
                snconv2d.SNConv2d(in_channels, out_channels, kernel_size=1),
            )
        elif in_channels == out_channels:
            self.left_branch = nn.Identity()
        else:
            self.left_branch = nn.Sequential(
                snconv2d.SNConv2d(in_channels, out_channels, kernel_size=1),
                downsample,
            )

        self.right_branch = nn.Sequential(
            nn.ReLU() if not first_block else nn.Identity(),
            snconv2d.SNConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            snconv2d.SNConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            downsample,
        )
        self.initialize()

    def initialize(self):
        gain = torch.tensor(1.0)
        for m in self.left_branch.modules():
            if isinstance(m, snconv2d.SNConv2d):
                nn.init.xavier_uniform_(m.weight, gain)
        gain = torch.tensor(2.0).sqrt()
        for m in self.right_branch.modules():
            if isinstance(m, snconv2d.SNConv2d):
                nn.init.xavier_uniform_(m.weight, gain)

    def forward(self, x):
        left = self.left_branch(x)
        right = self.right_branch(x)
        # print(left.shape, right.shape, self.out_size)
        return left + right

# class DiscriminatorResBlock(chainer.Chain):
#     def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
#                  activation=nn.ReLU(), downsample=False):
#         super().__init__()
#         self.activation = activation
#         self.downsample = downsample
#         self.learnable_sc = (in_channels != out_channels) or downsample
#         hidden_channels = in_channels if hidden_channels is None else hidden_channels
#         with self.init_scope():
#             self.c1 = snconv2d.SNConv2d(in_channels, hidden_channels, ksize=ksize, pad=pad)
#             self.c2 = snconv2d.SNConv2d(hidden_channels, out_channels, ksize=ksize, pad=pad)
#             if self.learnable_sc:
#                 self.c_sc = snconv2d.SNConv2d(in_channels, out_channels, ksize=1, pad=0)

#     def residual(self, x):
#         h = x
#         h = self.activation(h)
#         h = self.c1(h)
#         h = self.activation(h)
#         h = self.c2(h)
#         if self.downsample:
#             h = _downsample(h)
#         return h

    # def shortcut(self, x):
    #     if self.learnable_sc:
    #         x = self.c_sc(x)
    #         if self.downsample:
    #             return _downsample(x)
    #         else:
    #             return x
    #     else:
    #         return x

    # def __call__(self, x):
    #     return self.residual(x) + self.shortcut(x)



if __name__ == '__main__':
    disk_down_64 = DiscriminatorResBlock(out_size=(64, 64), in_channels=1, out_channels=64)

    x = torch.zeros((16, 1, 256, 256))
    print(disk_down_64(x).shape)
