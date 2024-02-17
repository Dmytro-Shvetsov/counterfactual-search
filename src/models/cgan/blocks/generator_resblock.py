import torch
from torch import nn

from src.models.cgan.blocks import cbn, snconv2d


def get_upsampling_layer(kind, scale_factor):
    if kind == 'nearest':
        return nn.UpsamplingNearest2d(scale_factor=scale_factor)
    elif kind == 'bilinear':
        # `nn.UpsamplingBilinear2d` is non-deterministic op: https://github.com/pytorch/pytorch/issues/7068#issuecomment-716719820
        return nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    elif kind == 'transposed':
        return nn.PixelShuffle(scale_factor)
    else:
        raise ValueError(f'Unsupported upsampling layer kind: {kind}')


class GeneratorResBlock(nn.Module):
    """Table 16 (b) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(
        self,
        num_classes,
        in_channels=1024,
        out_channels=512,
        scale_factor=2,
        upsample_kind='nearest',
        use_snconv=True,
    ):
        super().__init__()

        upsample = get_upsampling_layer(upsample_kind, scale_factor) if scale_factor > 1 else nn.Identity()
        ds_conv = snconv2d.SNConv2d if use_snconv else nn.Conv2d
        self.left_branch = nn.Sequential(
            upsample,
            ds_conv(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )

        self.cbn_relu_first = cbn.ConditionalBatchNorm2d(in_channels, num_classes, act=nn.ReLU())
        self.cbn_relu_second = cbn.ConditionalBatchNorm2d(out_channels, num_classes, act=nn.ReLU())
        self.upsample_conv = nn.Sequential(
            upsample,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.last_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.initialize()

    def initialize(self):
        gain = torch.tensor(1.0)
        for m in self.left_branch.modules():
            if isinstance(m, (snconv2d.SNConv2d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain)
        gain = torch.tensor(2.0).sqrt()
        for m in self.upsample_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain)
        nn.init.xavier_uniform_(self.last_conv.weight, gain)

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
