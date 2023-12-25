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

    def __init__(self, in_channels=1, downsample_scales=[2, 2, 2, 2, 2], out_channels=[64, 128, 256, 512, 1024]):
        super().__init__()
        assert len(downsample_scales) == len(out_channels)
        self.n_blocks = len(out_channels)

        self.out_channels = out_channels
        self.first_block = nn.Sequential(
            # original paper has bn->relu->conv
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.blocks = nn.ModuleList(
            [
                blocks.EncoderResBlock(64, out_channels[0], downsample_scales[0]),
                *(
                    blocks.EncoderResBlock(out_channels[i-1], out_channels[i], downsample_scales[i])
                    for i in range(1, self.n_blocks)
                ),
            ]
        )
        self.latent_dim = out_channels[-1]
        self.initialize()
        # print('Initialized encoder')
        # print(self)

    def initialize(self):
        nn.init.xavier_normal_(self.first_block[0].weight, torch.tensor(1.0))

    def forward(self, imgs):
        outs = self.first_block(imgs)
        features = []
        for b in self.blocks:
            outs = b(outs)
            features.append(outs)
        return features


class ResBlocksGenerator(nn.Module):
    """Table 5(b) - https://arxiv.org/pdf/2101.04230v3.pdf"""

    def __init__(
        self, 
        n_classes, 
        in_channels=[64, 128, 256, 512, 1024], 
        upsample_scales=[2, 2, 2, 2, 2], 
        out_channels=[1024, 512, 256, 128, 64],
        upsample_kind='nearest',
        skip_conn=None,
        ptb_fuse_type='skip_add_tanh',
    ):
        super().__init__()
        self.n_blocks = len(out_channels)
        assert len(upsample_scales) == len(out_channels)

        self.skip_conn = set(skip_conn or [])
        in_channels = in_channels[::-1]
        
        print('Generator in channels', in_channels)
        print('Generator out channels', out_channels)
        if skip_conn is None:
            in_channels = [in_channels[0]] + [out_channels[i-1] for i in range(1, self.n_blocks)]
        else:
            in_channels = [in_channels[0]] + [out_channels[i-1] + (in_channels[i] if i in self.skip_conn else 0) for i in range(1, self.n_blocks)]

        print('Generator in channels', in_channels)
        print('Generator out channels', out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                blocks.GeneratorResBlock(n_classes, in_channels[0], out_channels[0], scale_factor=upsample_scales[0], upsample_kind=upsample_kind),
                *(
                    blocks.GeneratorResBlock(n_classes, in_channels[i], out_channels[i], scale_factor=upsample_scales[i], upsample_kind=upsample_kind)
                    for i in range(1, self.n_blocks)
                ),
            ]
        )
        self.last_block = nn.Sequential(
            # TODO: think if relu -> BN is better
            nn.BatchNorm2d(out_channels[-1]), 
            nn.ReLU(),
            nn.Conv2d(out_channels[-1], 1, kernel_size=3, padding=1), 
        )
        self.ptb_fuse_type = ptb_fuse_type
        if self.ptb_fuse_type == 'skip_stack_3x3conv_tanh':
            self.fuse_conv = nn.Conv2d(2, 1, 3, padding=1)
        elif self.ptb_fuse_type == 'skip_stack_1x1conv_tanh':
            self.fuse_conv = nn.Conv2d(2, 1, 1, padding=1)
        else:
            self.fuse_conv = None
        self.tanh = nn.Tanh()
        self.initialize()
        print('Using perturbation fuse scheme:', ptb_fuse_type)
        # print('Initialized generator')
        # print(self)

    def initialize(self):
        nn.init.xavier_uniform_(self.last_block[2].weight, torch.tensor(1.0))

    def forward(self, enc_features, labels, x=None, ret_features=False):
        enc_features = enc_features[::-1] # revert the enc_features to begin with encoder head
        outs = enc_features[0] # latent variable `z`; (B, 1024, 8, 8) for img_shape=(256, 256)
        
        dec_features = {}
        for i, b in enumerate(self.blocks):
            # print(outs.shape)
            if i > 0 and i in self.skip_conn:
                # print('stack', enc_features[i].shape)
                outs = torch.cat((outs, enc_features[i]), 1)
            outs = b(outs, labels)
            if ret_features:
                dec_features[f'dec_block_{i}'] = outs
        outs = self.last_block(outs)
        if ret_features: dec_features['dec_last_block'] = outs
        outs = self._build_img(outs, x)
        return outs if not ret_features else (outs, dec_features)

    def _build_img(self, outs, x=None):
        if x is not None:
            if self.ptb_fuse_type == 'skip_add_tanh':
                outs = outs + x
                return self.tanh(outs)
            elif self.ptb_fuse_type == 'skip_add_l2_tanh':
                norms = torch.norm(outs, dim=(2, 3), keepdim=True)
                outs = outs / norms.clamp(min=1.0)
                return self.tanh(outs + x)
            elif self.ptb_fuse_type == 'skip_stack_3x3conv_tanh':
                outs = torch.cat((outs, x), 1)
                outs = self.fuse_conv(outs)
                return self.tanh(outs)
            elif self.ptb_fuse_type == 'skip_tanh_avg':
                return (x + self.tanh(outs)) / 2
            elif self.ptb_fuse_type == 'skip_tanh_clamp':
                return (x + self.tanh(outs)).clamp_(-1, 1)
                # return (x + self.tanh(outs) / 2).clamp_(-1, 1)
        return self.tanh(outs)


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
