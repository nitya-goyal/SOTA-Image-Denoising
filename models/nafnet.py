"""NAFNet base architecture and controlled attention variants.
Standard architecture adapted from megvii-research/NAFNet (ECCV 2022).
CBAM attention mechanisms adapted from Woo et al., ECCV 2018.
Copyright (c) 2022 megvii-model. See third_party/NAFNet-LICENSE.
Uses native autograd for channel LayerNorm; no BasicSR or local-pooling dependency.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial position."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, image):
        channels_last = image.permute(0, 2, 3, 1)
        normalized = F.layer_norm(channels_last, (image.shape[1],), self.weight, self.bias, self.eps)
        return normalized.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, image):
        first, second = image.chunk(2, dim=1)
        return first * second


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention (Woo et al., ECCV 2018).
    Concatenates channel-wise mean and maximum feature maps, applies a 7x7 conv
    with padding 3 and explicit bias=True (99 params), followed by a sigmoid to scale spatial positions.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        maximum = x.max(dim=1, keepdim=True)[0]
        mask = torch.sigmoid(self.conv(torch.cat([mean, maximum], dim=1)))
        return x * mask


class CBAMChannelAttention(nn.Module):
    """CBAM-style channel attention (Woo et al., ECCV 2018).
    Uses global average and maximum pooling, a shared two-layer 1x1 MLP with ReLU
    and reduction ratio 16 (hidden channels at least 1), with explicit bias=True on both conv layers,
    sums branch outputs, and applies sigmoid and feature multiplication.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        mask = torch.sigmoid(avg_out + max_out)
        return x * mask


class NAFBlock(nn.Module):
    def __init__(self, channels, attention='sca'):
        super().__init__()
        if attention not in ('sca', 'spatial', 'channel', 'combined'):
            raise ValueError(f"Unknown attention type: {attention}; choose 'sca', 'spatial', 'channel', or 'combined'")
        self.attention = attention
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, expanded, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.sg = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if attention in ('sca', 'spatial'):
            self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        else:
            self.channel_att = CBAMChannelAttention(channels)

        if attention in ('spatial', 'combined'):
            self.spatial_att = SpatialAttention()

    def forward(self, image):
        spatial = self.sg(self.conv2(self.conv1(self.norm1(image))))
        if self.attention in ('sca', 'spatial'):
            spatial = spatial * self.sca(spatial)
        else:
            spatial = self.channel_att(spatial)
        if self.attention in ('spatial', 'combined'):
            spatial = self.spatial_att(spatial)
        spatial = self.conv3(spatial)
        intermediate = image + self.beta * spatial
        feed_forward = self.conv5(self.sg(self.conv4(self.norm2(intermediate))))
        return intermediate + self.gamma * feed_forward


class NAFNet(nn.Module):
    def __init__(self, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2),
                 attention='sca'):
        super().__init__()
        if (width < 1 or middle_blk_num < 1 or len(enc_blk_nums) != len(dec_blk_nums)
                or not enc_blk_nums or any(n < 1 for n in (*enc_blk_nums, *dec_blk_nums))):
            raise ValueError('Require positive width/block counts and matching encoder/decoder levels')
        if attention not in ('sca', 'spatial', 'channel', 'combined'):
            raise ValueError(f"Unknown attention type: {attention}; choose 'sca', 'spatial', 'channel', or 'combined'")
        self.attention = attention
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels, attention=attention) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(*(NAFBlock(channels, attention=attention) for _ in range(middle_blk_num)))
        for count in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels, attention=attention) for _ in range(count))))
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, image):
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, (-width) % self.padder_size, 0, (-height) % self.padder_size))
        features, skips = self.intro(padded), []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle_blks(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        return (self.ending(features) + padded)[..., :height, :width]
