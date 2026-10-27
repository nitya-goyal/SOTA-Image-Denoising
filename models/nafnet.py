"""NAFNet base architecture, adapted from megvii-research/NAFNet (ECCV 2022).
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


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, expanded, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.sg = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        spatial = self.sg(self.conv2(self.conv1(self.norm1(image))))
        spatial = self.conv3(spatial * self.sca(spatial))
        intermediate = image + self.beta * spatial
        feed_forward = self.conv5(self.sg(self.conv4(self.norm2(intermediate))))
        return intermediate + self.gamma * feed_forward


class NAFNet(nn.Module):
    def __init__(self, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)):
        super().__init__()
        if (width < 1 or middle_blk_num < 1 or len(enc_blk_nums) != len(dec_blk_nums)
                or not enc_blk_nums or any(n < 1 for n in (*enc_blk_nums, *dec_blk_nums))):
            raise ValueError('Require positive width/block counts and matching encoder/decoder levels')
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blk_num)))
        for count in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
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
