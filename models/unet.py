"""U-Net adapted for RGB regression: padded convolutions, no batch norm,
four scales, concatenation skips, bilinear upsampling and global residual.
This is a denoising baseline, not the original segmentation network.
"""
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        if width < 1:
            raise ValueError('width must be positive')
        channels = [width * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([double_conv(3, channels[0])])
        self.encoders.extend(double_conv(a, b) for a, b in zip(channels, channels[1:]))
        self.decoders = nn.ModuleList(
            double_conv(channels[i + 1] + channels[i], channels[i])
            for i in reversed(range(4))
        )
        self.output = nn.Conv2d(width, 3, 1)

    def forward(self, image):
        height, width = image.shape[-2:]
        x = F.pad(image, (0, (-width) % 16, 0, (-height) % 16))
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return image + self.output(x)[..., :height, :width]
