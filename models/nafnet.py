"""
NAFNet: Nonlinear Activation Free Network for Image Restoration (ECCV 2022)
Paper: "Simple Baselines for Image Restoration" - Liangyu Chen et al.
ArXiv: https://arxiv.org/abs/2204.04661
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - mean_g - y * mean_gy)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D Spatial Tensors (B, C, H, W)"""
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """
    SimpleGate: Replaces non-linear activation functions (ReLU, GELU).
    Splits the channel dimension into two halves and computes their elementwise product.
    Formula: f(X) = X_1 * X_2 where X = [X_1, X_2]
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA):
    Replaces standard SE-Block with channel pooling and a single linear projection.
    """
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (NAFBlock).
    Consists of Spatial Mixing (DWConv + SimpleGate + SCA) and Feed-Forward Network.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # Spatial mixing branch
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Feed-forward Network (FFN)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Spatial mixing
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        # Feed-Forward Network
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet Architecture for Image Denoising and Restoration.
    Hierarchical Encoder-Decoder with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.check_image_size(inp)

        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:, :, :x.shape[2], :x.shape[3]]

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


def nafnet_sidd(width=32, enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2]):
    """Standard NAFNet configuration optimized for SIDD / DND Real Denoising"""
    return NAFNet(in_channels=3, out_channels=3, width=width, middle_blk_num=4,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


def nafnet_tiny(width=16, enc_blks=[1, 1, 1, 1], dec_blks=[1, 1, 1, 1]):
    """Ultra-lightweight NAFNet for fast prototyping and low-memory training"""
    return NAFNet(in_channels=3, out_channels=3, width=width, middle_blk_num=1,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
