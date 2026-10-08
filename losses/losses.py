"""
Loss Functions for Image Denoising and Restoration:
1. Charbonnier Loss (Standard in Restormer, NAFNet, MPRNet)
2. Frequency Domain (FFT) Loss
3. Edge Gradient Loss (Sobel / Laplacian)
4. Perceptual Feature Loss (VGG)
5. Compound Multi-Objective Denoising Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: A differentiable variant of L1 Loss.
    Formula: L(x, y) = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class FFTLoss(nn.Module):
    """
    Frequency Domain Loss using Fast Fourier Transform.
    Constrains spectral amplitude and phase differences.
    """
    def __init__(self, loss_weight=0.05):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        # Real and imaginary components
        loss = self.criterion(pred_fft.real, target_fft.real) + \
               self.criterion(pred_fft.imag, target_fft.imag)
        return self.loss_weight * loss


class EdgeGradientLoss(nn.Module):
    """
    Edge Gradient Loss using Sobel filters to preserve fine texture & structural edges.
    """
    def __init__(self, loss_weight=0.05):
        super(EdgeGradientLoss, self).__init__()
        self.loss_weight = loss_weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        gx = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        gy = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        return gx.view(b, c, h, w), gy.view(b, c, h, w)

    def forward(self, pred, target):
        gx_pred, gy_pred = self._gradient(pred)
        gx_target, gy_target = self._gradient(target)

        loss = F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
        return self.loss_weight * loss


class CompoundDenoisingLoss(nn.Module):
    """
    Comprehensive Loss combining Charbonnier + FFT + Edge Gradient loss.
    """
    def __init__(self, eps=1e-3, fft_weight=0.02, edge_weight=0.05):
        super(CompoundDenoisingLoss, self).__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.fft = FFTLoss(loss_weight=fft_weight)
        self.edge = EdgeGradientLoss(loss_weight=edge_weight)

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_fft = self.fft(pred, target)
        l_edge = self.edge(pred, target)
        total = l_char + l_fft + l_edge
        return total, {'char': l_char.item(), 'fft': l_fft.item(), 'edge': l_edge.item()}
