"""
Evaluation Metrics for Image Denoising:
1. PSNR (Peak Signal-to-Noise Ratio) - RGB & Y (Luminance) channel
2. SSIM (Structural Similarity Index) - Wang et al., IEEE TIP 2004
3. LPIPS (Learned Perceptual Image Patch Similarity)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2, data_range=1.0, crop_border=0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img1, img2: ndarray or torch.Tensor in range [0, data_range]
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    if crop_border > 0:
        if img1.ndim == 3:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
        elif img1.ndim == 2:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))


def _ssim_single(img1, img2, data_range=1.0):
    """Single channel or grayscale SSIM computation"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 11x11 Gaussian window with sigma 1.5
    kernel = cv2_gaussian_kernel(11, 1.5)

    mu1 = convolve2d(img1, kernel)
    mu2 = convolve2d(img2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = convolve2d(img2 * img2, kernel) - mu2_sq
    sigma12 = convolve2d(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cv2_gaussian_kernel(size=11, sigma=1.5):
    """Generate 2D Gaussian kernel"""
    coords = np.arange(size) - (size - 1) / 2
    g = np.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = np.outer(g, g)
    return kernel2d / kernel2d.sum()


def convolve2d(img, kernel):
    """Fast 2D convolution with reflection padding"""
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode='reflect')
    # Use scipy or OpenCV if available, or manual strided / scipy.signal.fftconvolve
    try:
        from scipy.signal import convolve2d as sp_conv
        return sp_conv(padded, kernel, mode='valid')
    except Exception:
        import cv2
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def calculate_ssim(img1, img2, data_range=1.0, crop_border=0):
    """
    Calculate SSIM (Structural Similarity Index Measure).
    Supports (H, W, 3), (H, W), or (C, H, W).
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Channel first to channel last if (C, H, W)
    if img1.ndim == 3 and img1.shape[0] in [1, 3]:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    if crop_border > 0:
        if img1.ndim == 3:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
        elif img1.ndim == 2:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    if img1.ndim == 3 and img1.shape[2] == 3:
        ssims = [_ssim_single(img1[:, :, c], img2[:, :, c], data_range) for c in range(3)]
        return float(np.mean(ssims))
    elif img1.ndim == 3 and img1.shape[2] == 1:
        return float(_ssim_single(img1[:, :, 0], img2[:, :, 0], data_range))
    else:
        return float(_ssim_single(img1, img2, data_range))


def rgb2ycbcr(img):
    """Convert RGB float image [0, 1] to YCbCr, returns Y channel in [0, 1]"""
    # ITU-R BT.601 conversion
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    return y


def calculate_psnr_y(img1, img2):
    """Calculate PSNR on Y (luminance) channel only (standard in classical benchmarks)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    y1 = rgb2ycbcr(img1)
    y2 = rgb2ycbcr(img2)
    return calculate_psnr(y1, y2)
