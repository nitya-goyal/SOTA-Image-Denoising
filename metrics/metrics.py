"""RGB float metrics on [0,1], no quantization, no border crop.
SSIM: Gaussian 11x11 (sigma 1.5), population covariance, valid region.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def image_metrics(prediction, target):
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 11:
        raise ValueError('Metrics require matching HWC RGB arrays at least 11x11')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite image values')
    mse = np.mean((a - b)**2)
    psnr = float('inf') if mse == 0 else float(-10 * np.log10(mse))
    blur = lambda x: gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=3.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a) - ma*ma, blur(b*b) - mb*mb, blur(a*b) - ma*mb
    ssim = ((2*ma*mb + 0.01**2) * (2*cov + 0.03**2)) / (
        (ma*ma + mb*mb + 0.01**2) * (va + vb + 0.03**2))
    return {'psnr': psnr, 'ssim': float(ssim[5:-5, 5:-5].mean())}
