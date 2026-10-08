"""
Classical and Baseline Image Denoising Algorithms:
1. BM3D (Block-Matching and 3D Filtering) - Dabov et al., IEEE TIP 2007
2. Non-Local Means (NLM) - Buades et al., CVPR 2005
3. Bilateral Filter - Tomasi & Manduchi, ICCV 1998
4. Wavelet BayesShrink / VisuShrink Denoising
5. Total Variation (TV) Chambolle Denoising
"""

import numpy as np
import cv2

try:
    import bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False

try:
    from skimage.restoration import denoise_wavelet, denoise_tv_chambolle, denoise_nl_means, estimate_sigma
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def denoise_bm3d(img_np, sigma_psd=25/255.0):
    """
    BM3D (Block-Matching 3D Filtering)
    Args:
        img_np: float numpy array in range [0, 1], shape (H, W, 3) or (H, W)
        sigma_psd: noise standard deviation (e.g. 25/255.0)
    """
    if not HAS_BM3D:
        # Fallback to high quality Non-Local Means if bm3d package is not installed
        return denoise_nlm(img_np, h=sigma_psd)
    
    img_clamped = np.clip(img_np, 0.0, 1.0)
    denoised = bm3d.bm3d(img_clamped, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.clip(denoised, 0.0, 1.0)


def denoise_nlm(img_np, h=0.1, templateWindowSize=7, searchWindowSize=21):
    """
    Non-Local Means (NLM) Filter
    Explores non-local self-similarity in natural images.
    """
    img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    h_param = float(h * 255.0 * 0.5)
    
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8,
            None,
            h=h_param,
            hColor=h_param,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
    else:
        denoised = cv2.fastNlMeansDenoising(
            img_uint8,
            None,
            h=h_param,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
    return denoised.astype(np.float32) / 255.0


def denoise_bilateral(img_np, d=9, sigma_color=0.1, sigma_space=5.0):
    """
    Bilateral Filtering - Edge-preserving smoothing filter
    """
    img_float = np.clip(img_np, 0.0, 1.0).astype(np.float32)
    denoised = cv2.bilateralFilter(img_float, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    return np.clip(denoised, 0.0, 1.0)


def denoise_wavelet_skimage(img_np, method='BayesShrink', mode='soft'):
    """
    Wavelet Thresholding (BayesShrink / VisuShrink)
    """
    if not HAS_SKIMAGE:
        return denoise_bilateral(img_np)
    
    img_float = np.clip(img_np, 0.0, 1.0)
    channel_axis = -1 if (len(img_np.shape) == 3 and img_np.shape[2] == 3) else None
    denoised = denoise_wavelet(img_float, method=method, mode=mode, channel_axis=channel_axis, rescale_sigma=True)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


def denoise_tv(img_np, weight=0.1):
    """
    Total Variation (TV) Chambolle Denoising
    """
    if not HAS_SKIMAGE:
        return denoise_bilateral(img_np)
    
    channel_axis = -1 if (len(img_np.shape) == 3 and img_np.shape[2] == 3) else None
    denoised = denoise_tv_chambolle(np.clip(img_np, 0.0, 1.0), weight=weight, channel_axis=channel_axis)
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)
