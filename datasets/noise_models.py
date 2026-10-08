"""
Noise Simulation Models:
1. Additive White Gaussian Noise (AWGN) - Standard benchmark noise
2. Poisson Noise (Photon / Shot Noise) - Low light / sensor photon noise
3. Salt & Pepper (Impulse) Noise
4. Heteroscedastic Poisson-Gaussian Sensor Noise Model (Unprocessed RAW noise)
5. Synthesized Real-World ISP Noise Pipeline
"""

import torch
import numpy as np


class NoiseInjector:
    """Configurable noise injection for PyTorch tensors in range [0, 1]"""

    @staticmethod
    def add_gaussian_noise(img, sigma=25.0):
        """
        Add Additive White Gaussian Noise (AWGN).
        Args:
            img: Tensor of shape (B, C, H, W) or (C, H, W) in [0, 1]
            sigma: standard deviation in [0, 255] or float
        """
        sigma_norm = sigma / 255.0 if sigma > 1.0 else sigma
        noise = torch.randn_like(img) * sigma_norm
        noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        return noisy_img

    @staticmethod
    def add_blind_gaussian_noise(img, sigma_min=5.0, sigma_max=50.0):
        """
        Add AWGN with random sigma per image in batch.
        """
        b = img.shape[0] if img.dim() == 4 else 1
        sigmas = torch.empty(b, 1, 1, 1, device=img.device).uniform_(sigma_min / 255.0, sigma_max / 255.0)
        noise = torch.randn_like(img) * sigmas
        return torch.clamp(img + noise, 0.0, 1.0)

    @staticmethod
    def add_poisson_noise(img, peak=30.0):
        """
        Add Poisson (Shot) noise.
        Formula: y = Poisson(x * peak) / peak
        Lower peak = stronger noise.
        """
        # img in [0, 1]
        img_clamped = torch.clamp(img, 1e-6, 1.0)
        vals = img_clamped * peak
        # Approximate Poisson with Gaussian for differentiable/fast GPU simulation or use torch.poisson
        noisy_vals = torch.poisson(vals)
        noisy_img = torch.clamp(noisy_vals / peak, 0.0, 1.0)
        return noisy_img

    @staticmethod
    def add_salt_and_pepper_noise(img, amount=0.05, s_vs_p=0.5):
        """
        Add Salt and Pepper impulse noise.
        """
        noisy_img = img.clone()
        rand = torch.rand_like(img)
        # Salt
        noisy_img[rand < (amount * s_vs_p)] = 1.0
        # Pepper
        noisy_img[(rand >= (amount * s_vs_p)) & (rand < amount)] = 0.0
        return noisy_img

    @staticmethod
    def add_poisson_gaussian_sensor_noise(img, shot_noise_scale=0.05, read_noise_scale=0.02):
        """
        Heteroscedastic Poisson-Gaussian Model (Foi et al., 2008).
        Variance of pixel y at true intensity x is:
        Var(y|x) = \beta_1 * x + \beta_2
        where \beta_1 is shot noise gain and \beta_2 is read noise variance.
        """
        variance = shot_noise_scale * torch.clamp(img, 0.0, 1.0) + (read_noise_scale ** 2)
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0.0, 1.0)

    @staticmethod
    def add_realistic_isp_noise(img, iso=1600):
        """
        Simulates camera ISP noise by adding heteroscedastic noise in linear space
        followed by a non-linear tone curve & color channel correlation.
        """
        # Approximate inverse gamma: sRGB -> Linear
        linear = torch.pow(torch.clamp(img, 1e-6, 1.0), 2.2)

        # Scale noise with ISO
        shot = 0.01 * (iso / 100.0) ** 0.5
        read = 0.005 * (iso / 100.0)
        noisy_linear = NoiseInjector.add_poisson_gaussian_sensor_noise(linear, shot, read)

        # Re-apply gamma: Linear -> sRGB
        noisy_srgb = torch.pow(torch.clamp(noisy_linear, 0.0, 1.0), 1.0 / 2.2)
        return torch.clamp(noisy_srgb, 0.0, 1.0)
