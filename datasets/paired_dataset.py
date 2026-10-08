"""
Dataset Loaders for Image Denoising:
1. PairedImageDataset: For real-world paired data (e.g. SIDD, DND, PolyU, RENOIR)
2. SyntheticDenoisingDataset: Generates on-the-fly noisy-clean pairs from clean images (e.g. DIV2K, BSD500, Kodak)
"""

import os
import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .noise_models import NoiseInjector


class SyntheticDenoisingDataset(Dataset):
    """
    Takes clean images, applies random crops/augmentations, and injects noise dynamically.
    """
    def __init__(self, image_dir, patch_size=128, noise_type='gaussian', sigma=25.0,
                 sigma_min=5.0, sigma_max=50.0, is_train=True, max_samples=None):
        super().__init__()
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.PNG', '*.JPG']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        self.image_paths = sorted(self.image_paths)
        if max_samples and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]

        self.patch_size = patch_size
        self.noise_type = noise_type
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, clean_img):
        # clean_img: Tensor (C, H, W)
        if not self.is_train:
            return clean_img

        # Random horizontal flip
        if random.random() > 0.5:
            clean_img = TF.hflip(clean_img)
        # Random vertical flip
        if random.random() > 0.5:
            clean_img = TF.vflip(clean_img)
        # Random 90 deg rotation
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            clean_img = torch.rot90(clean_img, rot_k, [1, 2])
        return clean_img

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_pil = Image.open(path).convert('RGB')
        img_tensor = TF.to_tensor(img_pil)  # (3, H, W) in [0, 1]

        # Crop patch if training
        if self.patch_size is not None:
            _, h, w = img_tensor.shape
            if h >= self.patch_size and w >= self.patch_size:
                if self.is_train:
                    top = random.randint(0, h - self.patch_size)
                    left = random.randint(0, w - self.patch_size)
                else:
                    # Center crop for evaluation
                    top = (h - self.patch_size) // 2
                    left = (w - self.patch_size) // 2
                clean = img_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
            else:
                # Pad if smaller
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                clean = TF.pad(img_tensor, [0, 0, pad_w, pad_h], padding_mode='reflect')
        else:
            clean = img_tensor

        clean = self._augment(clean)

        # Inject noise
        if self.noise_type == 'gaussian':
            noisy = NoiseInjector.add_gaussian_noise(clean, self.sigma)
        elif self.noise_type == 'blind_gaussian':
            sigma_val = random.uniform(self.sigma_min, self.sigma_max)
            noisy = NoiseInjector.add_gaussian_noise(clean, sigma_val)
        elif self.noise_type == 'poisson':
            noisy = NoiseInjector.add_poisson_noise(clean)
        elif self.noise_type == 'sensor':
            noisy = NoiseInjector.add_poisson_gaussian_sensor_noise(clean)
        elif self.noise_type == 'isp':
            noisy = NoiseInjector.add_realistic_isp_noise(clean)
        else:
            noisy = clean.clone()

        return {'noisy': noisy, 'clean': clean, 'path': path}


class PairedDenoisingDataset(Dataset):
    """
    Loads paired noisy and ground-truth clean images from two folders.
    Useful for SIDD, DND, PolyU, RENOIR.
    """
    def __init__(self, noisy_dir, clean_dir, patch_size=128, is_train=True):
        super().__init__()
        self.noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.*')))
        self.clean_paths = sorted(glob.glob(os.path.join(clean_dir, '*.*')))
        assert len(self.noisy_paths) == len(self.clean_paths), "Mismatch between noisy and clean image counts"
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_pil = Image.open(self.noisy_paths[idx]).convert('RGB')
        clean_pil = Image.open(self.clean_paths[idx]).convert('RGB')

        noisy_tensor = TF.to_tensor(noisy_pil)
        clean_tensor = TF.to_tensor(clean_pil)

        if self.patch_size is not None:
            _, h, w = noisy_tensor.shape
            if h >= self.patch_size and w >= self.patch_size:
                if self.is_train:
                    top = random.randint(0, h - self.patch_size)
                    left = random.randint(0, w - self.patch_size)
                else:
                    top = (h - self.patch_size) // 2
                    left = (w - self.patch_size) // 2
                noisy = noisy_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
                clean = clean_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
            else:
                noisy = noisy_tensor
                clean = clean_tensor
        else:
            noisy = noisy_tensor
            clean = clean_tensor

        if self.is_train:
            # Sync augmentations
            if random.random() > 0.5:
                noisy = TF.hflip(noisy)
                clean = TF.hflip(clean)
            if random.random() > 0.5:
                noisy = TF.vflip(noisy)
                clean = TF.vflip(clean)

        return {'noisy': noisy, 'clean': clean, 'path': self.noisy_paths[idx]}
