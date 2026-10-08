"""
Utilities to download standard benchmarks or generate synthetic test suites.
Standard benchmarks:
- Kodak24: 24 pristine true-color photographic images (768x512)
- CBSD68: 68 color images from Berkeley Segmentation Dataset
- Set12: 12 classic grayscale/color testing images
- SIDD (Smartphone Image Denoising Dataset)
"""

import os
import urllib.request
import zipfile
import tarfile
import numpy as np
from PIL import Image, ImageDraw


KODAK_URLS = [
    f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)
]


def download_kodak24(dest_dir="./data/Kodak24"):
    """Downloads the standard Kodak24 image denoising benchmark dataset"""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Checking / downloading Kodak24 to {dest_dir}...")
    for url in KODAK_URLS:
        filename = os.path.basename(url)
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Could not download {url}: {e}")
    print(f"Kodak24 ready in {dest_dir} ({len(os.listdir(dest_dir))} files).")


def generate_synthetic_samples(dest_dir="./data/synthetic_clean", count=10):
    """
    Generates synthetic high-frequency pattern test images if offline or for quick unit tests.
    Includes textures, sharp geometric edges, gradients, and naturalistic shapes.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (256, 256), color=(i * 20 % 255, 120, 200))
        draw = ImageDraw.Draw(img)
        # Add high frequency patterns and shapes
        for j in range(0, 256, 16):
            draw.line([(0, j), (256, 256 - j)], fill=((j * 7) % 255, (j * 13) % 255, (j * 17) % 255), width=2)
            draw.ellipse([j, j, 256 - j, 256 - j], outline=((j * 11) % 255, 200, (j * 5) % 255), width=2)
            draw.rectangle([j + 10, j + 10, j + 30, j + 30], fill=((j * 23) % 255, 150, 80))
        
        # Add text for sharp edge testing
        draw.text((20, 20), f"SOTA Denoising Test Pattern #{i+1}", fill=(255, 255, 255))
        draw.text((20, 40), "High Frequency Edges & Details", fill=(0, 255, 255))

        img.save(os.path.join(dest_dir, f"sample_{i+1:03d}.png"))
    print(f"Generated {count} synthetic benchmark test images in {dest_dir}")
