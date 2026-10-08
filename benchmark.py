"""
Comprehensive Benchmark Comparison Suite:
Compares SOTA Deep Learning Models (Restormer, NAFNet, DnCNN) vs Classical Baselines (BM3D, NLM, Bilateral, Wavelet).
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from models import NAFNet, nafnet_tiny, Restormer, restormer_tiny, DnCNN
from models.baselines import denoise_bm3d, denoise_nlm, denoise_bilateral, denoise_wavelet_skimage, denoise_tv
from datasets import download_kodak24, NoiseInjector
from metrics import calculate_psnr, calculate_ssim


def benchmark_suite(test_dir="./data/Kodak24", noise_sigma=25.0, max_images=6):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Benchmark on {device.upper()} (Gaussian Sigma={noise_sigma}) ---")

    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        download_kodak24(test_dir)

    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_images]
    assert len(images) > 0, "No test images found"

    # Deep learning models to test
    dl_models = {
        'NAFNet (Lite)': nafnet_tiny(width=16).to(device).eval(),
        'Restormer (Lite)': restormer_tiny(dim=24).to(device).eval(),
        'DnCNN': DnCNN(in_channels=3, out_channels=3).to(device).eval()
    }

    # Classical methods
    classical_methods = {
        'BM3D / NLM-Color': lambda img: denoise_bm3d(img, sigma_psd=noise_sigma/255.0),
        'Bilateral Filter': lambda img: denoise_bilateral(img, d=7, sigma_color=noise_sigma/255.0),
        'Wavelet (BayesShrink)': lambda img: denoise_wavelet_skimage(img),
        'Total Variation (TV)': lambda img: denoise_tv(img, weight=0.08)
    }

    results = []

    # 1. Evaluate DL Models
    for name, model in dl_models.items():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        psnr_list, ssim_list, latency_list = [], [], []

        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            clean_tensor = TF.to_tensor(img_pil).unsqueeze(0)
            noisy_tensor = NoiseInjector.add_gaussian_noise(clean_tensor, sigma=noise_sigma)

            start = time.time()
            with torch.no_grad():
                pred = model(noisy_tensor.to(device)).cpu()
                pred = torch.clamp(pred, 0.0, 1.0)
            elapsed = (time.time() - start) * 1000

            psnr_list.append(calculate_psnr(pred[0], clean_tensor[0]))
            ssim_list.append(calculate_ssim(pred[0], clean_tensor[0]))
            latency_list.append(elapsed)

        results.append({
            'Method': name,
            'Type': 'Deep Learning (SOTA)',
            'Params (M)': f"{params:.2f}",
            'Avg PSNR (dB)': f"{np.mean(psnr_list):.2f}",
            'Avg SSIM': f"{np.mean(ssim_list):.4f}",
            'Latency (ms)': f"{np.mean(latency_list):.1f}"
        })

    # 2. Evaluate Classical Methods
    for name, func in classical_methods.items():
        psnr_list, ssim_list, latency_list = [], [], []

        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            clean_np = np.array(img_pil, dtype=np.float32) / 255.0
            
            # Inject noise on numpy
            noise = np.random.normal(0, noise_sigma/255.0, clean_np.shape).astype(np.float32)
            noisy_np = np.clip(clean_np + noise, 0.0, 1.0)

            start = time.time()
            pred_np = func(noisy_np)
            elapsed = (time.time() - start) * 1000

            psnr_list.append(calculate_psnr(pred_np, clean_np))
            ssim_list.append(calculate_ssim(pred_np, clean_np))
            latency_list.append(elapsed)

        results.append({
            'Method': name,
            'Type': 'Classical / Handcrafted',
            'Params (M)': '0.00 (Rule-based)',
            'Avg PSNR (dB)': f"{np.mean(psnr_list):.2f}",
            'Avg SSIM': f"{np.mean(ssim_list):.4f}",
            'Latency (ms)': f"{np.mean(latency_list):.1f}"
        })

    # Print Table
    print("\n" + "="*85)
    print(f"{'Method':<25} | {'Type':<22} | {'Params (M)':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency':<8}")
    print("="*85)
    for r in results:
        print(f"{r['Method']:<25} | {r['Type']:<22} | {r['Params (M)']:<10} | {r['Avg PSNR (dB)']:<10} | {r['Avg SSIM']:<8} | {r['Latency (ms)'] + ' ms':<8}")
    print("="*85)
    return results


if __name__ == '__main__':
    benchmark_suite()
