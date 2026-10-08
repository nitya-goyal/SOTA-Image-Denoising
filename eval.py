"""
Evaluation Script for Denoising Models on Test Datasets (Kodak24, CBSD68, Set12, SIDD)
"""

import os
import argparse
import time
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models import NAFNet, nafnet_tiny, nafnet_sidd, Restormer, restormer_tiny, DnCNN
from datasets import download_kodak24, NoiseInjector
from metrics import calculate_psnr, calculate_ssim
from utils import tile_inference, save_comparison_grid


def load_model(model_name, weights_path=None, device='cuda'):
    name = model_name.lower()
    if name == 'nafnet':
        model = NAFNet(width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    elif name == 'nafnet_tiny':
        model = nafnet_tiny(width=16)
    elif name == 'nafnet_sidd':
        model = nafnet_sidd(width=32)
    elif name == 'restormer':
        model = Restormer(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8])
    elif name == 'restormer_tiny':
        model = restormer_tiny(dim=24, num_blocks=[2, 2, 2, 2], num_refinement_blocks=2, heads=[1, 2, 2, 4])
    elif name == 'dncnn':
        model = DnCNN(in_channels=3, out_channels=3)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Running with initialized/untrained model weights or eval mode.")

    model.to(device)
    model.eval()
    return model


def evaluate_dataset(model, test_dir, noise_sigma=25.0, output_dir="./results", use_tiling=False, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images.sort()

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0

    print(f"Evaluating {len(images)} images from {test_dir} with Gaussian sigma={noise_sigma}...")

    for img_name in tqdm(images):
        img_path = os.path.join(test_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        clean_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)

        # Inject noise
        noisy_tensor = NoiseInjector.add_gaussian_noise(clean_tensor, sigma=noise_sigma)

        # Inference
        start = time.time()
        with torch.no_grad():
            if use_tiling:
                pred_tensor = tile_inference(model, noisy_tensor, tile_size=256, tile_overlap=32, device=device)
            else:
                pred_tensor = model(noisy_tensor.to(device)).cpu()
                pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
        elapsed = time.time() - start

        # Metrics
        psnr_val = calculate_psnr(pred_tensor[0], clean_tensor[0])
        ssim_val = calculate_ssim(pred_tensor[0], clean_tensor[0])

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_time += elapsed

        # Save visual comparison
        out_save_path = os.path.join(output_dir, f"denoised_{img_name}")
        save_comparison_grid(
            noisy_tensor[0], pred_tensor[0], clean_tensor[0],
            save_path=out_save_path,
            psnr=psnr_val, ssim=ssim_val,
            roi_box=(30, 130, 30, 130) if clean_tensor.shape[2] >= 130 else None
        )

    avg_psnr = total_psnr / len(images)
    avg_ssim = total_ssim / len(images)
    avg_time = total_time / len(images)

    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY ({len(images)} images)")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Inference Time: {avg_time * 1000:.1f} ms / image")
    print(f"Results saved to: {output_dir}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nafnet_tiny')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default='./data/Kodak24')
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--output_dir', type=str, default='./results/eval')
    parser.add_argument('--tiling', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(args.test_dir) or len(os.listdir(args.test_dir)) == 0:
        download_kodak24(args.test_dir)

    model = load_model(args.model, args.weights, device=device)
    evaluate_dataset(model, args.test_dir, args.sigma, args.output_dir, args.tiling, device)
