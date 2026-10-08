"""
Single Image Denoising Command-Line Utility:
Run denoising on any input image or folder.
"""

import os
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models import NAFNet, nafnet_tiny, Restormer, restormer_tiny, DnCNN
from utils import tile_inference, save_comparison_grid, save_difference_heatmap
from datasets import NoiseInjector


def denoise_file(input_path, output_path, model_name='nafnet_tiny', weights_path=None,
                 noise_sigma=None, use_tiling=False, save_comparison=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Model
    if model_name == 'nafnet':
        model = NAFNet(width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    elif model_name == 'nafnet_tiny':
        model = nafnet_tiny(width=16)
    elif model_name == 'restormer':
        model = Restormer(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8])
    elif model_name == 'restormer_tiny':
        model = restormer_tiny(dim=24, num_blocks=[2, 2, 2, 2], num_refinement_blocks=2, heads=[1, 2, 2, 4])
    elif model_name == 'dncnn':
        model = DnCNN(in_channels=3, out_channels=3)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from: {weights_path}")

    model.to(device).eval()

    # Load Image
    img_pil = Image.open(input_path).convert('RGB')
    orig_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)

    # Optional noise addition if testing with clean input
    if noise_sigma is not None and noise_sigma > 0:
        print(f"Simulating Gaussian noise with sigma={noise_sigma}...")
        noisy_tensor = NoiseInjector.add_gaussian_noise(orig_tensor, sigma=noise_sigma)
        input_for_model = noisy_tensor
    else:
        input_for_model = orig_tensor

    # Denoise
    with torch.no_grad():
        if use_tiling:
            denoised_tensor = tile_inference(model, input_for_model, tile_size=256, tile_overlap=32, device=device)
        else:
            denoised_tensor = model(input_for_model.to(device)).cpu()
            denoised_tensor = torch.clamp(denoised_tensor, 0.0, 1.0)

    # Save output image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    denoised_pil = TF.to_pil_image(denoised_tensor.squeeze(0))
    denoised_pil.save(output_path)
    print(f"Denoised image saved to: {output_path}")

    if save_comparison:
        comp_path = os.path.splitext(output_path)[0] + "_comparison.png"
        save_comparison_grid(
            input_for_model[0],
            denoised_tensor[0],
            orig_tensor[0] if noise_sigma is not None else None,
            save_path=comp_path,
            model_name=model_name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to noisy input image")
    parser.add_argument('--output', type=str, default="./results/output.png", help="Path to save denoised image")
    parser.add_argument('--model', type=str, default='nafnet_tiny', choices=['nafnet', 'nafnet_tiny', 'restormer', 'restormer_tiny', 'dncnn'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--sigma', type=float, default=None, help="Inject synthetic Gaussian noise for testing (e.g. 25.0)")
    parser.add_argument('--tiling', action='store_true', help="Use overlapping tiling inference for large images")
    args = parser.parse_args()

    denoise_file(args.input, args.output, args.model, args.weights, args.sigma, args.tiling)
