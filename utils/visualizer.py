"""
Visualizer and Result Generator:
1. Side-by-side comparison (Noisy vs Denoised vs Clean)
2. Interactive/Zoomed-in Region of Interest (ROI) with bounding boxes
3. Difference heatmap generation (Error residue)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def tensor_to_np(tensor):
    """Convert (C, H, W) or (1, C, H, W) float tensor [0, 1] to (H, W, C) numpy [0, 1]"""
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        np_img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    else:
        np_img = tensor
    return np.clip(np_img, 0.0, 1.0)


def save_comparison_grid(noisy, denoised, clean=None, save_path="./comparison.png",
                         psnr=None, ssim=None, model_name="NAFNet", roi_box=None):
    """
    Plots a multi-panel comparison with zoomed-in inset.
    roi_box: tuple of (y1, y2, x1, x2) for zoom-in crop
    """
    noisy_np = tensor_to_np(noisy)
    denoised_np = tensor_to_np(denoised)
    clean_np = tensor_to_np(clean) if clean is not None else None

    cols = 3 if clean_np is not None else 2
    fig, axes = plt.subplots(2 if roi_box else 1, cols, figsize=(cols * 5, 8 if roi_box else 5))
    if not roi_box:
        axes = np.expand_dims(axes, 0)

    # Panel 1: Noisy
    axes[0, 0].imshow(noisy_np)
    axes[0, 0].set_title("Input (Noisy)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Denoised
    title_denoised = f"Restored ({model_name})"
    if psnr is not None:
        title_denoised += f"\nPSNR: {psnr:.2f} dB"
    if ssim is not None:
        title_denoised += f" | SSIM: {ssim:.4f}"
    axes[0, 1].imshow(denoised_np)
    axes[0, 1].set_title(title_denoised, fontsize=13, fontweight='bold', color='darkgreen')
    axes[0, 1].axis('off')

    # Panel 3: Clean (if present)
    if clean_np is not None:
        axes[0, 2].imshow(clean_np)
        axes[0, 2].set_title("Ground Truth (Clean)", fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')

    # ROI zoom-in row
    if roi_box:
        y1, y2, x1, x2 = roi_box
        
        # Draw bounding boxes on top row
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        rect2 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 1].add_patch(rect2)
        if clean_np is not None:
            rect3 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 2].add_patch(rect3)

        # Bottom row: Zoomed In
        axes[1, 0].imshow(noisy_np[y1:y2, x1:x2])
        axes[1, 0].set_title("Zoom-in Noisy ROI", fontsize=11)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(denoised_np[y1:y2, x1:x2])
        axes[1, 1].set_title("Zoom-in Restored ROI", fontsize=11, color='darkgreen')
        axes[1, 1].axis('off')

        if clean_np is not None:
            axes[1, 2].imshow(clean_np[y1:y2, x1:x2])
            axes[1, 2].set_title("Zoom-in Clean ROI", fontsize=11)
            axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to: {save_path}")


def save_difference_heatmap(denoised, clean, save_path="./error_heatmap.png"):
    """
    Saves a normalized L1 error residue heatmap between denoised and clean images.
    """
    den_np = tensor_to_np(denoised)
    clean_np = tensor_to_np(clean)
    error = np.mean(np.abs(den_np - clean_np), axis=-1)  # (H, W)

    plt.figure(figsize=(6, 5))
    plt.imshow(error, cmap='turbo')
    plt.colorbar(label='Absolute Error Magnitude')
    plt.title('Denoising Error Residue Map (Lower is Better)')
    plt.axis('off')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
