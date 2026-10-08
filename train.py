"""
Training Pipeline for SOTA Image Denoising Models (Restormer, NAFNet, DnCNN)
Includes AMP Mixed Precision, Cosine Annealing, Gradient Clipping, and Validation.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import NAFNet, nafnet_sidd, nafnet_tiny, Restormer, restormer_tiny, DnCNN
from datasets import SyntheticDenoisingDataset, PairedDenoisingDataset, generate_synthetic_samples
from losses import CharbonnierLoss, CompoundDenoisingLoss
from metrics import calculate_psnr, calculate_ssim
from utils import save_comparison_grid


def get_model(name, width=32):
    name = name.lower()
    if name == 'nafnet':
        return NAFNet(in_channels=3, out_channels=3, width=width, middle_blk_num=2,
                      enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    elif name == 'nafnet_tiny':
        return nafnet_tiny(width=16)
    elif name == 'nafnet_sidd':
        return nafnet_sidd(width=32)
    elif name == 'restormer':
        return Restormer(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8])
    elif name == 'restormer_tiny':
        return restormer_tiny(dim=24, num_blocks=[2, 2, 2, 2], num_refinement_blocks=2, heads=[1, 2, 2, 4])
    elif name == 'dncnn':
        return DnCNN(in_channels=3, out_channels=3, num_layers=17, num_features=64)
    else:
        raise ValueError(f"Unknown model name: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train SOTA Image Denoising Models")
    parser.add_argument('--model', type=str, default='nafnet_tiny', choices=['nafnet', 'nafnet_tiny', 'nafnet_sidd', 'restormer', 'restormer_tiny', 'dncnn'])
    parser.add_argument('--train_dir', type=str, default='./data/train_clean')
    parser.add_argument('--val_dir', type=str, default='./data/val_clean')
    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'blind_gaussian', 'poisson', 'sensor', 'isp'])
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Model: {args.model} | Noise: {args.noise_type} (sigma={args.sigma})")

    # If data doesn't exist, create synthetic samples for quick start
    if not os.path.exists(args.train_dir) or len(os.listdir(args.train_dir)) == 0:
        print(f"Training directory {args.train_dir} not found. Generating synthetic samples...")
        generate_synthetic_samples(args.train_dir, count=30)
    if not os.path.exists(args.val_dir) or len(os.listdir(args.val_dir)) == 0:
        generate_synthetic_samples(args.val_dir, count=8)

    # Datasets
    train_dataset = SyntheticDenoisingDataset(
        image_dir=args.train_dir,
        patch_size=args.patch_size,
        noise_type=args.noise_type,
        sigma=args.sigma,
        is_train=True
    )
    val_dataset = SyntheticDenoisingDataset(
        image_dir=args.val_dir,
        patch_size=args.patch_size,
        noise_type=args.noise_type,
        sigma=args.sigma,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model, Optimizer, Scheduler, Loss
    model = get_model(args.model).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params / 1e6:.2f} Million")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = CompoundDenoisingLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == 'cuda'))
    os.makedirs(args.save_dir, exist_ok=True)
    best_psnr = 0.0

    print("--- Starting Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == 'cuda')):
                pred = model(noisy)
                loss, loss_dict = criterion(pred, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for idx, val_batch in enumerate(val_loader):
                v_noisy = val_batch['noisy'].to(device)
                v_clean = val_batch['clean'].to(device)
                v_pred = model(v_noisy)
                v_pred = torch.clamp(v_pred, 0.0, 1.0)

                val_psnr += calculate_psnr(v_pred[0], v_clean[0])
                val_ssim += calculate_ssim(v_pred[0], v_clean[0])

                if idx == 0:
                    # Save sample image
                    save_comparison_grid(
                        v_noisy[0], v_pred[0], v_clean[0],
                        save_path=os.path.join(args.save_dir, f"epoch_{epoch}_sample.png"),
                        psnr=calculate_psnr(v_pred[0], v_clean[0]),
                        ssim=calculate_ssim(v_pred[0], v_clean[0]),
                        model_name=args.model
                    )

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] - Train Loss: {avg_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f}")

        # Save Best Model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': avg_ssim,
                'args': vars(args)
            }, best_path)
            print(f"==> Saved New Best Checkpoint: {best_path} (PSNR: {best_psnr:.2f} dB)")

    print(f"Training Complete! Best Validation PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
