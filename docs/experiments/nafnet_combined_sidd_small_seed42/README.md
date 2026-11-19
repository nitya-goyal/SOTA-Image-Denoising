# NAFNet with combined channel and spatial attention: SIDD Small

Completed 20,000 optimizer steps on the internal SIDD Small sRGB split (seed 42; 125 train, 20 validation and 15 test pairs, grouped by physical scene). This controlled candidate replaces NAFNet's Simplified Channel Attention with CBAM-style channel attention and then applies CBAM-style spatial attention. The channel module uses shared average/max pooled MLP branches, reduction 16, ReLU and sigmoid; the spatial module uses channel-wise mean/max, a 7x7 convolution and sigmoid. This is an ablation candidate, not a claim of architectural novelty or a reproduction of published scores.

## Training protocol

Width 32; encoder blocks [2,2,4,8], middle 12, decoder [2,2,2,2]; batch 2; 128x128 training patches; L1 loss; AdamW with betas 0.9/0.999 and zero weight decay; gradient clipping 1.0; cosine learning rate 0.0002 to 0.000001; float32, AMP disabled. Validation during training used deterministic 256x256 center crops every 1,000 steps. The run metadata records RTX 4050 Laptop GPU, PyTorch 2.5.1+cu121, 125 training pairs and 20 validation pairs. The complete validation history and configuration are in this folder.

| Checkpoint | Step | Validation center-crop PSNR | Validation center-crop SSIM |
| --- | ---: | ---: | ---: |
| Best (`best.pt`) | 16,000 | 35.1025 dB | 0.89347 |
| Final (`last.pt`) | 20,000 | 35.0918 dB | 0.89372 |

The best checkpoint was selected by center-crop validation PSNR. These are patch metrics logged during training; no separate full-image tiled evaluation or test evaluation is recorded for this variant. In particular, no test score is reported here.

## Checkpoints

The full checkpoints are available in the [GitHub release](https://github.com/nitya-goyal/SOTA-Image-Denoising/releases/tag/sidd-small-combined-attention-v1), with a manifest containing steps, configurations and SHA-256 hashes. Their original local paths were `runs/nafnet_combined_sidd_small_seed42/best.pt` and `runs/nafnet_combined_sidd_small_seed42/last.pt`. Dataset images are not bundled. For training continuation, keep the complete run folder (including history and metadata) and the original local dataset; a checkpoint download alone does not include the dataset.
