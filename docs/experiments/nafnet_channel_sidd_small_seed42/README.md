# NAFNet with CBAM-style channel attention: SIDD Small

Completed 20,000-step controlled candidate using the internal SIDD Small sRGB
physical-scene split (125 train / 20 validation / 15 test pairs), seed 42. The
variant replaces NAFNet Simplified Channel Attention (SCA) with CBAM-style
channel attention: shared two-layer 1x1 MLP (reduction 16, ReLU), global average
and max pooling, summed logits, sigmoid, and feature multiplication. It does not
add spatial attention. This is a controlled ablation candidate, not a claim of
architectural novelty or reproduction of a published score.

## Training protocol

Width 32; encoder blocks [2,2,4,8], middle 12, decoder [2,2,2,2]; 20,000
optimizer steps; batch 2; 128x128 training patches; L1; AdamW, betas 0.9/0.999,
zero weight decay; gradient clip 1.0; cosine LR 0.0002 to 0.000001; float32,
AMP disabled. Validation during training uses deterministic 256x256 center
crops. Full step history and run metadata are in this folder.

The best checkpoint was selected at step 16,000 by validation center-crop PSNR.
The final checkpoint is step 20,000. At step 16,000, center-crop validation
PSNR/SSIM were 35.1578 dB / 0.89599. At step 20,000 they were 35.1537 dB /
0.89632. These patch validation figures are not the tiled full-image metrics.

## Full-image test evaluation

| Split | Pairs | PSNR | SSIM | Noisy-input PSNR | Noisy-input SSIM | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal test | 15 | 40.0411 dB | 0.95640 | 28.3911 dB | 0.52223 | 54,484.7 ms/image |

Evaluation used the best checkpoint (step 16,000), 256px tiles, 32px overlap,
float32, three timing repeats, four CPU threads, and mean per-image PSNR plus
Gaussian valid-region SSIM. Detailed per-image metrics, environment, manifest
hash, and checkpoint hash are in `test_summary.json` and `test_per_image.csv`.
The test split has already been inspected as part of this project; treat these
numbers as internal single-seed results, not official SIDD benchmark results.

## Checkpoints

The full `best.pt` and `last.pt` files are available from the GitHub release
[SIDD Small NAFNet channel-attention checkpoint](https://github.com/nitya-goyal/SOTA-Image-Denoising/releases/tag/sidd-small-channel-attention-v1).
The release manifest gives each file's step, configuration, and SHA-256. In the
local project, their original paths were
`runs/nafnet_channel_sidd_small_seed42_v2/best.pt` and
`runs/nafnet_channel_sidd_small_seed42_v2/last.pt`. Dataset images are not
included. For exact training continuation, use the matching code/configuration
and retain the full run folder, not only a checkpoint file.
