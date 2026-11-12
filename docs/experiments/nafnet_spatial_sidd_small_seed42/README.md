# NAFNet with spatial attention: SIDD Small

Completed controlled candidate, published 2026-11-12. Existing simplified
channel attention (SCA) is retained. Each NAFBlock adds channel-wise mean/max
pooling, concatenation, a biased 7x7 convolution (padding 3), sigmoid, and feature
multiplication after SCA and before conv3. This adds 99 parameters per block:
29,163,279 total, 3,564 more than standard NAFNet across 36 blocks.

The spatial module is CBAM-style (Woo et al., *CBAM: Convolutional Block
Attention Module*, ECCV 2018, https://arxiv.org/abs/1807.06521). NAFNet is from
Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022; attribution
and license are in `third_party/`. Sigmoid reintroduces an activation, so this
variant does not retain NAFNet's activation-free property. This is an ablation
candidate, not a claim of architectural novelty or published-score reproduction.

## Protocol

SIDD Small sRGB, seed 42: 125 training, 20 validation (scene 08), 15 test
(scene 04) pairs. Physical scenes are disjoint; hashes are in `split_info.json`.
The configuration matches the reference training recipe except model identity:
width 32; encoder [2,2,4,8], middle 12, decoder [2,2,2,2]; 20,000 updates;
batch 2, 128px training patches, 256px validation centre crops; L1; AdamW with
betas 0.9/0.999 and zero weight decay; clipping 1.0; cosine LR 0.0002 to
0.000001; float32 with AMP disabled. Full history and config are preserved.

## Results

The evaluated best checkpoint is step 16,000, selected using validation
centre-crop PSNR. Final-step centre-crop validation is 35.415618 dB / 0.908950
SSIM; this differs from the tiled full-image results below.

| Full-image split | Pairs | Noisy PSNR | Spatial PSNR | Noisy SSIM | Spatial SSIM | Mean ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 20 | 26.467438 | 35.206797 | 0.555477 | 0.883848 | 26873.306 |
| Test | 15 | 28.391063 | 40.140854 | 0.522231 | 0.956735 | 11926.111 |

Inference uses 256px tiles with 32px overlap, float32, four CPU threads and
three repeats after warm-up. RGB metrics use clipped floats, mean per-image
PSNR and Gaussian SSIM. Device: RTX 4050 Laptop GPU; PyTorch 2.5.1+cu121,
Python 3.10.10. Timings exclude loading, transfers, metrics and saving.

Relative to standard NAFNet's saved evaluations, gains are 0.209039 dB on
validation and 0.013895 dB on test. These are single-seed measurements;
the tiny test gain does not establish reliable improvement. Timing increased,
but separate sessions did not control GPU load/power conditions. A controlled
timing comparison is needed before attributing the full difference to the module.

These are internal results, not official SIDD scores. The test set was already
inspected during this research series; do not use it to guide future variants
or describe it as untouched. Only one physical scene per held-out split and
one training seed limit generalization. Different model initialization streams
can also contribute to single-seed variation.

## Evidence

Summaries and history are copied from the completed local run without rerunning
training or inference. Checkpoint SHA-256:
`db628bce2d55f2055ca630960db2506bc82c613f8ad6504845e451e64dcc727b`.
Weights remain at `runs/nafnet_spatial_sidd_small_seed42/best.pt` outside Git.
Original dataset images and generated comparison PNGs are not bundled.

From the repository root with the original data and checkpoint:

```powershell
python evaluate.py --manifest data/sidd_small_internal_seed42/val.csv --checkpoints runs/nafnet_spatial_sidd_small_seed42/best.pt --output results/spatial_validation_new --device cuda --tile-size 256 --overlap 32 --repeats 3 --save-images 5
```

Use a fresh output directory. The original code and data identities must be
preserved for exact experimental reproduction. Channel-replacement and combined
variants are separate experiments and are excluded from this publication.
