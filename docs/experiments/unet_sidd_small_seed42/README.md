# U-Net on SIDD Small: preliminary reference experiment

This record documents an existing completed experiment; publication date is
2026-11-10. It is an internal physical-scene-disjoint evaluation, not an official
SIDD benchmark result or reproduction of published scores.

## Training

- Model: denoising U-Net, width 32, 7,846,723 parameters.
- Dataset: SIDD Small sRGB, 160 noisy/clean pairs; no synthetic noise.
- Split: 125 training pairs across eight physical scenes, 20 validation pairs
  from scene 08, and 15 test pairs from scene 04. See `split_info.json` for hashes.
- Seed 42; 20,000 optimizer steps; batch 2; random 128px training patches.
- L1 loss; AdamW, betas 0.9/0.999, zero weight decay; gradient clipping 1.0.
- Cosine learning rate from 0.0002 to 0.000001; float32, AMP disabled.
- Configuration: `../../../configs/unet_sidd_small_seed42.json`.
- Recorded evaluation environment: RTX 4050 Laptop GPU, PyTorch 2.5.1+cu121,
  Python 3.10.10. The recorded Python version differs from current README advice.

Training validation used fixed 256px centre crops every 1,000 steps. The final
step scored 34.138826 dB PSNR / 0.866538 SSIM. Validation PSNR selected the
step-18,000 checkpoint (34.157220 dB / 0.867635); see `history.csv`.

## Saved-checkpoint evaluation

Both evaluations use the same validation-selected checkpoint: 256px tiles,
32px overlap, float32, three timed repeats after warm-up, four CPU threads.
Metrics use clipped float RGB, mean per-image PSNR, and Gaussian-window SSIM.
Timings exclude image loading, transfers, metrics, and saving, and include tiling.

| Split | Pairs | Noisy PSNR | U-Net PSNR | Noisy SSIM | U-Net SSIM | Mean inference ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 20 | 26.467438 | 34.242140 | 0.555477 | 0.853888 | 1256.451 |
| Test | 15 | 28.391063 | 39.431912 | 0.522231 | 0.949813 | 1364.672 |

Values are transcribed from locally saved evaluation JSONs. The accompanying
summaries preserve full precision, checkpoint hash, configuration, and manifest
identities. `test_summary.json` contains only the U-Net entry extracted from the
joint evaluation; no inference was rerun for this documentation commit.

Validation centre-crop and tiled full-image metrics measure different content
and must not be combined into one curve. Different scene difficulty can produce
higher test scores than validation scores. A single seed and one physical scene
per held-out split limit generalization conclusions. These test results have now
been inspected and must not be described as an untouched test for future tuning.

## Artifact locations and reproduction

The checkpoint remains outside Git at `runs/unet_sidd_small_seed42/best.pt`.
SHA-256: `89702ba3095848cb45cb92d468ac21acf3a1a3fa464813b29b9633c5a89b9404`.
Images, weights, original manifests, and generated comparisons are not bundled.
Reproducing the measurements requires the same checkpoint and original images
with matching manifests; this report alone is not a complete reproduction bundle.

From the repository root, using the preserved local data and checkpoint:

```powershell
python evaluate.py --manifest data/sidd_small_internal_seed42/val.csv --checkpoints runs/unet_sidd_small_seed42/best.pt --output results/unet_validation_new --device cuda --tile-size 256 --overlap 32 --repeats 3 --save-images 5
```

Use a new output folder for any rerun. Source definitions are in `models/unet.py`,
`train.py`, `evaluate.py`, and `docs/RESEARCH_PROTOCOL.md`.
