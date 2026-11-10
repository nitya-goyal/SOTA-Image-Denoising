# Standard NAFNet on SIDD Small

Completed preliminary reference experiment, published 2026-11-10. These are
internal physical-scene-disjoint results, not official SIDD benchmark scores
or reproduction of the original paper's training recipe.

## Method and training

Standard width-32 NAFNet: encoder blocks [2,2,4,8], 12 middle blocks, decoder
blocks [2,2,2,2], 29,159,715 parameters. Existing Simplified Channel Attention
and SimpleGate are retained; no additional attention modules are included.

SIDD Small contains 160 real noisy/clean sRGB pairs. Preserved split: 125 training
pairs across eight physical scenes, 20 validation pairs from scene 08, and 15
test pairs from scene 04. See `split_info.json` for identities and hashes.

Training completed 20,000 optimizer steps at seed 42 with batch 2, 128px random
patches, L1 loss, AdamW (betas 0.9/0.999, zero weight decay), gradient clipping
1.0, cosine LR 0.0002 to 0.000001, and float32 with AMP disabled. Settings match
the U-Net reference experiment. Config: `../../../configs/nafnet_sidd_small_seed42.json`.

Validation during training used fixed 256px centre crops every 1,000 steps.
The best checkpoint was step 16,000: PSNR 35.469524 dB, SSIM 0.906028.
The final step scored 35.418148 dB / 0.905060. Full history is included.

## Saved-checkpoint evaluation

Both splits use the same validation-selected step-16,000 checkpoint. Settings:
256px tiles, 32px overlap, float32, four CPU threads, three timed repeats after
warm-up. Metrics use clipped float RGB, mean per-image PSNR and Gaussian-window
SSIM. Timing includes tiling and excludes loading, transfers, metrics and saving.

| Split | Pairs | Noisy PSNR | NAFNet PSNR | Noisy SSIM | NAFNet SSIM | Inference ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 20 | 26.467438 | 34.997759 | 0.555477 | 0.878366 | 9039.358 |
| Test | 15 | 28.391063 | 40.126959 | 0.522231 | 0.956597 | 8002.399 |

Recorded device: NVIDIA RTX 4050 Laptop GPU, PyTorch 2.5.1+cu121,
Python 3.10.10. The recorded Python version differs from current README advice.

Compared with the saved U-Net evaluations, NAFNet improves validation PSNR by
0.755619 dB and test PSNR by 0.695047 dB. It has higher measured inference latency.
Timings depend on device load and operating conditions; no controlled timing
rerun was performed for this report.

## Evidence and limitations

JSON summaries preserve full precision, configs, checkpoint hash and manifest
identities. The test summary extracts only NAFNet from the existing joint
evaluation. No training or evaluation was rerun to publish this record.

Checkpoint SHA-256:
`39071bfced6efcc0a8a23087d4369ad7eff885fd2c5f92b41da08a7e5d81c6b0`.
Weights remain outside Git at `runs/nafnet_sidd_small_seed42/best.pt`.
Source images, weights and generated comparisons are not bundled. Reproducing
these results requires the same checkpoint and data with matching manifests.

Single-seed results with one physical scene per held-out split provide limited
generalization evidence. Centre-crop training validation and tiled full-image
evaluation are distinct measurements. Test results have been inspected and are
not an untouched test for subsequent architecture development. SIDD Small and
Medium share scenes; a future protocol must account for this overlap.

## Validation evaluation command

From the repository root with the preserved data and checkpoint, use a fresh
output directory:

```powershell
python evaluate.py --manifest data/sidd_small_internal_seed42/val.csv --checkpoints runs/nafnet_sidd_small_seed42/best.pt --output results/nafnet_validation_new --device cuda --tile-size 256 --overlap 32 --repeats 3 --save-images 5
```

Architecture attribution: Chen et al., *Simple Baselines for Image Restoration*,
ECCV 2022; see `third_party/README.md`. Metric details are in
`docs/RESEARCH_PROTOCOL.md`.
