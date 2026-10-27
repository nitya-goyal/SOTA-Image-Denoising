# Experiment protocol

## Question and scope

Task: supervised single-image sRGB denoising on SIDD. Compare an explicitly
defined U-Net regression baseline and NAFNet; neither is a novel architecture.
Use the reduced pair for laptop experiments, or the standard pair for the
main width-32 architecture study. Report which was used.

## Data

- Use original SIDD Medium sRGB pairs; no synthetic noise is added.
- `prepare_sidd.py` pairs by explicit NOISY_SRGB -> GT_SRGB filename identity.
- Internal protocol groups all instances of each physical scene, using the
  second folder-name field, before creating train/val/test manifests. This is
  stricter than randomly splitting the two shots of a scene instance.
- Official protocol trains on Medium and validates on the original validation
  blocks. It has no independent labeled test split in this repo.
- Preserve manifests and `split_info.json` for the experiment. Paths are relative
  to the CSV; hashes detect manifest changes, not changes to source image bytes.
- Only use the internal test split after model choices are fixed. Training
  validates on one fixed center crop per pair; final evaluation uses full images.

## Fair comparison

Hold the training data, sampled pair/crop/augmentation sequence, patch size,
batch size, optimizer, loss, precision and step budget fixed across models.
Each step samples pairs with replacement. Common settings improve control but
are not necessarily optimal hyperparameters for either architecture. Report
the selection procedure and do not claim they reproduce original-paper scores.

Use L1 loss, AdamW (betas 0.9/0.999, zero weight decay), cosine LR decay and
gradient clipping at 1.0. Record all config values and hardware/software versions.
Model widths are not parameter-matched; report actual parameter counts.

Select `best.pt` using validation RGB PSNR. `last.pt` stores optimizer, scheduler,
AMP scaler and torch RNG state for continuation. Sampling is deterministic by
seed and absolute step. Reproducibility across devices/versions is not promised.
Checkpoints are written at each validation and at a requested early stop.

For a stronger result use at least three independent training seeds with the
same fixed split; summarize variability. Split-seed experiments answer a
different question and should be reported separately. SIDD has few physical
scenes; do not imply broad camera or scene generalization from one split.

## Evaluation

- Real noisy images, unchanged; no noise regeneration between models.
- Mandatory strict checkpoint loading, architecture rebuilt from saved config.
- Float32 inference; predictions clipped to [0,1]. PNGs are for viewing only.
- RGB PSNR on [0,1], arithmetic mean of per-image PSNR values.
- RGB SSIM: 11x11 Gaussian, sigma 1.5, K1=0.01, K2=0.03, population covariance;
  average over channels and valid spatial windows. No extra border crop.
- No self-ensemble, test augmentation or local-pooling conversion.
- Record tiling; defaults are 256px tiles, 32px overlap, uniform overlap average.
  Tiles can change global attention/pooling behavior, so do not mix tiled and
  full-image results without labeling the inference difference.
- Inference timing: one warm-up per image, requested repeats, device sync before
  and after, excluding loading, transfers, metrics and saving; includes tiling.
- Per-image CSV includes the noisy baseline, dimensions and latency. JSON records
  checkpoint/manifest hashes, environment, configs and aggregate measurements.

Internal scene-holdout, official validation and official hidden-test benchmark
scores belong in separate tables. A validation-selected checkpoint's validation
score is not independent test performance. Inspect the original paper's metrics
and evaluation setup before comparing numbers.

## Results template (fill only after real experiments)

| Model/config | Protocol | Seed | Steps / batch / patch | Params | PSNR | SSIM | Inference ms | Device / tiles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Noisy input | pending | — | — | — | pending | pending | — | — |
| U-Net | pending | pending | pending | pending | pending | pending | pending | pending |
| NAFNet | pending | pending | pending | pending | pending | pending | pending | pending |

Show fixed representative crops of noisy/prediction/ground truth, including
fine textures, dark regions and failures. Avoid selecting only favorable images.
Store results outside Git and share evidence with the teammate separately.
