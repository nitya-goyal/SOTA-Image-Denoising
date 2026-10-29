# Image denoising research protocol

## Project and current stage

The project studies real-world image denoising and aims to investigate future
improvements. Current work establishes a reliable supervised SIDD pipeline,
a U-Net baseline and an implementation of NAFNet to understand an established
published method. The initial comparison supports learning and later research;
it is not the complete project contribution.

Preserve these baseline runs when introducing changes. Choose an improvement
from measured failure cases and literature, state a testable hypothesis, then
compare against the unchanged methods using ablations and repeated seeds.
The standard config follows the base width-32 NAFNet architecture, but our
common training recipe is not the original paper recipe. Published-result
reproduction remains a distinct task requiring matching training, data and
evaluation details, including the upstream inference choices.

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


## Official validation commands

If the original validation MAT files are available, prepare a separate folder:

```powershell
python prepare_sidd.py --source "D:/datasets/SIDD_Medium_Srgb" --output data/sidd_official --protocol official --val-noisy "D:/datasets/ValidationNoisyBlocksSrgb.mat" --val-clean "D:/datasets/ValidationGtBlocksSrgb.mat"
```

The expected uint8 variables are `ValidationNoisyBlocksSrgb` and
`ValidationGtBlocksSrgb`, shaped (scenes, blocks, H, W, 3). Both arrays are loaded
into RAM once for conversion. This uses all discovered Medium pairs for training
and separate validation blocks, creating train.csv and val.csv, not test.csv.
Use `--data data/sidd_official` for BOTH training runs, different run folders,
and `--manifest data/sidd_official/val.csv` for evaluation. A validation-selected
checkpoint's score on these blocks is official validation, not hidden-test
benchmark performance. Do not combine it with internal scene-holdout scores.
