# Image Denoising: U-Net vs NAFNet on SIDD

A focused research codebase for supervised **real smartphone image denoising**.
Input: a noisy sRGB image. Target: its aligned SIDD ground-truth image.
No synthetic noise is added during training or evaluation.

The research question is: **How do a conventional U-Net denoising baseline and
NAFNet compare in restoration quality, parameter count and inference time under
the same training budget?**

NAFNet achieved state-of-the-art results when published at ECCV 2022. This project
does not claim that NAFNet is the latest 2026 model or that this implementation
reproduces its published scores. Recent denoising papers and their different
settings are covered in [RESEARCH_AND_PAPERS.md](RESEARCH_AND_PAPERS.md).

## Start here

Use Python 3.11 or 3.12. Create a virtual environment and install PyTorch
for your hardware using the [official selector](https://pytorch.org/get-started/locally/).
Windows CPU example:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the environment's Python for the commands below (or activate the environment).
On macOS/Linux create it with `python3 -m venv .venv` and activate with
`source .venv/bin/activate`; install the hardware-appropriate PyTorch build.
CPU works for checks; sustained training benefits from a GPU.

```text
prepare_sidd.py        Original SIDD PNG pairing, scene splits, validation MAT export
train.py               Shared L1 training, reproducible sampling, checkpoint/resume
evaluate.py            Checkpoint-only comparison, RGB PSNR/SSIM, timings, CSV/JSON
denoise.py             Denoise one photograph using a trained checkpoint
models/                U-Net baseline and NAFNet
datasets/              Explicit paired-image CSV loader
metrics/               RGB PSNR and Gaussian-window SSIM
configs/               Matched laptop and standard-width experiment settings
tests/                 Data integrity, model gradients, metrics, pipeline tests
docs/                  Research protocol and verification notes
third_party/           NAFNet attribution and upstream license
```

## Minimal workflow

Install PyTorch for your hardware first, then `python -m pip install -r requirements.txt`.
Download **SIDD Medium sRGB** from the [official dataset page](https://abdokamel.github.io/sidd/)
and extract it without renaming the scene folders.

```bash
python prepare_sidd.py --source data/SIDD_Medium_Srgb --output data/sidd
python train.py --config configs/unet_laptop.json --data data/sidd --output runs/unet
python train.py --config configs/nafnet_laptop.json --data data/sidd --output runs/nafnet
python evaluate.py --manifest data/sidd/test.csv --checkpoints runs/unet/best.pt runs/nafnet/best.pt --output results/comparison
```

These commands use a **custom physical-scene holdout**, not the official SIDD test
benchmark. For comparison with papers, use the official validation-block workflow
below. Do not train on validation blocks. Official benchmark test
ground truth is not supplied by this repo.

## Existing SIDD data and official validation

The source path can be anywhere on your laptop. Preparation searches recursively
for original SIDD folders such as `0001_001_S6_00100_00060_3200_L`, containing
`NOISY_SRGB_010.PNG` and `GT_SRGB_010.PNG` (and the 011 pair).
Keep original folder names: the second field identifies the physical scene.
Pairing is by filename identity, not independently sorted image lists.
The source must be sRGB PNG pairs; RAW data and benchmark noisy-only data are
not supported by this supervised workflow. CSV paths are relative to their
manifest directory when possible; Windows paths on another drive are absolute.
Move data together or regenerate manifests locally.

For the official train/validation protocol, provide the original validation
MAT files. They must contain uint8 variables `ValidationNoisyBlocksSrgb` and
`ValidationGtBlocksSrgb`, shaped (scenes, blocks, H, W, 3).

```bash
python prepare_sidd.py --source path/to/existing/SIDD --output data/sidd_official --protocol official --val-noisy path/to/ValidationNoisyBlocksSrgb.mat --val-clean path/to/ValidationGtBlocksSrgb.mat
```

This produces `train.csv` and `val.csv`, with all discovered Medium pairs used
for training and the validation blocks exported as PNGs. Both MAT arrays are
loaded into RAM once during preparation. Train with `--data data/sidd_official`
and evaluate `--manifest data/sidd_official/val.csv`. There is no `test.csv` for
this protocol. Label these scores as validation; they are not hidden-test scores.

## Smoke run, resume and single-photo inference

```bash
python train.py --config configs/unet_laptop.json --data data/sidd --output runs/unet_smoke --stop-after 2 --device cpu
python train.py --config configs/nafnet_laptop.json --data data/sidd --output runs/nafnet_smoke --stop-after 2 --device cpu
python train.py --config configs/unet_laptop.json --data data/sidd --output runs/unet_smoke --resume runs/unet_smoke/last.pt
python denoise.py --input path/to/noisy.png --checkpoint runs/nafnet/best.pt --output results/denoised.png
```

`--stop-after` is an absolute total step and preserves the original LR schedule.
Resume requires the same config/manifests and complete original run folder.
Unexpected interruption loses work since the last validation checkpoint. Each
training step samples pairs with replacement, with identical crop/augmentation
sequences for both models at the same seed and step. Validation uses a fixed
center crop. Reading original full-resolution PNGs before cropping can be slow.
Use SSD storage. Device selection is `auto` (CUDA, then Apple MPS, then CPU),
or force `--device cpu`. CUDA/MPS require a compatible PyTorch build and hardware.

Outputs must be empty/new for a fresh run or evaluation, to avoid mixing results.
If memory is limited, use laptop configs or lower batch/patch size for BOTH
models in new configs; use tiled inference for evaluation. A two-step smoke run
is only a software check. It is not a trained research model.

## Models and experiment scope

| Config | Role | Width | Encoder / middle / decoder |
| --- | --- | --- | --- |
| `unet_laptop.json` | U-Net adapted for denoising | 16 | Four downsampling scales, double conv + ReLU, concat skips |
| `nafnet_laptop.json` | Reduced NAFNet for accessible experiments | 16 | [1,1,1,1] / 1 / [1,1,1,1] |
| `unet_standard.json` | Larger U-Net baseline | 32 | Same baseline structure |
| `nafnet_standard.json` | Official SIDD width-32 NAFNet block configuration | 32 | [2,2,4,8] / 12 / [2,2,2,2] |

All supplied configs share the same step budget, input patches, batch size,
optimizer, loss, seed and validation procedure. Equal width does not mean equal
parameter count or compute. The laptop NAFNet is a smaller variant, so label it
**NAFNet-small (ours, reduced configuration)** in a report.

The standard config matches the upstream base architecture's block counts, but
uses our controlled training recipe, not the original multi-GPU recipe. We use
base NAFNet global pooling, without upstream NAFNetLocal/TLC conversion.

## Outputs and reporting

- `runs/<name>/config.json`, `run_info.json`, `history.csv`, `best.pt`, `last.pt`.
- `results/comparison/summary.json`: measured metrics, parameters, synchronized
  inference timings, checkpoint/manifest hashes and training configurations.
- `per_image.csv` and side-by-side PNGs: noisy / prediction / ground truth.

Report internal test, official validation and official benchmark results separately.
Metrics use float RGB [0,1], clipped predictions, no extra border crop, mean of
per-image PSNRs, and 11x11 Gaussian SSIM (sigma 1.5, population covariance, valid
region). Timings use float32 inference after warm-up and exclude image loading,
host-to-device transfer, metrics and saving. Tile processing is included.

Tiled inference defaults to 256px with 32px overlap to bound activation memory.
It changes pooling context; use the same settings for both models and report them.
Use `--tile-size 0` for full-image evaluation when memory allows. Center-crop
validation selects checkpoints; final evaluation uses every full manifest image.

**No trained weights or SIDD research results are bundled.** Software smoke tests
are not evidence of denoising performance. See [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md)
before writing the results section.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## References and attribution

- [SIDD: A High-Quality Denoising Dataset for Smartphone Cameras (CVPR 2018)](https://abdokamel.github.io/sidd/)
- [NAFNet: Simple Baselines for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)](https://arxiv.org/abs/1505.04597)

NAFNet implementation adapted from [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet).
See [third_party/README.md](third_party/README.md) and its upstream license.
