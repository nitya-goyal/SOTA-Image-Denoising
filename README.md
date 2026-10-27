# Real-World Image Denoising

A research project on removing noise from real photographs while preserving
textures, edges and colours. We use paired noisy/clean sRGB images from SIDD.

**Current stage: implement and understand existing methods.** U-Net is our first
baseline; NAFNet is the stronger published method we study next. Comparing these
models is an initial experiment, not the final project title or contribution.
Later stages will investigate weaknesses, propose improvements and test them
against these preserved reference implementations.

## Research roadmap

1. Establish a reliable SIDD data, training and evaluation pipeline.
2. Train U-Net and the standard NAFNet architecture; understand the paper,
   inspect outputs, and document differences from the original recipe.
3. Study failure cases and recent work to choose a specific improvement.
4. Implement that improvement in a separate configuration/model and evaluate
   it with controlled comparisons, repeated seeds and ablations.

NAFNet achieved state-of-the-art results when published at ECCV 2022. Implementing
it is the first published-method step; the supplied common-budget recipe does
not reproduce its original training procedure or establish current SOTA results.
No trained weights or SIDD experiment scores are bundled. Recent papers and
citations are in [RESEARCH_AND_PAPERS.md](RESEARCH_AND_PAPERS.md).

## Run this stage in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/nitya-goyal/SOTA-Image-Denoising/blob/main/notebooks/Image_Denoising_Colab.ipynb)

The laptop does not need its own GPU. In Colab, choose **Runtime > Change runtime
type > GPU**, then run the notebook cells in order. GPU availability and runtime
limits vary; see the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

1. Put the **already downloaded SIDD Medium sRGB** folder or its ZIP archive in
   Google Drive. Do not download the dataset again or rename the original folders.
2. Mount Drive in the notebook and edit its dataset path and experiment name.
3. The notebook copies/extracts the data to the runtime's local disk for reading,
   validates pairs, prepares fixed splits, and runs a two-step sanity check.
4. Run the U-Net and NAFNet training cells in chunks. Configs, manifests,
   checkpoints and logs live in your Drive experiment folder. Rerun a training
   cell to continue from `last.pt` until the configured budget is reached.
5. Evaluate both models on the same held-out data and save metrics and comparison
   images to Drive. The notebook labels partial-budget runs as preliminary.

After a disconnect, reconnect, rerun setup/mount/data staging, select the SAME
experiment name and model size, then rerun training. Do not change the copied
configs or split settings for an existing run. Runtime `/content` files are
transient; only files saved in Drive persist. Checkpoint work since the last
save can be lost. The code records the revision so a resumed run uses the same
implementation. Notebook outputs include a download link for a small results ZIP.

Start with `MODEL_SIZE = "standard"` to study the published width-32 NAFNet base
architecture. If memory requires the smaller pair, use a new experiment name
with `MODEL_SIZE = "laptop"` and label NAFNet as a reduced variant. The notebook
supports both a custom scene holdout and official validation MAT files; official
validation results are not independent test results. Only the custom protocol
creates a local `test.csv`.

## Local setup (optional)

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
notebooks/            Colab setup, persistent checkpoints and staged training
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

## Local command-line workflow

Install PyTorch for your hardware first, then `python -m pip install -r requirements.txt`.
Point `--source` to your existing **SIDD Medium sRGB** extraction. Keep original
scene folders. If needed, the dataset is available from the
[official SIDD page](https://abdokamel.github.io/sidd/).

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
