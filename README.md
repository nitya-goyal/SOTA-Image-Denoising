# Real-World Image Denoising

Research on removing noise from real photographs while preserving edges,
textures and colours. We use real paired SIDD sRGB images; no synthetic noise
is added to training or evaluation.

**Stage one:** understand and implement existing methods. U-Net is the initial
baseline; NAFNet is the stronger published method. These references support our
later investigation of improvements; they are not the final project title.
NAFNet achieved SOTA results in 2022. Our initial common-budget experiments do
not establish current SOTA or reproduce its original training recipe.

## Two-person workflow

Use your friend's **local GPU PC** as the main training machine. The repository
owner manages reviewed code/configs, papers and experiment analysis. The GPU
operator manages the local dataset, runs and backups. Both review the methods,
outputs and conclusions. Only the owner needs push access; the friend can clone
this public repo without collaborator access.

Keep source/configs in Git and data/checkpoints outside Git. Keep both baseline
runs unchanged when starting an improvement experiment. The work sequence and
what to show your supervisor are in [docs/WORK_PLAN.md](docs/WORK_PLAN.md).

## 1. Set up the GPU PC

The Acer Predator Helios Neo 16 name does not identify its exact GPU or VRAM.
Check the actual machine before choosing batch sizes or estimating runtime.
Use Python 3.11/3.12 and run from the repository folder.

Windows PowerShell:

```powershell
git clone https://github.com/nitya-goyal/SOTA-Image-Denoising.git
cd SOTA-Image-Denoising
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`;
no execution-policy change is required. On Linux, create the environment with
`python3 -m venv .venv` and activate with `source .venv/bin/activate`.

**Install the GPU-capable PyTorch build first** using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) for the
actual OS, GPU backend and driver. This repo does not assume a CUDA version.
A CPU-only installation cannot train on an NVIDIA GPU. AMD GPU support depends
on the supported OS/backend; do not install CUDA based only on the word GPU.

Then:

```bash
python -m pip install -r requirements.txt
python check_environment.py --require-gpu --output runs/hardware.json
```

The report records the actual device and available GPU memory and verifies
execution on that device. If it reports CPU, fix the installation/driver before
starting a long run. `--device cuda` below means a supported CUDA/ROCm PyTorch
GPU backend; use `--device mps` on a compatible Apple machine. GPU execution on
your friend's hardware still needs to be verified there.

## 2. Prepare the existing SIDD data

Place the existing **SIDD Medium sRGB** dataset on the GPU PC, preferably an SSD.
If a fresh copy is needed, use the [official SIDD source](https://abdokamel.github.io/sidd/).
Keep original scene folders and paired PNGs, e.g.
`0001_001_S6_00100_00060_3200_L/NOISY_SRGB_010.PNG` and `GT_SRGB_010.PNG`.
RAW MAT data and noisy-only benchmark test images are not training pairs.
The source can be on another drive; it is read, not modified. The complete
Medium set has 320 pairs. Missing partners or unrecognized folders are rejected.

```bash
python prepare_sidd.py --source "D:/datasets/SIDD_Medium_Srgb" --output data/sidd --protocol internal --seed 42
```

Replace the example source with the actual folder. The default internal
train/val/test split groups by physical scene so camera/ISO variants stay
together. Keep `split_info.json` and manifests unchanged for both models.
**Internal scores are not official SIDD benchmark scores.** Regenerate manifests
when moving the dataset; paths on a different Windows drive may be absolute.

For official training + validation, supply the original validation MAT files:

```bash
python prepare_sidd.py --source "D:/datasets/SIDD_Medium_Srgb" --output data/sidd_official --protocol official --val-noisy "D:/datasets/ValidationNoisyBlocksSrgb.mat" --val-clean "D:/datasets/ValidationGtBlocksSrgb.mat"
```

Expected variables are uint8 `ValidationNoisyBlocksSrgb` and
`ValidationGtBlocksSrgb`, shaped (scenes, blocks, H, W, 3). Conversion loads both
arrays into RAM. This mode uses Medium training pairs and separate validation
blocks, producing train.csv and val.csv, **no test.csv**. Use
`--data data/sidd_official` for both training runs. Label evaluation of val.csv
as official validation, not independent or hidden-test benchmark performance.

## 3. Check memory using real data

Run one disposable training step for each standard model before long training:

```bash
python check_environment.py --device cuda --require-gpu --config configs/unet_standard.json --data data/sidd --output runs/preflight_unet.json
python check_environment.py --device cuda --require-gpu --config configs/nafnet_standard.json --data data/sidd --output runs/preflight_nafnet.json
```

This checks forward/backward/optimizer allocation on real pairs and reports
peak allocated/reserved GPU memory. It produces no trained weights or research
metrics. A first-step timing is not a reliable full-run estimate. Actual
validation/inference memory must also be checked in a short run.

If memory is insufficient, close other GPU apps first. Change batch/patch sizes
for BOTH models in new configs or use the clearly labeled reduced `*_laptop.json`
pair. Changing precision should also apply to both models in a new experiment.
Equal widths do not imply equal parameter count or compute. Keep standard NAFNet
for the published-architecture study when the hardware supports it.

## 4. Start, inspect and resume both reference runs

Start with a short **100-step pilot**, in main run folders so it can be resumed:

```bash
python train.py --config configs/unet_standard.json --data data/sidd --output runs/unet_standard_seed42 --device cuda --stop-after 100
python train.py --config configs/nafnet_standard.json --data data/sidd --output runs/nafnet_standard_seed42 --device cuda --stop-after 100
```

Inspect finite losses, validation histories and checkpoint files. The pilot is
pipeline evidence, not a completed denoising experiment. Do not tune on test data.
If satisfactory, continue the SAME runs to their configured budgets:

```bash
python train.py --config configs/unet_standard.json --data data/sidd --output runs/unet_standard_seed42 --device cuda --resume runs/unet_standard_seed42/last.pt
python train.py --config configs/nafnet_standard.json --data data/sidd --output runs/nafnet_standard_seed42 --device cuda --resume runs/nafnet_standard_seed42/last.pt
```

Run models sequentially on the same GPU; keep the PC plugged in and awake
during the planned session. Supplied settings: 20,000 optimizer
steps, batch 2, 128px patches, L1, AdamW, cosine schedule and fixed seed. Both
models see matching sampled data/crops. Standard NAFNet uses width 32,
encoder [2,2,4,8], middle 12, decoder [2,2,2,2], with base global pooling.
This recipe is an initial controlled study, not the original paper's recipe/TLC.

`--stop-after` is an absolute total step without shortening the LR schedule.
Config/manifests must match on resume. Do not change code or pull updates during
an active run. Checkpoints save at validation intervals and requested stops;
unexpected termination loses later unsaved work. Original full-resolution PNG
reads may bottleneck training; use an SSD and measure actual wall time.

Record the exact code/environment with the experiment:

```bash
git rev-parse HEAD > runs/code_revision.txt
python -m pip freeze > runs/environment.txt
```

## 5. Inspect validation first; evaluate test only after choices are fixed

For a pilot, use a new output directory and validation only:

```bash
python evaluate.py --manifest data/sidd/val.csv --checkpoints runs/unet_standard_seed42/best.pt runs/nafnet_standard_seed42/best.pt --output results/pilot_validation --device cuda --tile-size 256 --overlap 32
```

After both planned budgets finish and model choices are fixed:

```bash
python evaluate.py --manifest data/sidd/test.csv --checkpoints runs/unet_standard_seed42/best.pt runs/nafnet_standard_seed42/best.pt --output results/standard_seed42_test --device cuda --tile-size 256 --overlap 32 --repeats 3
```

For the official protocol use `data/sidd_official/val.csv` and label scores as
validation. Choose new result folders rather than overwriting prior evaluations.
Outputs: `summary.json`, `per_image.csv`, and noisy | prediction | GT PNGs.
Report RGB PSNR/SSIM, parameters, device, training step/budget and inference mode.
Tile size affects NAFNet's pooling context; use identical inference settings for
both models. `--tile-size 0` gives full-image inference when memory permits.
Timings are warm float32 inference with synchronization, excluding loading,
transfers, metrics and saving. They are not end-to-end application runtimes.

Metrics: clipped float RGB [0,1], mean per-image PSNR, Gaussian 11x11 SSIM with
sigma 1.5 and population covariance over valid windows, no extra border crop.
Published scores and experimental results belong in separate tables.

For a separate photograph (visual inspection only, unless GT is available):

```bash
python denoise.py --input path/to/noisy.png --checkpoint runs/nafnet_standard_seed42/best.pt --output results/denoised.png --device cuda
```

## Files and optional Colab fallback

```text
check_environment.py   Hardware report and optional real-data training-memory check
prepare_sidd.py        Explicit SIDD pairing and fixed split preparation
train.py               Shared training and checkpoint/resume
evaluate.py            Checkpoint-only metrics, timings and comparison images
denoise.py             Single-photo denoising
models/                U-Net and NAFNet reference architectures
configs/               Standard and reduced matched experiment settings
docs/WORK_PLAN.md      Two-person responsibilities, milestones and supervisor update
RESEARCH_AND_PAPERS.md Source papers and recent reading
```

Local GPU training is the main workflow. If the local machine is unavailable,
[the existing Colab notebook](https://colab.research.google.com/github/nitya-goyal/SOTA-Image-Denoising/blob/main/notebooks/Image_Denoising_Colab.ipynb)
remains an optional fallback. Follow its Drive staging/checkpoint instructions;
do not mix different data splits or configurations when reporting results.

Software tests: `python -m pip install -r requirements-dev.txt`, then
`python -m pytest -q`. Generated test fixtures are software checks only.
See [docs/VERIFICATION.md](docs/VERIFICATION.md) for actual verification limits and
[docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) for experiment controls.
NAFNet attribution/license: [third_party/README.md](third_party/README.md).
