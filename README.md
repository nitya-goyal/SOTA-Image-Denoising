# Real-World Image Denoising

Research on removing real camera noise while preserving edges, textures and
colours. **Current stage: implement and understand reference methods.** U-Net is
our baseline and NAFNet is the stronger published method. Later work will study
failure cases, propose an improvement and verify it with controlled experiments.
These two initial methods are not the final project title or contribution.

Use paired **SIDD Medium sRGB** images. No synthetic noise is added. NAFNet was
a SOTA method at publication in 2022; this repo's common-budget recipe does not
claim current SOTA or reproduction of its published scores. No trained weights
or real SIDD experiment results are bundled.

## Setup: Windows 11 + NVIDIA RTX 4050 Laptop GPU

Use Python 3.11/3.12, Git and the existing dataset on local SSD storage.
Run commands in PowerShell from the repo root:

```powershell
git clone https://github.com/nitya-goyal/SOTA-Image-Denoising.git
cd SOTA-Image-Denoising
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`.
Install a **CUDA-enabled PyTorch build** using the
[official installer selector](https://pytorch.org/get-started/locally/) for
Windows and the installed NVIDIA driver. Do not select the CPU-only build.
Then:

```powershell
python -m pip install -r requirements.txt
python check_environment.py --require-gpu --output runs/hardware.json
```

The check reports actual GPU name/free memory and verifies device execution.
The GPU model alone does not establish free VRAM, supported installation or
training speed. Resolve a failed GPU check before a long run. The scripts also
run on compatible cloud GPUs; no Colab, Drive or provider-specific setup is required.

## Prepare the real dataset

Keep original SIDD scene folders, e.g.
`0001_001_S6_00100_00060_3200_L/NOISY_SRGB_010.PNG` and `GT_SRGB_010.PNG`.
Use sRGB PNG pairs, not RAW MAT data or noisy-only benchmark images. The complete
Medium set contains 320 pairs. Replace the example source path with yours:

```powershell
python prepare_sidd.py --source "D:/datasets/SIDD_Medium_Srgb" --output data/sidd --protocol internal --seed 42
```

This verifies pairs and makes fixed physical-scene-disjoint train/val/test
manifests. Preserve the same split for both models. These are **internal holdout
results**, not official SIDD benchmark scores. If you have the official
validation noisy/GT MAT files, use the separate official protocol described in
[docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md). Do not mix protocols.

## Check memory and run the pilot

Both standard configs use batch 2, 128px training patches and float32. Check
forward, backward and optimizer allocation with real data before long training:

```powershell
python check_environment.py --device cuda --require-gpu --config configs/unet_standard.json --data data/sidd --output runs/preflight_unet.json
python check_environment.py --device cuda --require-gpu --config configs/nafnet_standard.json --data data/sidd --output runs/preflight_nafnet.json
```

These are disposable setup checks, not training results. If memory is insufficient,
close other GPU apps or lower batch size for BOTH models in copied configs and
use new run folders. Keep the standard architectures unchanged. Do not alter a
config mid-run. Validate actual validation/inference memory during the pilot too.

Run models sequentially, with the laptop plugged in and awake:

```powershell
python train.py --config configs/unet_standard.json --data data/sidd --output runs/unet_standard_seed42 --device cuda --stop-after 100
python train.py --config configs/nafnet_standard.json --data data/sidd --output runs/nafnet_standard_seed42 --device cuda --stop-after 100
```

Inspect finite loss/validation histories and saved checkpoints. A 100-step pilot
is preliminary pipeline evidence; do not claim trained-model quality or tune on
the test split. Full-resolution PNG loading can bottleneck training; measure the
actual pilot runtime before estimating completion time.

Record the code/environment and resume the SAME runs to their configured budgets:

```powershell
git rev-parse HEAD > runs/code_revision.txt
python -m pip freeze > runs/environment.txt
python train.py --config configs/unet_standard.json --data data/sidd --output runs/unet_standard_seed42 --device cuda --resume runs/unet_standard_seed42/last.pt
python train.py --config configs/nafnet_standard.json --data data/sidd --output runs/nafnet_standard_seed42 --device cuda --resume runs/nafnet_standard_seed42/last.pt
```

The initial recipe is 20,000 optimizer steps, L1, AdamW and cosine LR, with
matching sampled pairs/crops/augmentations at the same seed. NAFNet uses the
published width-32 base block configuration [2,2,4,8] / 12 / [2,2,2,2]. This is
not the original training recipe or local-pooling/TLC inference.
`--stop-after` is an absolute total step and retains the original LR schedule.
Unexpected interruption loses work since the last checkpoint. Do not pull code
changes while training, and retain the complete run folder for resuming.

## Evaluate and inspect images

For the pilot use validation only:

```powershell
python evaluate.py --manifest data/sidd/val.csv --checkpoints runs/unet_standard_seed42/best.pt runs/nafnet_standard_seed42/best.pt --output results/pilot_validation --device cuda
```

After both planned budgets finish and model choices are fixed:

```powershell
python evaluate.py --manifest data/sidd/test.csv --checkpoints runs/unet_standard_seed42/best.pt runs/nafnet_standard_seed42/best.pt --output results/standard_seed42_test --device cuda --repeats 3
```

Outputs: `summary.json`, `per_image.csv`, and **noisy | prediction | ground truth**
PNGs. Use fresh output folders. Report metrics with protocol, training steps,
config, device and inference mode. Default inference uses 256px tiles with 32px
overlap; use identical settings for both models. Tiling changes NAFNet's pooling
context. `--tile-size 0` enables full-image inference when memory allows.
Metrics use clipped float RGB [0,1], mean per-image PSNR and Gaussian 11x11 SSIM.
Full metric/timing conventions are in the research protocol.

Single-photo inference (without GT, visual inspection only):

```powershell
python denoise.py --input path/to/noisy.png --checkpoint runs/nafnet_standard_seed42/best.pt --output results/denoised.png --device cuda
```

## Two contributors, one shared research record

Both members can work on the repo after the collaborator invitation is accepted.
Before a new task, start from an up-to-date main branch with a clean working tree:

```powershell
git switch main
git pull --ff-only
git switch -c experiments/rtx4050-setup
```

Use a separate branch for each change and open a pull request for the other
member to review. Commit source, configs and concise experiment notes. Keep
SIDD, virtual environments, checkpoints and generated results out of Git; share
those separately. Do not commit credentials or machine-specific absolute paths.
Keep completed baseline code/configs/results unchanged when exploring improvements.

The role split, milestones and supervisor update outline are in
[docs/WORK_PLAN.md](docs/WORK_PLAN.md). Source papers and recent reading are in
[RESEARCH_AND_PAPERS.md](RESEARCH_AND_PAPERS.md).

## Repository

```text
check_environment.py   Hardware report and real-data memory check
prepare_sidd.py        SIDD pairing and reproducible split preparation
train.py               Training and checkpoint/resume
evaluate.py            Saved-model metrics, timings and image comparisons
denoise.py             Single-photo inference
models/                U-Net and NAFNet
configs/               One standard config per reference architecture
datasets/, metrics/, utils/  Shared implementation
```

Software checks: `python -m pip install -r requirements-dev.txt`, then
`python -m pytest -q`. Tiny generated test fixtures are software checks only;
they are not used in research training. Actual verification limits are recorded
in [docs/VERIFICATION.md](docs/VERIFICATION.md). NAFNet attribution and license:
[third_party/README.md](third_party/README.md).
