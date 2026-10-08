# 🌟 SOTA Image Denoising & Noise Reduction Framework

An end-to-end research and production framework implementing modern State-of-the-Art (SOTA) Image Denoising architectures (NAFNet, Restormer, DnCNN, BM3D baselines), standard benchmarks, loss functions, seamless overlapping tiling inference, and ready-to-use Google Colab Free GPU integration.

---

## 📁 Repository Structure

```
NoiseReduction/
├── Image_Denoising_SOTA_Colab.ipynb  # 🚀 Turnkey Google Colab notebook (Free T4 GPU)
├── RESEARCH_AND_PAPERS.md           # 📚 Deep-dive research, math formulations & benchmark tables
├── requirements.txt                  # Python dependencies
├── train.py                          # Training pipeline with AMP, Cosine Annealing, Checkpoints
├── eval.py                           # Benchmark evaluation script (PSNR, SSIM, Latency)
├── benchmark.py                      # Multi-model comparison suite (DL vs Classical)
├── denoise_single.py                 # Single image CLI denoiser with comparison generator
├── models/
│   ├── nafnet.py                     # NAFNet (Nonlinear Activation Free Network - ECCV 2022)
│   ├── restormer.py                  # Restormer (MDTA + GDFN Transformer - CVPR 2022)
│   ├── dncnn.py                      # DnCNN Deep Residual CNN Baseline (TIP 2017)
│   └── baselines.py                  # BM3D, Non-Local Means (NLM), Bilateral, Wavelet TV
├── datasets/
│   ├── noise_models.py               # AWGN, Poisson, Sensor Poisson-Gaussian & ISP Noise
│   ├── paired_dataset.py             # Paired and synthetic dataset loaders with augmentations
│   └── synthetic_data.py             # Kodak24 downloader & test suite generator
├── losses/
│   └── losses.py                     # Charbonnier Loss, FFT Frequency Loss, Sobel Edge Loss
├── metrics/
│   └── metrics.py                    # PSNR (RGB & Y-channel), SSIM (Wang et al.)
└── utils/
    ├── tiling.py                     # Seamless overlapping sliding-window inference for large images
    └── visualizer.py                 # Side-by-side comparison & ROI zoom-in crop generator
```

---

## 🚀 Quickstart Guide

### 1. Run on Google Colab (Free GPU)
1. Upload [`Image_Denoising_SOTA_Colab.ipynb`](file:///c:/Users/user/Documents/ImageProcessing/NoiseReduction/Image_Denoising_SOTA_Colab.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells to execute training, benchmark evaluation, and interactive visual ROI comparisons.

---

### 2. Local Usage & CLI Commands

#### Install Requirements
```bash
pip install -r requirements.txt
```

#### Train a Model (e.g. NAFNet or Restormer)
```bash
# Train NAFNet-Lite on synthetic Gaussian noise (sigma=25)
python train.py --model nafnet_tiny --epochs 30 --batch_size 8 --sigma 25.0

# Train Restormer-Lite
python train.py --model restormer_tiny --epochs 30 --batch_size 4 --sigma 25.0
```

#### Run Multi-Model Benchmark Comparison
```bash
# Compares NAFNet vs Restormer vs DnCNN vs BM3D vs NLM vs Bilateral Filter
python benchmark.py
```

#### Denoise a Single Image
```bash
# Denoise an image and generate a side-by-side ROI comparison
python denoise_single.py --input path/to/image.png --output ./results/denoised.png --model nafnet_tiny --sigma 25.0
```

#### Evaluate on Kodak24 or SIDD
```bash
python eval.py --model nafnet_tiny --test_dir ./data/Kodak24 --sigma 25.0 --output_dir ./results/eval
```

---

## 🏆 SOTA Performance Comparison

| Model | Venue | SIDD (Real Noise) PSNR | DND (Real Noise) PSNR | Kodak24 ($\sigma=25$) | Key Innovation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM3D** | TIP 2007 | 25.65 dB | 34.51 dB | 31.68 dB | Collaborative 3D Filtering |
| **DnCNN** | TIP 2017 | 23.66 dB | 32.43 dB | 32.14 dB | Deep Residual Learning |
| **SwinIR** | ICCVW 2021 | 39.82 dB | 39.92 dB | 32.89 dB | Shifted Window Attention |
| **Restormer** | CVPR 2022 | 40.02 dB | 40.03 dB | **33.04 dB** | Multi-Dconv Head Transposed Attention (MDTA) |
| **NAFNet** | ECCV 2022 | **40.30 dB** | **40.15 dB** | 33.02 dB | Nonlinear Activation Free (SimpleGate + SCA) |

See [`RESEARCH_AND_PAPERS.md`](file:///c:/Users/user/Documents/ImageProcessing/NoiseReduction/RESEARCH_AND_PAPERS.md) for full mathematical analysis, benchmark datasets, and loss derivations.
