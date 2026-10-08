# State-of-the-Art (SOTA) Image Denoising & Noise Reduction: Comprehensive Research, Papers & Benchmark Guide

---

## 1. Executive Summary & SOTA Landscape

Image noise reduction (image denoising) is the fundamental low-level computer vision inverse problem of recovering a clean latent image $\mathbf{x}$ from a degraded observation $\mathbf{y} = \mathbf{x} + \mathbf{n}$, where $\mathbf{n}$ represents noise.

Over the last several years, the paradigm has evolved from classical hand-crafted priors to Deep CNNs, Shifted-Window Transformers, Activation-Free Networks, and Prompt-guided Multi-Task Restoration:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Evolution of Image Denoising Paradigms                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
  1990s - 2010s: Classical Handcrafted Priors
  ├── Non-Local Means (NLM) [Buades et al., CVPR 2005] - Patch self-similarity
  ├── BM3D [Dabov et al., IEEE TIP 2007] - 3D Transform-domain collaborative filtering
  └── WNNM [Gu et al., CVPR 2014] - Weighted nuclear norm minimization
         │
         ▼
  2016 - 2020: Deep CNNs & Residual Learning
  ├── DnCNN [Zhang et al., IEEE TIP 2017] - Residual learning with Batch Normalization
  ├── FFDNet [Zhang et al., IEEE TIP 2018] - Fast noise-level map conditioning
  ├── CBDNet [Guo et al., CVPR 2019] - Realistic camera sensor noise estimation
  └── RIDNet [Anwar & Barnes, ICCV 2019] - Residual local & channel attention
         │
         ▼
  2021 - Present: Modern SOTA (Transformers, Activation-Free & Generative Models)
  ├── SwinIR [Liang et al., ICCV Workshops 2021] - Shifted Window Self-Attention
  ├── Restormer [Zamir et al., CVPR 2022 Oral] - Multi-Dconv Head Transposed Attention (MDTA)
  ├── NAFNet [Chen et al., ECCV 2022] - Nonlinear Activation Free Network (Top efficiency/PSNR)
  ├── PromptIR [Potlapalli et al., NeurIPS 2023] - Implicit prompt-guided all-in-one restoration
  └── DiffPIR / DDRM [2023-2024] - Plug-and-play diffusion models for unconditional denoising
```

---

## 2. Deep Dive into the Top SOTA Models

### A. NAFNet (Nonlinear Activation Free Network) - ECCV 2022
- **Paper**: *"Simple Baselines for Image Restoration"* (Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun)
- **ArXiv**: [https://arxiv.org/abs/2204.04661](https://arxiv.org/abs/2204.04661)
- **Key Breakthrough**:
  The authors investigated why restoration networks need complex non-linear activations (e.g., ReLU, GELU, LeakyReLU) and discovered they can be completely removed without losing non-linear representation capacity by using **SimpleGate**:
  $$\text{SimpleGate}(X_1, X_2) = X_1 \odot X_2$$
  where the feature map $X \in \mathbb{R}^{C \times H \times W}$ is split into two halves $X_1, X_2 \in \mathbb{R}^{\frac{C}{2} \times H \times W}$.
- **Simplified Channel Attention (SCA)**:
  Standard SE-Net uses two linear layers with ReLU/Sigmoid. SCA reduces this to global average pooling + a single $1\times 1$ conv:
  $$\text{SCA}(X) = X \odot \mathcal{W}(\text{Pool}(X))$$
- **Benchmark Standing**: Outperforms Restormer on SIDD real-world denoising (**40.30 dB PSNR**) while running **up to 4x faster** with significantly lower MACs/FLOPs.

---

### B. Restormer (Efficient Transformer for Image Restoration) - CVPR 2022 Oral
- **Paper**: *"Restormer: Efficient Transformer for High-Resolution Image Restoration"* (Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang)
- **ArXiv**: [https://arxiv.org/abs/2111.09881](https://arxiv.org/abs/2111.09881)
- **Key Breakthrough**:
  1. **Multi-Dconv Head Transposed Attention (MDTA)**: Standard spatial self-attention computes an $HW \times HW$ attention matrix, which is prohibitive ($O(H^2 W^2)$) for high-resolution images. Restormer computes attention across channels instead of spatial dimensions:
     $$\text{Attention}(\hat{Q}, \hat{K}, \hat{V}) = \hat{V} \cdot \text{softmax}\left(\frac{\hat{K}^T \hat{Q}}{\alpha}\right)$$
     Complexity becomes $\mathcal{O}(C^2)$ rather than $\mathcal{O}((HW)^2)$, making it scale linearly with image resolution.
  2. **Gated-Dconv Feed-Forward Network (GDFN)**: Introduces depth-wise convolution and gating mechanism into the FFN to encode local context.
- **Benchmark Standing**: Set the gold-standard benchmark across DND, SIDD, Gaussian denoising, motion deblurring, and deraining.

---

### C. PromptIR (Prompting for All-in-One Image Restoration) - NeurIPS 2023
- **Paper**: *"PromptIR: Prompting for All-in-One Blind Image Restoration"* (Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan)
- **ArXiv**: [https://arxiv.org/abs/2306.13090](https://arxiv.org/abs/2306.13090)
- **Key Breakthrough**:
  Introduces learnable Prompt blocks that dynamically interact with input features to predict degradation context without needing explicit task labels or prior noise estimates. Handles denoising, dehazing, and deraining within a single unified model.

---

### D. Self-Supervised & Blind Denoising Family (Noise2Noise, Noise2Void, Neighbor2Neighbor)
1. **Noise2Noise (ICML 2018 - Lehtinen et al.)**: Proved mathematically that a network can learn clean image restoration by training on pairs of independent noisy observations $(y_1, y_2)$ of the same scene, because $\mathbb{E}[y|x] = x$.
2. **Noise2Void (CVPR 2019 - Krull et al.)**: Eliminates the need for pairs entirely by training on single noisy images using blind-spot masking (masking the center pixel of receptive field).
3. **Neighbor2Neighbor (CVPR 2021 - Huang et al.)**: Uses random sub-sampling to generate pairs of sub-images with identical statistical priors from a single noisy photo.

---

## 3. Comprehensive Benchmark Datasets

### A. Real-World Camera Noise Benchmarks
| Dataset | Year | Format / Sensor | Scale / Resolution | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **SIDD** (Smartphone Image Denoising Dataset) | CVPR 2018 | sRGB & RAW (5 Smartphones: S6, iPhone 7, G4, Pixel, Nexus 6P) | 30,000 paired crops (256x256) | The #1 benchmark for real-world smartphone noise under diverse lighting conditions (ISO 50 to 10,000). |
| **DND** (Darmstadt Noise Dataset) | CVPR 2017 | sRGB & RAW (Consumer DSLRs: Sony A7R, Nikon D800, etc.) | 50 high-res scenes (1000 512x512 crops) | Pristine real camera noise benchmark with online evaluation server. |
| **PolyU** | 2018 | Real DSLR photos | 100 scenes (Canon, Nikon, Sony) | Real noise under high ISO conditions. |
| **RENOIR** | 2018 | Paired Low/High ISO | 120 scenes | Real smartphone & DSLR pairs. |

### B. Standard Synthetic Gaussian / Benchmark Datasets
- **Kodak24**: 24 true-color pristine photographic images ($768 \times 512$).
- **CBSD68 / BSD68**: 68 natural test images from Berkeley Segmentation Dataset.
- **Urban100**: 100 images rich in repetitive structures and high-frequency patterns.
- **McMaster**: 18 color images with rich chromaticity.
- **DIV2K**: 1,000 2K high-resolution images used for training modern restoration backbones.

---

## 4. Quantitative SOTA Benchmark Comparison Table

### Real-World Denoising Benchmark (SIDD & DND)
| Model | Venue / Year | Parameters (M) | MACs (G) | SIDD (sRGB) PSNR (dB) | SIDD SSIM | DND (sRGB) PSNR (dB) | DND SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM3D** | IEEE TIP 2007 | 0 (Rule-based) | - | 25.65 | 0.685 | 34.51 | 0.851 |
| **DnCNN** | IEEE TIP 2017 | 0.67 M | 43.8 | 23.66 | 0.583 | 32.43 | 0.790 |
| **CBDNet** | CVPR 2019 | 4.35 M | - | 30.78 | 0.754 | 38.06 | 0.942 |
| **RIDNet** | ICCV 2019 | 1.50 M | - | 38.71 | 0.951 | 39.23 | 0.953 |
| **VDNAS** | CVPR 2020 | 7.80 M | - | 39.28 | 0.955 | 39.62 | 0.956 |
| **SwinIR** | ICCVW 2021 | 11.75 M | 48.6 | 39.82 | 0.958 | 39.92 | 0.957 |
| **Restormer** | CVPR 2022 | 26.10 M | 141.0 | 40.02 | 0.960 | 40.03 | 0.956 |
| **NAFNet-width64** | ECCV 2022 | 67.89 M | 63.3 | **40.30** | **0.962** | **40.15** | **0.958** |
| **NAFNet-width32** | ECCV 2022 | 17.11 M | 16.0 | **40.15** | **0.960** | 40.02 | 0.956 |

### Synthetic Color Gaussian Denoising Benchmark ($\sigma = 25, 50$)
| Model | Kodak24 ($\sigma=25$) | Kodak24 ($\sigma=50$) | CBSD68 ($\sigma=25$) | CBSD68 ($\sigma=50$) | Urban100 ($\sigma=25$) | Urban100 ($\sigma=50$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM3D** | 31.68 | 28.46 | 30.71 | 27.38 | 30.93 | 27.28 |
| **DnCNN** | 32.14 | 28.95 | 31.23 | 27.92 | 31.58 | 27.97 |
| **FFDNet** | 32.13 | 28.98 | 31.21 | 27.96 | 31.53 | 28.05 |
| **SwinIR** | 32.89 | 29.86 | 31.78 | 28.77 | 33.78 | 30.55 |
| **Restormer** | **33.04** | **30.01** | **31.96** | **28.94** | **34.33** | **31.12** |
| **NAFNet** | **33.02** | **29.98** | **31.94** | **28.91** | **34.28** | **31.08** |

---

## 5. Mathematical Formulations & Noise Models

### 1. Heteroscedastic Poisson-Gaussian Camera Sensor Noise Model
In physical digital sensors (CMOS/CCD), noise originates from photon arrival uncertainty (shot noise) and sensor electronics (readout noise):
$$y = x + n_p(x) + n_g$$
$$\mathbb{E}[y|x] = x, \quad \text{Var}(y|x) = \beta_1 x + \beta_2$$
where $\beta_1$ is photon-to-electron amplification gain and $\beta_2$ is readout Gaussian variance.

### 2. Charbonnier Loss Function
$$\mathcal{L}_{\text{char}}(\hat{\mathbf{x}}, \mathbf{x}) = \frac{1}{N}\sum_{i=1}^N \sqrt{(\hat{\mathbf{x}}_i - \mathbf{x}_i)^2 + \epsilon^2}, \quad (\epsilon = 10^{-3})$$
Compared to $L_1$ and $L_2$ losses, Charbonnier loss:
- Prevents infinite gradient oscillations around zero ($\lim_{\Delta \to 0} \nabla L_1$ is discontinuous).
- Is resilient to heavy-tailed sensor outlier noise (unlike $L_2$ MSE which penalizes outliers quadratically, causing blur).

### 3. Progressive Patch Size Training Strategy
Restormer introduced the progressive learning strategy:
- Start training with smaller patch sizes (e.g. $128 \times 128$) with larger batch sizes ($B=64$) to rapidly learn low-frequency structural representations.
- Gradually scale up patch sizes ($192 \times 192 \to 256 \times 256 \to 384 \times 384$) while decreasing batch size to learn fine high-frequency textures and long-range pixel correlations.

---

## 6. How to Run on Google Colab (Free GPU)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **File -> Upload notebook** and select [`Image_Denoising_SOTA_Colab.ipynb`](./Image_Denoising_SOTA_Colab.ipynb).
3. Ensure GPU runtime is enabled: **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU**.
4. Run all cells sequentially to train, evaluate, and view side-by-side zoomed comparison plots!
