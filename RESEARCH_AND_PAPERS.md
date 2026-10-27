# Image denoising: research papers and project scope

Literature checked against primary sources on **27 October 2026**. This is a
focused reading list, not a claim to have exhaustively ranked every 2026 method.
New publication date alone does not establish state-of-the-art performance.

## Papers directly used by this project

| Paper | Publication | Why it matters here |
| --- | --- | --- |
| [A High-Quality Denoising Dataset for Smartphone Cameras](https://abdokamel.github.io/sidd/) — Abdelhamed, Lin, Brown | CVPR 2018 | SIDD supplies real smartphone noisy/ground-truth image pairs. We use its sRGB domain. |
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger, Fischer, Brox | MICCAI 2015 | Encoder-decoder with skip connections; our baseline adapts the architecture to RGB regression. |
| [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676) — Chen, Chu, Zhang, Sun | ECCV 2022 | Introduces NAFNet and reports then-SOTA denoising performance. This is our stronger supervised model. |

NAFNet's official code and architecture/training settings:

- [Repository](https://github.com/megvii-research/NAFNet)
- [SIDD width-32 training config](https://github.com/megvii-research/NAFNet/blob/main/options/train/SIDD/NAFNet-width32.yml)
- [SIDD preparation and evaluation instructions](https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md)

Our standard NAFNet uses width 32, encoder blocks [2,2,4,8], middle blocks 12,
decoder blocks [2,2,2,2]. Our shared L1/laptop training budget is a controlled
comparison recipe, not a reproduction of the original training procedure.
We implement base NAFNet global pooling, not its local pooling conversion.

## Recent image-denoising literature

These papers keep the literature review current while our implementation stays
focused. None is implemented or benchmarked by this repository.

| Paper | Publication / status | Setting and relevance |
| --- | --- | --- |
| [TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](https://arxiv.org/abs/2604.04484) — Park, Oh, Cho | CVPR 2026; arXiv April 2026 | Models spatially correlated sRGB noise with a masked blind spot and distillation. Relevant recent real-image denoising; it uses a different supervision setting from our paired training. |
| [Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising](https://openaccess.thecvf.com/content/CVPR2026/html/Shan_Next-Scale_Prediction_A_Self-Supervised_Approach_for_Real-World_Image_Denoising_CVPR_2026_paper.html) — Shan et al. | CVPR 2026 | Cross-scale prediction addresses noise correlation versus detail preservation. Read for the current self-supervised denoising direction. |
| [Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs](https://arxiv.org/abs/2605.03680) — Kayani et al. | CVPR Workshops 2026, Mobile AI; arXiv May 2026 | Hardware-aware distillation illustrates why device runtime and operator compatibility matter alongside restoration quality. Its mobile challenge protocol is not our SIDD comparison. |
| [Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising](https://ojs.aaai.org/index.php/AAAI/article/view/32506) — Li, Zhang, Zuo | AAAI 2025 | TBSN redesigns attention for blind-spot denoising; useful context for the 2026 methods. |
| [SwinIA: Self-Supervised Blind-Spot Image Denoising without Convolutions](https://openaccess.thecvf.com/content/WACV2025/html/Papkov_SwinIA_Self-Supervised_Blind-Spot_Image_Denoising_without_Convolutions_WACV_2025_paper.html) — Papkov, Chizhov, Parts | WACV 2025 | Transformer-based self-supervised denoising; broader architectural context. |

Use the primary paper and released code to verify each method's dataset,
supervision, preprocessing and evaluation protocol before comparing any number.
Self-supervised, paired-supervised, RAW, sRGB, mobile challenge, validation and
hidden benchmark results must be labeled separately. We do not assert a single
latest best model across these settings.

## What we can claim after running experiments

Proposed title: **Real-World Image Denoising on SIDD: A Controlled Comparison of
U-Net and NAFNet**.

The core study measures RGB PSNR, SSIM, parameter count and device-specific warm
inference time under a documented common training budget. A stronger research
analysis would repeat training seeds and report scene-level behavior, failure
cases and quality/compute tradeoffs. A new SOTA or novel-method claim requires
additional evidence and a contribution beyond implementing two known networks.

Published numbers are not copied into a local results table. Populate that table
only from saved experiment outputs. See `docs/RESEARCH_PROTOCOL.md`.

## Citations

```bibtex
@inproceedings{abdelhamed2018sidd,
  title={A High-Quality Denoising Dataset for Smartphone Cameras},
  author={Abdelhamed, Abdelrahman and Lin, Stephen and Brown, Michael S.},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention},
  year={2015}
}
@inproceedings{chen2022nafnet,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
@inproceedings{park2026tmbsn,
  title={TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising},
  author={Park, Junyoung and Oh, Youngjin and Cho, Nam Ik},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
@inproceedings{shan2026nsp,
  title={Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising},
  author={Shan, Yiwen and Zhao, Haiyu and Hu, Peng and Peng, Xi and Gou, Yuanbiao},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
