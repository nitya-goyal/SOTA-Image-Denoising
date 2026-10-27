# NAFNet attribution

`models/nafnet.py` adapts the NAFNet architecture and channel LayerNorm from
[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), copyright
2022 megvii-model. The complete upstream license is in `NAFNet-LICENSE`.

Reference: Chen, Chu, Zhang and Sun, *Simple Baselines for Image Restoration*,
ECCV 2022, https://arxiv.org/abs/2204.04676.

Changes include a standalone PyTorch model without BasicSR dependencies,
native PyTorch channel LayerNorm, project-specific constructors and no local-pooling/TLC implementation. The
standard block counts follow upstream SIDD width-32 configuration; the laptop
configuration is intentionally smaller. The U-Net regression baseline is a
separate implementation inspired by the published encoder-decoder design.
