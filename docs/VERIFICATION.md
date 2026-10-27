# Software verification

Checked on 27 October 2026 with Windows, Python 3.12.0, CPU PyTorch 2.14.1,
NumPy 2.5.4, Pillow 13.0.0, SciPy 1.18.1 and pytest 9.1.1.

**Result: 8 tests passed** on the final code (204.61 seconds).

The test suite covers explicit image pairing and physical-scene separation,
synchronized augmentation, model gradients and arbitrary image sizes, LayerNorm
gradient checking, PSNR/SSIM against known values, overlapping tile coverage,
official-format MAT conversion, and a small train/resume/evaluate/denoise run
for both architectures. It verifies CPU resume against uninterrupted training.
Small generated fixtures test software behavior only; they are not research data.

All four supplied model configurations passed a 35x37 RGB forward check.
The standard NAFNet width-32 architecture has 29,159,715 parameters and loaded
the official architecture's state dictionary with strict key/shape checking.
Its output matched the official architecture on the checked input. Native
PyTorch channel LayerNorm was used in the reference wrapper; local-pooling/TLC
was not involved. Official source used:

`basicsr/models/archs/NAFNet_arch.py` from megvii-research/NAFNet, SHA256
`01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`.

No real SIDD training or benchmark experiment was run in this workspace.
CUDA, AMP and Apple MPS execution were not tested on hardware. Full training
speed, final denoising quality and memory requirements must be measured on the
teammate's actual dataset and device. Source compatibility checks do not establish
published-score reproduction or pretrained-weight import support.

Run `python -m pytest -q`. In this restricted Windows sandbox the OS temporary
directory blocked checkpoint renames, so verification used a workspace-owned
temporary directory: `python -m pytest -q --basetemp=data/verification/pytest_final`.


## Colab notebook

The stage-one notebook's code cells compile and contain no saved execution
outputs. Local checks emulated Colab/Drive setup and verified saved revision
round-tripping, local folder staging, execution of the real preparation script,
manifest hash checks, config copying, and smoke/chunk/resume command selection.
Cloud mounting, cloning/installing in Colab, CUDA execution and a full SIDD
training session were not performed by these local checks. The existing model
and training code is unchanged by the notebook addition.
