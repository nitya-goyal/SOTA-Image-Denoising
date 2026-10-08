"""
Tiling Inference Engine:
Performs seamless overlapping patch-based inference for high-resolution images
to avoid Out-Of-Memory (OOM) on GPUs.
"""

import math
import torch
import torch.nn.functional as F


def tile_inference(model, img_tensor, tile_size=256, tile_overlap=32, device='cuda'):
    """
    Seamless Overlapping Tiling Inference
    Args:
        model: PyTorch restoration model
        img_tensor: Tensor of shape (1, C, H, W) in [0, 1]
        tile_size: Size of square tiles (e.g. 256 or 512)
        tile_overlap: Pixel overlap between adjacent tiles (e.g. 32 or 64)
        device: 'cuda' or 'cpu'
    Returns:
        output: Tensor of shape (1, C, H, W) in [0, 1]
    """
    model.eval()
    b, c, h, w = img_tensor.shape
    assert b == 1, "Tile inference currently supports batch size 1"

    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            return model(img_tensor.to(device)).cpu()

    stride = tile_size - tile_overlap
    h_tiles = int(math.ceil((h - tile_overlap) / stride))
    w_tiles = int(math.ceil((w - tile_overlap) / stride))

    # Weighting window (blend border artifacts)
    window = torch.ones((1, 1, tile_size, tile_size), device=device)
    # Cosine/linear taper on edges
    if tile_overlap > 0:
        ramp = torch.linspace(0, 1, tile_overlap, device=device)
        for i in range(tile_overlap):
            window[:, :, i, :] *= ramp[i]
            window[:, :, tile_size - 1 - i, :] *= ramp[i]
            window[:, :, :, i] *= ramp[i]
            window[:, :, :, tile_size - 1 - i] *= ramp[i]

    output_tensor = torch.zeros((1, c, h, w), device=device)
    weight_tensor = torch.zeros((1, 1, h, w), device=device)

    with torch.no_grad():
        for hi in range(h_tiles):
            for wi in range(w_tiles):
                hs = min(hi * stride, h - tile_size)
                ws = min(wi * stride, w - tile_size)
                he = hs + tile_size
                we = ws + tile_size

                tile = img_tensor[:, :, hs:he, ws:we].to(device)
                pred_tile = model(tile)

                output_tensor[:, :, hs:he, ws:we] += pred_tile * window
                weight_tensor[:, :, hs:he, ws:we] += window

    output_tensor = output_tensor / torch.clamp(weight_tensor, min=1e-8)
    return torch.clamp(output_tensor.cpu(), 0.0, 1.0)
