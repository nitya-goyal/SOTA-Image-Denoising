"""Optional overlapping tiles. Tiling changes NAFNet's pooling context;
record the tile size and use the same inference mode for both models.
"""
import torch


@torch.inference_mode()
def predict(model, image, tile_size=0, overlap=32):
    if tile_size == 0:
        return model(image).clamp(0, 1)
    if tile_size < 16 or overlap < 0 or overlap >= tile_size:
        raise ValueError('Require tile_size >= 16 and 0 <= overlap < tile_size')
    height, width = image.shape[-2:]
    def starts(length):
        return sorted(set(list(range(0, max(1, length - tile_size + 1), tile_size - overlap))
                          + [max(0, length - tile_size)]))
    result, count = torch.zeros_like(image), torch.zeros_like(image)
    for top in starts(height):
        for left in starts(width):
            tile = image[..., top:top + tile_size, left:left + tile_size]
            result[..., top:top + tile_size, left:left + tile_size] += model(tile)
            count[..., top:top + tile_size, left:left + tile_size] += 1
    return (result / count).clamp(0, 1)
