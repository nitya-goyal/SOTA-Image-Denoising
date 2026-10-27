"""Denoise one RGB photograph using a trained repository checkpoint."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from datasets.paired_dataset import tensor_rgb
from utils.inference import predict
from utils.runtime import as_hwc, choose_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    with Image.open(args.input) as image:
        noisy = tensor_rgb(np.array(image.convert('RGB'))).unsqueeze(0).to(device)
    prediction = predict(model, noisy, args.tile_size, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(as_hwc(prediction[0]) * 255).astype(np.uint8)).save(args.output)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
