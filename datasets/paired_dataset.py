"""Explicit CSV pairing, synchronized crops and augmentation, no added noise."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'id', 'noisy', 'clean', 'group'}.issubset(rows[0]):
        raise ValueError(f'{path}: expected nonempty CSV with id,noisy,clean,group')
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError(f'Duplicate pair ID: {row["id"]}')
        seen.add(row['id'])
        for key in ('noisy', 'clean'):
            row[key] = (path.parent / row[key]).resolve()
            if not row[key].is_file():
                raise FileNotFoundError(row[key])
    return rows


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'{path}: expected 8-bit RGB PNG, got {image.mode}')
        return np.array(image, dtype=np.uint8)


def tensor_rgb(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)


class PairedDataset(Dataset):
    def __init__(self, manifest, patch_size=None, training=False):
        self.rows = read_manifest(manifest)
        self.patch_size = patch_size
        self.training = training

    def __len__(self):
        return len(self.rows)

    def sample(self, index, rng):
        row = self.rows[index]
        noisy, clean = read_rgb(row['noisy']), read_rgb(row['clean'])
        if noisy.shape != clean.shape:
            raise ValueError(f'Shape mismatch in pair {row["id"]}')
        if self.patch_size:
            size = self.patch_size
            h, w = noisy.shape[:2]
            if min(h, w) < size:
                raise ValueError(f'{row["id"]} is smaller than patch_size={size}')
            if self.training:
                top, left = rng.randrange(h - size + 1), rng.randrange(w - size + 1)
            else:
                top, left = (h - size) // 2, (w - size) // 2
            noisy = noisy[top:top + size, left:left + size]
            clean = clean[top:top + size, left:left + size]
        if self.training:
            rotations, flip = rng.randrange(4), rng.randrange(2)
            noisy, clean = np.rot90(noisy, rotations), np.rot90(clean, rotations)
            if flip:
                noisy, clean = np.fliplr(noisy), np.fliplr(clean)
        return {'id': row['id'], 'noisy': tensor_rgb(noisy), 'clean': tensor_rgb(clean)}

    def __getitem__(self, index):
        import random
        return self.sample(index, random)
