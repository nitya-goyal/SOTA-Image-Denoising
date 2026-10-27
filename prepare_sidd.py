"""Prepare explicit SIDD sRGB manifests. Never download or invent research data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
from scipy.io import loadmat


def discover_pairs(root):
    root = Path(root).resolve()
    rows = []
    # Handles NOISY_SRGB_010.PNG and older prefixed naming conventions.
    noisy_files = sorted(p for p in root.rglob('*') if p.suffix.lower() == '.png'
                         and 'NOISY_SRGB' in p.name.upper())
    for noisy in noisy_files:
        expected = re.sub('NOISY_SRGB', 'GT_SRGB', noisy.name, flags=re.I).lower()
        candidates = [p for p in noisy.parent.iterdir() if p.name.lower() == expected]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one GT_SRGB partner for {noisy}')
        clean = candidates[0]
        # Official folder: instance_scene_camera_ISO_shutter_temperature_brightness.
        # Group by physical scene, NOT scene instance, so camera/ISO variants stay together.
        match = re.fullmatch(r'\d+_(\d+)_[A-Za-z0-9]+_.+', noisy.parent.name)
        if not match:
            raise ValueError(f'Unrecognized SIDD scene folder: {noisy.parent.name}; keep original folders')
        with Image.open(noisy) as a, Image.open(clean) as b:
            if a.mode != 'RGB' or b.mode != 'RGB' or a.size != b.size:
                raise ValueError(f'Expected aligned RGB pair: {noisy}')
            if min(a.size) < 11:
                raise ValueError(f'Image too small for metrics: {noisy}')
        rows.append({'id': noisy.relative_to(root).as_posix(), 'noisy': noisy,
                     'clean': clean, 'group': f'scene_{int(match[1]):02d}'})
    if not rows:
        raise ValueError(f'No SIDD NOISY_SRGB PNGs under {root}')
    # Also reject orphan clean images instead of silently ignoring an incomplete extract.
    found_clean = {r['clean'].resolve() for r in rows}
    all_clean = {p.resolve() for p in root.rglob('*') if p.suffix.lower() == '.png'
                 and 'GT_SRGB' in p.name.upper()}
    if found_clean != all_clean:
        raise ValueError('Orphan GT_SRGB images found; verify the dataset extraction')
    return rows


def group_split(rows, seed=42):
    groups = sorted({r['group'] for r in rows})
    if len(groups) < 3:
        raise ValueError('Internal train/val/test split needs at least 3 physical scenes')
    random.Random(seed).shuffle(groups)
    holdout = max(1, round(len(groups) * 0.1))
    mapping = {g: 'train' for g in groups}
    mapping.update({g: 'val' for g in groups[:holdout]})
    mapping.update({g: 'test' for g in groups[holdout:2*holdout]})
    return {split: [r for r in rows if mapping[r['group']] == split]
            for split in ('train', 'val', 'test')}


def export_validation(noisy_mat, clean_mat, output):
    """Official SIDD validation blocks, shape (scene, block, H, W, 3), uint8."""
    def blocks(path, key):
        try:
            values = loadmat(path)[key]
        except NotImplementedError as error:
            raise ValueError('Use official non-HDF5 SIDD validation MAT files') from error
        except KeyError as error:
            raise ValueError(f'{path} must contain {key}') from error
        if values.ndim != 5 or values.shape[-1] != 3 or values.dtype != np.uint8:
            raise ValueError(f'{key}: expected uint8 (scenes,blocks,H,W,3), got {values.shape}/{values.dtype}')
        return values
    noisy = blocks(noisy_mat, 'ValidationNoisyBlocksSrgb')
    clean = blocks(clean_mat, 'ValidationGtBlocksSrgb')
    if noisy.shape != clean.shape:
        raise ValueError('Validation MAT shapes differ')
    folder = Path(output) / 'validation_blocks'
    folder.mkdir(parents=True)
    rows = []
    for scene in range(noisy.shape[0]):
        for block in range(noisy.shape[1]):
            key = f'{scene:03d}_{block:03d}'
            a, b = folder / f'{key}_noisy.png', folder / f'{key}_clean.png'
            Image.fromarray(noisy[scene, block]).save(a)
            Image.fromarray(clean[scene, block]).save(b)
            rows.append({'id': key, 'noisy': a.resolve(), 'clean': b.resolve(),
                         'group': f'official_validation_{scene:03d}'})
    return rows


def write_manifest(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'noisy', 'clean', 'group'])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for key in ('noisy', 'clean'):
                try:
                    row[key] = Path(os.path.relpath(row[key], Path(path).parent)).as_posix()
                except ValueError:
                    # Windows datasets can live on a different drive.
                    row[key] = Path(row[key]).resolve().as_posix()
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Extracted SIDD Medium sRGB root (recursive)')
    parser.add_argument('--output', type=Path, default=Path('data/sidd'))
    parser.add_argument('--protocol', choices=['internal', 'official'], default='internal')
    parser.add_argument('--val-noisy', type=Path)
    parser.add_argument('--val-clean', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty/new; use another folder to preserve existing splits')
    if args.protocol == 'official' and not (args.val_noisy and args.val_clean):
        parser.error('Official protocol needs --val-noisy and --val-clean MAT files')
    if args.protocol == 'internal' and (args.val_noisy or args.val_clean):
        parser.error('Use --protocol official with validation MAT files')
    rows = discover_pairs(args.source)
    if len(rows) != 320:
        print(f'NOTE: Found {len(rows)} pairs; complete SIDD Medium has 320. Record any subset use.')
    args.output.mkdir(parents=True, exist_ok=True)
    splits = (group_split(rows, args.seed) if args.protocol == 'internal' else
              {'train': rows, 'val': export_validation(args.val_noisy, args.val_clean, args.output)})
    for split, items in splits.items():
        write_manifest(args.output / f'{split}.csv', items)
    metadata = {'protocol': args.protocol, 'seed': args.seed,
                'grouping': 'physical scene (second original folder field)',
                'splits': {s: {'pairs': len(items), 'groups': sorted({r['group'] for r in items}),
                               'sha256': hashlib.sha256((args.output / f'{s}.csv').read_bytes()).hexdigest()}
                           for s, items in splits.items()}}
    (args.output / 'split_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
