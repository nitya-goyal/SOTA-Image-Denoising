"""Evaluate saved models on the SAME real noisy/clean images and export results."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from datasets.paired_dataset import PairedDataset
from metrics import image_metrics
from utils.inference import predict
from utils.runtime import as_hwc, choose_device, environment, file_hash, load_checkpoint, save_json, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/comparison'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--tile-size', type=int, default=256, help='0 = full image; record this when reporting')
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1, help='Timed repeats after one warm-up per image')
    parser.add_argument('--save-images', type=int, default=5, help='Number of side-by-side comparisons per model')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.save_images < 0:
        parser.error('Require repeats/threads >= 1 and save-images >= 0')
    if args.tile_size and (args.tile_size < 16 or not 0 <= args.overlap < args.tile_size):
        parser.error('Require tile-size >= 16 and 0 <= overlap < tile-size, or tile-size 0')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty/new output directory to avoid mixing evaluations')
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    dataset = PairedDataset(args.manifest)
    manifest_hash = file_hash(args.manifest)
    summaries, all_rows, budgets = [], [], []
    for model_index, checkpoint_path in enumerate(args.checkpoints):
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        if (checkpoint['manifest_hashes']['train'] == manifest_hash or
                set(checkpoint.get('train_pair_ids', [])) & {r['id'] for r in dataset.rows}):
            parser.error('Refusing to report training-manifest or overlapping training-pair scores as evaluation')
        args.output.mkdir(parents=True, exist_ok=True)
        label = f'{model_index + 1}_{checkpoint["config"]["model"]["name"]}'
        rows = []
        for index, pair in enumerate(dataset):
            noisy = pair['noisy'].unsqueeze(0).to(device)
            # Warm-up and timing use identical image, tiling, dtype and device.
            predict(model, noisy, args.tile_size, args.overlap)
            synchronize(device)
            started = time.perf_counter()
            for _ in range(args.repeats):
                prediction = predict(model, noisy, args.tile_size, args.overlap)
            synchronize(device)
            latency = (time.perf_counter() - started) * 1000 / args.repeats
            predicted, clean = as_hwc(prediction[0]), as_hwc(pair['clean'])
            input_image = as_hwc(pair['noisy'])
            scores = image_metrics(predicted, clean)
            baseline = image_metrics(input_image, clean)
            row = {'model': label, 'id': pair['id'], 'height': clean.shape[0], 'width': clean.shape[1],
                   **scores, 'noisy_psnr': baseline['psnr'], 'noisy_ssim': baseline['ssim'], 'latency_ms': latency}
            rows.append(row)
            if index < args.save_images:
                # Left to right: noisy / prediction / ground truth. Metrics use floats, PNGs use uint8.
                canvas = np.concatenate((input_image, predicted, clean), axis=1)
                Image.fromarray(np.rint(canvas * 255).clip(0, 255).astype(np.uint8)).save(
                    args.output / f'{label}_{index:04d}_noisy_prediction_gt.png')
        summary = {'model': label, 'checkpoint': str(checkpoint_path),
                   'checkpoint_sha256': file_hash(checkpoint_path), 'training_step': checkpoint['step'],
                   'parameters': sum(p.numel() for p in model.parameters()), 'pairs': len(rows),
                   **{k: float(np.mean([r[k] for r in rows])) for k in
                      ('psnr', 'ssim', 'noisy_psnr', 'noisy_ssim', 'latency_ms')},
                   'training_config': checkpoint['config'], 'training_manifests': checkpoint['manifest_hashes']}
        summaries.append(summary)
        all_rows.extend(rows)
        budget = {k: v for k, v in checkpoint['config'].items() if k != 'model'}
        budgets.append((budget, checkpoint['manifest_hashes']))
        print(f'{label}: PSNR={summary["psnr"]:.3f}, SSIM={summary["ssim"]:.5f}, '
              f'{summary["latency_ms"]:.1f} ms/image', flush=True)
    with (args.output / 'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    comparable = all(b == budgets[0] for b in budgets)
    save_json(args.output / 'summary.json', {
        'environment': environment(device), 'manifest': str(args.manifest), 'manifest_sha256': manifest_hash,
        'tile_size': args.tile_size, 'overlap': args.overlap, 'timing_repeats': args.repeats,
        'threads': args.threads, 'precision': 'float32', 'metric_protocol': 'RGB float [0,1]; mean image PSNR; Gaussian valid-region SSIM',
        'matched_configured_budget_and_manifests': comparable, 'models': summaries})
    if not comparable:
        print('NOTE: checkpoint configured training budgets/manifests differ; disclose this in comparisons.')


if __name__ == '__main__':
    main()
