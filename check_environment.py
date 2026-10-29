"""Check the local PyTorch device; optionally try one training step on real SIDD pairs."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from datasets.paired_dataset import PairedDataset
from models import build_model
from utils.runtime import choose_device, environment, save_json, seed_everything, synchronize


def check_training(config, data, device):
    """Exercise forward, backward and optimizer memory, without saving model weights."""
    seed_everything(config['seed'])
    dataset = PairedDataset(data / 'train.csv', config['patch_size'], training=True)
    rng = random.Random(config['seed'])
    pairs = [dataset.sample(rng.randrange(len(dataset)), rng) for _ in range(config['batch_size'])]
    noisy = torch.stack([pair['noisy'] for pair in pairs]).to(device)
    clean = torch.stack([pair['clean'] for pair in pairs]).to(device)
    model = build_model(config['model']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0, betas=(0.9, 0.999))
    amp = config.get('amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
        loss = F.l1_loss(model(noisy), clean)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss in real-data training check')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    synchronize(device)
    result = {'status': 'passed', 'real_pairs': len(pairs), 'patch_size': config['patch_size'],
              'amp_enabled': amp, 'parameters': sum(p.numel() for p in model.parameters()),
              'first_training_step_seconds': time.perf_counter() - started,
              'note': 'Memory/setup check only; no research metrics or trained weights produced.'}
    if device.type == 'cuda':
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved(device) / 2**30
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--require-gpu', action='store_true', help='Fail if the selected device is CPU')
    parser.add_argument('--config', type=Path, help='Also check a real-data optimizer step for this config')
    parser.add_argument('--data', type=Path, help='Prepared SIDD directory containing train.csv')
    parser.add_argument('--output', type=Path, help='Save a JSON report to a new filename')
    args = parser.parse_args()
    if bool(args.config) != bool(args.data):
        parser.error('--config and --data must be supplied together')
    if args.output and args.output.exists():
        parser.error('Report already exists; choose a new --output filename')
    torch.set_num_threads(4)
    device = choose_device(args.device)
    report = {'environment': environment(device), 'cuda_available': torch.cuda.is_available(),
              'cuda_build': torch.version.cuda, 'rocm_build': torch.version.hip,
              'mps_available': torch.backends.mps.is_available(), 'status': 'passed'}
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        report['gpu_memory_gib'] = {'free': free / 2**30, 'total': total / 2**30}
    # Verify actual device execution, not just driver detection. This is not an image dataset.
    try:
        values = torch.ones(16, 16, device=device)
        if (values @ values).sum().item() != 4096:
            raise RuntimeError('Unexpected device execution result')
    except RuntimeError as error:
        report['status'] = 'failed'
        report['reason'] = f'Device execution failed: {error}'
    if args.require_gpu and device.type == 'cpu':
        report['status'] = 'failed'
        report['reason'] = 'PyTorch selected CPU. Install the supported GPU build/driver before GPU training.'
    elif args.config and report['status'] == 'passed':
        config = json.loads(args.config.read_text(encoding='utf-8'))
        report['config'] = config
        try:
            report['training_check'] = check_training(config, args.data, device)
        except torch.OutOfMemoryError:
            report['status'] = 'failed'
            report['reason'] = 'Out of memory: close GPU apps, or lower batch/patch size for BOTH models in new configs.'
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, report)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
