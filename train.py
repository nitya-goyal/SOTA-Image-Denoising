"""Train U-Net or NAFNet on real SIDD pairs under a recorded step budget."""
import argparse
import csv
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from datasets.paired_dataset import PairedDataset
from metrics import image_metrics
from models import build_model
from utils.runtime import as_hwc, choose_device, environment, file_hash, save_json, seed_everything


def validate(model, dataset, device):
    model.eval()
    scores = []
    with torch.inference_mode():
        for pair in dataset:
            prediction = model(pair['noisy'].unsqueeze(0).to(device)).clamp(0, 1)[0]
            scores.append(image_metrics(as_hwc(prediction), as_hwc(pair['clean'])))
    model.train()
    return {key: float(np.mean([score[key] for score in scores])) for key in ('psnr', 'ssim')}


def assert_disjoint(train, val):
    if {r['group'] for r in train.rows} & {r['group'] for r in val.rows}:
        raise ValueError('Train/validation group overlap; regenerate disjoint scene splits')
    train_paths = {r[k] for r in train.rows for k in ('noisy', 'clean')}
    val_paths = {r[k] for r in val.rows for k in ('noisy', 'clean')}
    if train_paths & val_paths:
        raise ValueError('Train/validation images overlap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/sidd'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--resume', type=Path, help='last.pt in the same output directory')
    parser.add_argument('--stop-after', type=int, help='Stop at this total step without changing the LR schedule (smoke test)')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('max_steps', 'batch_size', 'patch_size', 'val_patch_size', 'eval_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            parser.error(f'{key} must be a positive integer')
    if min(config['patch_size'], config['val_patch_size']) < 16:
        parser.error('Patch sizes must be at least 16')
    if args.threads < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('--threads and --stop-after must be positive')
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    seed_everything(config['seed'])
    train_csv, val_csv = args.data / 'train.csv', args.data / 'val.csv'
    train = PairedDataset(train_csv, config['patch_size'], training=True)
    val = PairedDataset(val_csv, config['val_patch_size'])
    assert_disjoint(train, val)
    hashes = {'train': file_hash(train_csv), 'val': file_hash(val_csv)}
    if args.resume:
        if args.resume.name != 'last.pt' or args.resume.resolve().parent != args.output.resolve():
            parser.error('--resume must be last.pt inside --output to preserve the run history')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['manifest_hashes'] != hashes:
            parser.error('Resume config or data manifests differ from the saved run')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('Output is not empty; choose a new run folder or use --resume')
        checkpoint = None
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_steps'], eta_min=config['min_lr'])
    amp = config.get('amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start_step, best_psnr = 0, float('-inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_step, best_psnr = checkpoint['step'], checkpoint['best_psnr']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    finish = min(config['max_steps'], args.stop_after or config['max_steps'])
    if finish <= start_step:
        parser.error('No remaining steps; increase --stop-after or use the completed checkpoint')
    save_json(args.output / 'config.json', config)
    metadata = {'environment': environment(device), 'manifest_hashes': hashes,
                'parameters': sum(p.numel() for p in model.parameters()),
                'train_pairs': len(train), 'val_pairs': len(val),
                'validation_scope': f'deterministic center {config["val_patch_size"]}x{config["val_patch_size"]}',
                'loss': 'L1', 'amp_enabled': amp, 'threads': args.threads}
    save_json(args.output / ('resume_environment.json' if checkpoint else 'run_info.json'), metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    print('Starting training; checkpoints and validation history go to', args.output, flush=True)
    history = args.output / 'history.csv'
    if checkpoint and not history.is_file():
        raise ValueError('Missing history.csv; resume the complete original run folder')
    model.train()
    began = time.perf_counter()
    total_loss, interval_steps = 0.0, 0
    for step in range(start_step + 1, finish + 1):
        # Independent per-step RNG gives both models identical sampled pairs/crops.
        # It also makes data sampling resume exactly without worker prefetch state.
        rng = random.Random(config['seed'] * 1_000_003 + step)
        pairs = [train.sample(rng.randrange(len(train)), rng) for _ in range(config['batch_size'])]
        noisy = torch.stack([p['noisy'] for p in pairs]).to(device)
        clean = torch.stack([p['clean'] for p in pairs]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
            loss = F.l1_loss(model(noisy), clean)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        interval_steps += 1
        if step == 1 or step % 10 == 0:
            print(f'Step {step}/{config["max_steps"]}: L1={loss.item():.6f}', flush=True)
        if step % config['eval_every'] == 0 or step == finish:
            scores = validate(model, val, device)
            row = {'step': step, 'train_l1': total_loss / interval_steps, **scores,
                   'lr': scheduler.get_last_lr()[0], 'elapsed_this_session_s': time.perf_counter() - began}
            with history.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            improved = scores['psnr'] > best_psnr
            best_psnr = max(best_psnr, scores['psnr'])
            state = {'format_version': 1, 'config': config, 'step': step, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'scaler': scaler.state_dict(), 'best_psnr': best_psnr, 'manifest_hashes': hashes,
                     'train_pair_ids': [r['id'] for r in train.rows],
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
            # Atomic replacement avoids losing last.pt if interrupted during serialization.
            temporary = args.output / 'checkpoint.tmp'
            torch.save(state, temporary)
            temporary.replace(args.output / 'last.pt')
            if improved:
                torch.save(state, args.output / 'best.pt')
            print(f'Validation at {step}: PSNR={scores["psnr"]:.3f}, SSIM={scores["ssim"]:.5f}', flush=True)
            total_loss, interval_steps = 0.0, 0


if __name__ == '__main__':
    main()
