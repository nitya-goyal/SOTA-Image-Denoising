import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

from models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install the correct PyTorch build or use --device cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple MPS unavailable; use --device cpu')
    return torch.device(name)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'platform': platform.platform(), 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)}


def save_json(path, value):
    # JSON does not support Infinity; preserve exact-match PSNR as the string "inf".
    def clean(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def load_checkpoint(path, device):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint required: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Expected a checkpoint produced by this repository; external weights need explicit conversion')
    model = build_model(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint


def as_hwc(tensor):
    return tensor.detach().float().cpu().permute(1, 2, 0).numpy()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
