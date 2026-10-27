"""Small artificial fixtures test software only; they are NOT SIDD results."""
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from scipy.io import savemat
import torch

from datasets.paired_dataset import PairedDataset
from metrics import image_metrics
from models import build_model
from models.nafnet import LayerNorm2d
from prepare_sidd import discover_pairs, export_validation, group_split, write_manifest
from utils.inference import predict
from utils.runtime import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.fixture
def raw_pairs(tmp_path):
    rng = np.random.default_rng(123)
    source = tmp_path / 'raw'
    for scene in range(1, 5):
        # Two camera/ISO instances of the same physical scene.
        for camera in ('S6', 'IP'):
            folder = source / f'{scene:04d}_{scene:03d}_{camera}_00100_00060_3200_L'
            folder.mkdir(parents=True)
            clean = rng.integers(16, 240, (35, 37, 3), dtype=np.uint8)
            noisy = (clean.astype(np.int16) + 5).clip(0, 255).astype(np.uint8)
            Image.fromarray(clean).save(folder / 'GT_SRGB_010.PNG')
            Image.fromarray(noisy).save(folder / 'NOISY_SRGB_010.PNG')
    return source


def test_pairing_and_scene_split(raw_pairs):
    pairs = discover_pairs(raw_pairs)
    assert len(pairs) == 8
    splits = group_split(pairs)
    assert splits == group_split(pairs)
    groups = [{r['group'] for r in rows} for rows in splits.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(map(len, splits.values())) == 8
    pairs[0]['clean'].unlink()
    with pytest.raises(ValueError, match='partner'):
        discover_pairs(raw_pairs)


def test_dataset_augmentations_stay_aligned(raw_pairs, tmp_path):
    path = tmp_path / 'pairs.csv'
    write_manifest(path, discover_pairs(raw_pairs))
    data = PairedDataset(path, patch_size=32, training=True)
    for seed in range(8):
        pair = data.sample(0, random.Random(seed))
        assert pair['noisy'].shape == (3, 32, 32)
        assert torch.allclose(pair['noisy'] - pair['clean'], torch.full((3, 32, 32), 5/255), atol=1e-7)
    rows = list(csv.DictReader(path.open()))
    write_manifest(path, discover_pairs(raw_pairs)[:1]*2)
    with pytest.raises(ValueError, match='Duplicate'):
        PairedDataset(path)
    assert not Path(rows[0]['noisy']).is_absolute()


@pytest.mark.parametrize('name', ['unet', 'nafnet'])
def test_model_arbitrary_sizes_and_gradients(name):
    options = {'name': name, 'width': 2}
    if name == 'nafnet':
        options.update(enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], middle_blk_num=1)
    model = build_model(options)
    for size in [(32, 32), (35, 37), (1, 3)]:
        image = torch.rand(1, 3, *size)
        output = model(image)
        assert output.shape == image.shape
        loss = (output - image).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad()


def test_layer_norm_backward():
    norm = LayerNorm2d(3).double()
    sample = torch.randn(1, 3, 2, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(norm, sample)


def test_metrics_and_tiled_identity():
    rng = np.random.default_rng(3)
    image = rng.random((21, 23, 3)) * 0.5
    assert image_metrics(image, image) == {'psnr': float('inf'), 'ssim': 1.0}
    assert image_metrics(image + 0.1, image)['psnr'] == pytest.approx(20)
    # Independent direct 11x11 valid-window SSIM reference at one output location.
    coords = np.arange(-5, 6)
    gaussian = np.exp(-coords**2 / (2*1.5**2))
    kernel = np.outer(gaussian, gaussian)
    kernel /= kernel.sum()
    a, b = image[:11, :11, 0], (image + 0.1)[:11, :11, 0]
    ma, mb = (a*kernel).sum(), (b*kernel).sum()
    va, vb = (a*a*kernel).sum() - ma*ma, (b*b*kernel).sum() - mb*mb
    cov = (a*b*kernel).sum() - ma*mb
    reference = ((2*ma*mb+0.01**2)*(2*cov+0.03**2))/((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))
    patch_a = np.repeat(a[..., None], 3, axis=2)
    patch_b = np.repeat(b[..., None], 3, axis=2)
    assert image_metrics(patch_a, patch_b)['ssim'] == pytest.approx(reference, abs=1e-10)
    tensor = torch.rand(1, 3, 35, 37)
    assert torch.allclose(predict(torch.nn.Identity(), tensor, 16, 5), tensor)
    with pytest.raises(ValueError):
        predict(torch.nn.Identity(), tensor, 16, 16)


def test_official_mat_conversion(tmp_path):
    array = np.arange(2*3*16*16*3, dtype=np.uint8).reshape(2, 3, 16, 16, 3)
    a, b = tmp_path / 'noisy.mat', tmp_path / 'clean.mat'
    savemat(a, {'ValidationNoisyBlocksSrgb': array})
    savemat(b, {'ValidationGtBlocksSrgb': array})
    rows = export_validation(a, b, tmp_path / 'export')
    assert len(rows) == 6
    assert np.array_equal(np.array(Image.open(rows[4]['noisy'])), array[1, 1])


def run_cli(*args, succeeds=True):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=180)
    if succeeds:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_end_to_end_train_resume_compare_and_denoise(raw_pairs, tmp_path):
    data = tmp_path / 'prepared'
    run_cli('prepare_sidd.py', '--source', raw_pairs, '--output', data)
    checkpoints = []
    for name in ('unet', 'nafnet'):
        options = {'name': name, 'width': 2}
        if name == 'nafnet':
            options.update(enc_blk_nums=[1]*4, dec_blk_nums=[1]*4, middle_blk_num=1)
        config = {'model': options, 'seed': 42, 'max_steps': 3, 'batch_size': 1,
                  'patch_size': 32, 'val_patch_size': 32, 'eval_every': 1,
                  'lr': 0.0002, 'min_lr': 0.000001, 'amp': False}
        config_path = tmp_path / f'{name}.json'
        config_path.write_text(json.dumps(config))
        output = tmp_path / name
        common = ['train.py', '--config', config_path, '--data', data, '--output', output, '--device', 'cpu', '--threads', '1']
        run_cli(*common, '--stop-after', 1)
        run_cli(*common, '--resume', output / 'last.pt')
        # Exact CPU resume agrees with an uninterrupted run at the same total step.
        continuous = tmp_path / f'{name}_continuous'
        run_cli('train.py', '--config', config_path, '--data', data, '--output', continuous, '--device', 'cpu', '--threads', 1)
        resumed = torch.load(output / 'last.pt', weights_only=True)
        direct = torch.load(continuous / 'last.pt', weights_only=True)
        assert resumed['step'] == 3
        for key in resumed['model']:
            assert torch.equal(resumed['model'][key], direct['model'][key]), key
        assert len(list(csv.DictReader((output / 'history.csv').open()))) == 3
        checkpoints.append(output / 'best.pt')
    result_dir = tmp_path / 'results'
    run_cli('evaluate.py', '--manifest', data / 'test.csv', '--checkpoints', *checkpoints,
            '--output', result_dir, '--device', 'cpu', '--tile-size', 32, '--overlap', 8, '--threads', 1)
    summary = json.loads((result_dir / 'summary.json').read_text())
    assert len(summary['models']) == 2
    assert all(np.isfinite(model['psnr']) for model in summary['models'])
    source = discover_pairs(raw_pairs)[0]['noisy']
    destination = tmp_path / 'denoised.png'
    run_cli('denoise.py', '--input', source, '--output', destination, '--checkpoint', checkpoints[1], '--device', 'cpu')
    assert Image.open(destination).size == Image.open(source).size
    failed = run_cli('evaluate.py', '--manifest', data / 'train.csv', '--checkpoints', checkpoints[0],
                     '--output', tmp_path / 'bad_eval', '--device', 'cpu', succeeds=False)
    assert 'training-manifest' in failed.stderr
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'missing.pt', torch.device('cpu'))
