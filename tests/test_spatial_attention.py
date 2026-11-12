"""Spatial variant shape, checkpoint, and residual-gradient regression checks."""
import pytest
import torch

from models import build_model
from utils.runtime import load_checkpoint


def options(name):
    return dict(name=name, width=8, middle_blk_num=1,
                enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])


def test_spatial_checkpoint_and_odd_dimensions(tmp_path):
    torch.manual_seed(42)
    config = options('nafnet_spatial')
    model = build_model(config).eval()
    image = torch.rand(1, 3, 19, 23)
    with torch.no_grad():
        expected = model(image)
    assert expected.shape == image.shape
    path = tmp_path / 'spatial.pt'
    torch.save(dict(format_version=1, config={'model': config}, model=model.state_dict(), step=2), path)
    restored, _ = load_checkpoint(path, torch.device('cpu'))
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)


def test_spatial_gradient_after_residual_activation():
    torch.manual_seed(42)
    model = build_model(options('nafnet_spatial'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0)
    image, target = torch.rand(2, 3, 20, 24), torch.rand(2, 3, 20, 24)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(model(image), target).backward()
        grads = [p.grad for n, p in model.named_parameters() if 'spatial_att' in n]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        if step == 0:
            assert all(torch.count_nonzero(g) == 0 for g in grads)
        else:
            assert sum(g.abs().sum().item() for g in grads) > 0
        optimizer.step()


def test_spatial_parameter_cost_and_conflicting_settings():
    base = build_model(options('nafnet'))
    spatial = build_model(options('nafnet_spatial'))
    assert sum(p.numel() for p in spatial.parameters()) - sum(p.numel() for p in base.parameters()) == 5 * 99
    with pytest.raises(ValueError, match='Conflicting settings'):
        build_model(dict(**options('nafnet'), attention='spatial'))
