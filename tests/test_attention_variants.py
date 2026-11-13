"""Unit tests verifying NAFNet attention variants and backward compatibility."""
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from models import build_model
from models.nafnet import NAFNet, NAFBlock, SpatialAttention, CBAMChannelAttention
from utils.runtime import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_standard_nafnet_backward_compatibility():
    """Verify that default NAFNet and attention='sca' are identical."""
    torch.manual_seed(42)
    model_default = build_model({'name': 'nafnet', 'width': 8, 'middle_blk_num': 1,
                                 'enc_blk_nums': [1]*2, 'dec_blk_nums': [1]*2})
    torch.manual_seed(42)
    model_explicit = build_model({'name': 'nafnet', 'width': 8, 'middle_blk_num': 1,
                                  'enc_blk_nums': [1]*2, 'dec_blk_nums': [1]*2,
                                  'attention': 'sca'})
    x = torch.randn(2, 3, 32, 32)
    out_default = model_default(x)
    out_explicit = model_explicit(x)
    assert torch.allclose(out_default, out_explicit, atol=1e-7)

    # Check state_dict keys match standard NAFNet structure
    assert set(model_default.state_dict().keys()) == set(model_explicit.state_dict().keys())
    assert all('channel_att' not in k and 'spatial_att' not in k for k in model_default.state_dict().keys())


def test_existing_standard_nafnet_checkpoint_loads_strictly():
    """Verify that the previously trained standard NAFNet checkpoint loads with strict=True
    and executes deterministically on CPU."""
    ckpt_path = ROOT / 'runs' / 'nafnet_sidd_small_seed42' / 'best.pt'
    if ckpt_path.is_file():
        model, ckpt = load_checkpoint(ckpt_path, torch.device('cpu'))
        assert model.attention == 'sca'
        assert ckpt['config']['model']['name'] == 'nafnet'

        # Test deterministic CPU evaluation
        torch.manual_seed(123)
        inp = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out1 = model(inp)
            out2 = model(inp)
        assert torch.equal(out1, out2)


def test_conflicting_attention_settings_rejected():
    """Verify that passing conflicting model-name and attention raises ValueError."""
    with pytest.raises(ValueError, match="Conflicting settings"):
        build_model({'name': 'nafnet', 'attention': 'spatial'})

    with pytest.raises(ValueError, match="Conflicting settings"):
        build_model({'name': 'nafnet_spatial', 'attention': 'channel'})

    with pytest.raises(ValueError, match="Conflicting settings"):
        build_model({'name': 'nafnet_channel', 'attention': 'combined'})

    with pytest.raises(ValueError, match="Conflicting settings"):
        build_model({'name': 'nafnet_combined', 'attention': 'sca'})

    # Matching settings must succeed
    assert build_model({'name': 'nafnet_spatial', 'attention': 'spatial'}).attention == 'spatial'
    assert build_model({'name': 'nafnet_channel', 'attention': 'channel'}).attention == 'channel'
    assert build_model({'name': 'nafnet_combined', 'attention': 'combined'}).attention == 'combined'


def test_exact_instantiated_parameter_counts():
    """Verify actual parameter counts of all four instantiated models."""
    base = {'width': 32, 'middle_blk_num': 12, 'enc_blk_nums': [2, 2, 4, 8], 'dec_blk_nums': [2, 2, 2, 2]}
    p_nafnet = sum(p.numel() for p in build_model({'name': 'nafnet', **base}).parameters())
    p_spatial = sum(p.numel() for p in build_model({'name': 'nafnet_spatial', **base}).parameters())
    p_channel = sum(p.numel() for p in build_model({'name': 'nafnet_channel', **base}).parameters())
    p_combined = sum(p.numel() for p in build_model({'name': 'nafnet_combined', **base}).parameters())

    assert p_nafnet == 29159715
    assert p_spatial == 29163279
    # CBAM channel attention reduces parameters vs full 1x1 SCA
    assert p_channel == 25730443
    assert p_combined == 25734007
    # Spatial attention adds exactly 3564 parameters (36 blocks * 99 params/block)
    assert p_spatial - p_nafnet == 3564
    assert p_combined - p_channel == 3564


def test_multi_step_gradient_flow_after_beta_activation():
    """Verify that because beta starts at 0, first-branch attention parameters initially
    receive zero gradients at step 1, but after an optimizer step updates beta (beta != 0),
    active gradient flow reaches the attention parameters."""
    for model_name in ['nafnet_spatial', 'nafnet_channel', 'nafnet_combined']:
        config = {'name': model_name, 'width': 32, 'middle_blk_num': 1,
                  'enc_blk_nums': [1]*2, 'dec_blk_nums': [1]*2}
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Before any step, all beta parameters are strictly 0
        assert all(torch.equal(b, torch.zeros_like(b)) for b in [m.beta for m in model.modules() if isinstance(m, NAFBlock)])

        # Step 1: Forward & backward
        x = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        loss1 = F.l1_loss(model(x), target)
        loss1.backward()

        # At step 1, because beta=0, attention parameters have zero gradient
        for module in model.modules():
            if isinstance(module, SpatialAttention):
                assert module.conv.weight.grad is not None
                assert torch.allclose(module.conv.weight.grad, torch.zeros_like(module.conv.weight.grad))
            elif isinstance(module, CBAMChannelAttention):
                assert module.mlp[0].weight.grad is not None
                assert torch.allclose(module.mlp[0].weight.grad, torch.zeros_like(module.mlp[0].weight.grad))

        # Optimizer step updates beta because dloss/dbeta != 0
        optimizer.step()
        optimizer.zero_grad()

        # Verify betas are now non-zero
        betas = [m.beta for m in model.modules() if isinstance(m, NAFBlock)]
        assert all(not torch.equal(b, torch.zeros_like(b)) for b in betas), "All Betas should be updated to non-zero"

        # Step 2: Forward & backward after beta becomes active
        loss2 = F.l1_loss(model(x), target)
        loss2.backward()

        # Gradients now actively flow into the attention parameters!
        has_spatial = any(isinstance(m, SpatialAttention) for m in model.modules())
        has_channel = any(isinstance(m, CBAMChannelAttention) for m in model.modules())

        if has_spatial:
            spatial_grads = [m.conv.weight.grad.abs().sum().item() for m in model.modules() if isinstance(m, SpatialAttention)]
            assert sum(spatial_grads) > 0, "Spatial attention should receive non-zero active gradients"
            assert all(torch.isfinite(m.conv.weight.grad).all() for m in model.modules() if isinstance(m, SpatialAttention))

        if has_channel:
            channel_grads = [m.mlp[0].weight.grad.abs().sum().item() for m in model.modules() if isinstance(m, CBAMChannelAttention)]
            assert sum(channel_grads) > 0, "Channel attention should receive non-zero active gradients"
            assert all(torch.isfinite(m.mlp[0].weight.grad).all() for m in model.modules() if isinstance(m, CBAMChannelAttention))


@pytest.mark.parametrize('model_name', ['nafnet', 'nafnet_spatial', 'nafnet_channel', 'nafnet_combined'])
def test_attention_variants_checkpoint_roundtrip(tmp_path, model_name):
    """Verify that saving and strictly loading checkpoints works for all variants."""
    config = {'name': model_name, 'width': 8, 'middle_blk_num': 1,
              'enc_blk_nums': [1]*2, 'dec_blk_nums': [1]*2}
    model = build_model(config)
    save_path = tmp_path / f'{model_name}_ckpt.pt'

    state = {
        'format_version': 1,
        'config': {'model': config},
        'step': 1,
        'model': model.state_dict(),
    }
    torch.save(state, save_path)

    loaded_model, loaded_ckpt = load_checkpoint(save_path, torch.device('cpu'))
    assert loaded_ckpt['config']['model']['name'] == model_name
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], loaded_model.state_dict()[k])
