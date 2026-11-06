"""CPU checks for the 2026-11-05 extension baselines (MCPST-FSL, MetaDG)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_extension_20261103 import configs, sha, synthetic_batch, synthetic_context

ROOT = Path(__file__).resolve().parents[1]


def build(name, context, dataset='PEMS08', identity='structural'):
    from experiments.extension2_baselines_20261105 import ENTRY_POINTS
    module = __import__(ENTRY_POINTS[name], fromlist=['Model'])
    torch.manual_seed(0)
    c = configs(name)
    c.dloader_name = dataset
    c.model_family_cfg = {'model': {'node_identity': identity}}
    model = module.Model.from_runtime_context(c, context)
    if hasattr(model, 'set_scaler'):
        model.set_scaler({'mean': 50., 'std': 20.})
    return model


def test_source_copies_are_byte_identical():
    for name in ('MCPST', 'MetaDG'):
        manifest = json.loads((ROOT / f'model/comparison_model/{name}/SOURCE_MANIFEST.json').read_text())
        for entry in manifest['source_files']:
            assert not entry['modified'] and sha(ROOT / entry['copy']) == entry['sha256'] == sha(ROOT / entry['reference'])


@pytest.mark.parametrize('name', ('MCPST', 'MetaDG'))
def test_forward_backward_and_inference_blindness(name):
    context = synthetic_context()
    model = build(name, context)
    batch = synthetic_batch(context)
    model.train()
    out = model.forward_v4(batch, context)
    assert out['pred'].shape == (3, 12, context.num_nodes, 1)
    out['pred'].abs().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        reference = model.forward_v4(batch, context)['pred']
        poisoned = dict(batch, target=torch.full_like(batch['target'], 1e6),
                        lifecycle_context=torch.full_like(batch['lifecycle_context'], 7.),
                        loss_mask=torch.zeros_like(batch['loss_mask']),
                        decoder_available_mask=torch.zeros_like(batch['decoder_available_mask']))
        assert torch.equal(model.forward_v4(poisoned, context)['pred'], reference)


def test_mcpst_window_features_use_released_function_inside_window_only():
    context = synthetic_context()
    model = build('MCPST', context)
    batch = synthetic_batch(context)
    x = model.features(batch['base_visible_input'], batch['encoder_time_marks'])
    assert x.shape == (3, 12, context.num_nodes, 6)
    # degree channel is static topology; the other three depend only on the window
    assert torch.allclose(x[0, 0, :, 2], x[1, 5, :, 2])
    # sample 0 equals the released function applied to that window alone
    from model.comparison_model.MCPST.adapter import released
    _, data = released()
    tod = model.features(batch['base_visible_input'][:1], batch['encoder_time_marks'][:1])
    window = np.concatenate([np.clip(batch['base_visible_input'][0].numpy(), -5, 5),
                             tod[0, ..., 1:2].numpy()], axis=-1).astype(np.float32)
    assert np.allclose(data.enhanced_physics_features(window, model._raw_adjacency), x[0].numpy())


def test_mcpst_cached_spectral_decomposition_equals_released_loop():
    context = synthetic_context()
    model = build('MCPST', context)
    spectral = model.core.spectral_module
    adjacency = model.adjacency.unsqueeze(0).expand(5, -1, -1)
    laplacian = spectral.compute_normalized_laplacian(adjacency)
    cached = spectral.spectral_decomposition(laplacian)
    released = spectral.spectral_decomposition.original(laplacian)
    assert all(torch.equal(a, b) for a, b in zip(cached, released))  # CPU copies are bit-identical
    noisy = laplacian.clone()
    noisy[1] += 1e-7  # GPU-level rounding between copies still reuses the first decomposition
    again = spectral.spectral_decomposition(noisy)
    assert all(torch.equal(a, b) for a, b in zip(again, cached))
    far = laplacian.clone()
    far[1] += 1e-3  # a genuinely different matrix falls back to the released loop
    assert torch.equal(spectral.spectral_decomposition(far)[0], spectral.spectral_decomposition.original(far)[0])


def test_metadg_teacher_feeds_only_legal_targets():
    from model.comparison_model.MetaDG.adapter import LegalTeacher
    target = torch.randn(2, 12, 5, 1)
    available = torch.rand(2, 12, 5) > .5
    teacher = LegalTeacher(target, available)
    teacher.last = torch.full((2, 5, 1), -9.)
    step = teacher[:, 3, ...]
    assert torch.equal(step[available[:, 3]], target[:, 3][available[:, 3]])
    assert torch.all(step[~available[:, 3]] == -9.)


def test_metadg_node_identity_switch():
    context = synthetic_context()
    structural = build('MetaDG', context, identity='structural')
    released = build('MetaDG', context, identity='released')
    assert not structural.core.parametrizations.node_emb.original.requires_grad
    assert released.core.node_emb.requires_grad and not hasattr(released.core, 'parametrizations')


def test_metadg_and_mcpst_objectives():
    from experiments.extension2_baselines_20261105 import compute_losses
    pred = torch.randn(2, 12, 4, 1, requires_grad=True)
    target = torch.randn(2, 12, 4, 1)
    batch = {'loss_mask': torch.ones(2, 12, 4, dtype=torch.bool), 'target': target}
    out = compute_losses({'pred': pred, 'source_objective': 'metadg_released_loss', 'metadg_loss': 'huber'},
                         batch, 'mae', {'std': 3.})
    raw = (pred - target) * 3.
    assert torch.allclose(out['optimize'], torch.nn.functional.huber_loss(raw, torch.zeros_like(raw)))
    unc = torch.rand(2, 12, 4, 1)
    meta = {'pred': pred, 'source_objective': 'mcpst_physics_informed', 'mcpst_uncertainty': unc,
            'mcpst_order_parameter': torch.full((2, 12), .5), 'mcpst_spectral_gap': torch.zeros(2, 12, 4),
            'mcpst_physics_weights': torch.ones(3) / 3}
    out = compute_losses(meta, batch, 'mse', {'std': 3.})
    err = pred - target
    expected = 2 * err.square().mean() + .01 * (unc - err.abs().detach()).square().mean()
    assert torch.allclose(out['optimize'], expected)
