"""CPU checks for the 2026-11-03 extension baselines and whole-mechanism ablations."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from lib.runtime_context import RuntimeDataContext

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Merlin', 'HDTTS', 'STONE', 'STOP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_context(num_nodes=110, known=70, phase="train", seed=0):
    rng = np.random.RandomState(seed)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in (i - 1, i + 1, i + 3):
            adjacency[i, j % num_nodes] = rng.uniform(.2, 1.)
    np.fill_diagonal(adjacency, 1.)
    nodes = np.arange(num_nodes)
    known_nodes = np.sort(rng.choice(num_nodes, known, replace=False))
    rest = np.setdiff1d(nodes, known_nodes)
    val, test = rest[:len(rest) // 2], rest[len(rest) // 2:]
    scope = {'train': known_nodes, 'validation': np.union1d(known_nodes, val),
             'test': np.union1d(known_nodes, test)}[phase]
    return RuntimeDataContext(
        dataset='PEMS08', phase=phase, target='q', target_channel=0, payload_sha256='synthetic',
        node_ids=nodes, node_ids_sha256='synthetic', adjacency=torch.from_numpy(adjacency),
        adjacency_sha256='synthetic', positions=None, layout_parts=(tuple(nodes.tolist()),),
        layout_sha256='synthetic', layout_algorithm='graph_native_component_bfs',
        known_nodes=known_nodes, validation_unseen=val, test_unseen=test, signal_scope=scope,
        readiness={})


def configs(model):
    return SimpleNamespace(model=model, raw_enc_in=1, enc_in=1, c_out=1, input_len=12, pred_len=12,
                           seed=32, slide_step=1, lifecycle_training_strategy='lcbt_v4',
                           dloader_name='PEMS08', input_features='q', target_features='q',
                           device='cpu')


def synthetic_batch(context, batch=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = context.num_nodes
    scope = torch.zeros(n, dtype=torch.bool)
    scope[torch.as_tensor(context.signal_scope)] = True
    available = (torch.rand(batch, 12, n, generator=g) > .3) & scope
    base = torch.where(available, torch.randn(batch, 12, n, generator=g), torch.zeros(())).unsqueeze(-1)
    target_available = (torch.rand(batch, 12, n, generator=g) > .2) & scope
    target = torch.where(target_available, torch.randn(batch, 12, n, generator=g), torch.zeros(())).unsqueeze(-1)
    minute = torch.randint(0, 12, (batch, 12), generator=g) * 5.
    hour = torch.randint(0, 24, (batch, 12), generator=g).float()
    dow = torch.randint(0, 7, (batch, 12), generator=g).float()
    marks = torch.stack([minute, hour, dow], -1)
    identity = {k: [v] * batch if isinstance(v, str) else v for k, v in context.identity().items()}
    return {
        'base_visible_input': base, 'input_available_mask': available,
        'encoder_time_marks': marks, 'decoder_time_marks': marks.clone(),
        'target': target, 'decoder_available_mask': target_available,
        'target_available_mask': target_available, 'loss_mask': target_available,
        'lifecycle_context': torch.rand(batch, 12, n, 6), 'context_identity': identity,
        'origin': torch.arange(batch), 'scenario': ['steady'] * batch,
    }


def build(name, context):
    from experiments.extension_baselines_20261103 import ENTRY_POINTS
    module = __import__(ENTRY_POINTS[name], fromlist=['Model'])
    torch.manual_seed(0)
    return module.Model.from_runtime_context(configs(name), context)


def test_source_copies_are_byte_identical():
    for name in METHODS:
        manifest = json.loads((ROOT / f'model/comparison_model/{name}/SOURCE_MANIFEST.json').read_text())
        assert manifest['source_files']
        for entry in manifest['source_files']:
            assert not entry['modified']
            assert sha(ROOT / entry['copy']) == entry['sha256']
            reference = ROOT / entry['reference']
            if reference.exists():
                assert sha(reference) == entry['sha256']


def test_window_fill_is_pandas_ffill_bfill_inside_each_window():
    pytest.importorskip('torch_sparse')
    from model.comparison_model.HDTTS.adapter import window_ffill_bfill
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 12, 5, 1, generator=g)
    mask = torch.rand(4, 12, 5, generator=g) > .6
    mask[0, :, 0] = False
    out = window_ffill_bfill(x, mask)
    for b in range(4):
        for n in range(5):
            series = pd.Series(np.where(mask[b, :, n], x[b, :, n, 0], np.nan))
            expected = series.ffill().bfill().fillna(0.).to_numpy()
            assert np.allclose(out[b, :, n, 0].numpy(), expected)
    # values hidden by the mask are never read
    poisoned = torch.where(mask[..., None], x, torch.full_like(x, 1e6))
    assert torch.equal(window_ffill_bfill(poisoned, mask), out)


@pytest.mark.parametrize('name', METHODS)
def test_forward_backward_and_inference_blindness(name):
    if name == 'HDTTS':
        pytest.importorskip('torch_sparse')
    context = synthetic_context()
    model = build(name, context)
    batch = synthetic_batch(context)
    if name == 'Merlin':
        from model.comparison_model.Merlin.adapter import MISSING_RATES
        model.train()
        history = torch.where(torch.rand_like(batch['base_visible_input']) > 0, batch['base_visible_input'], 0.)
        loss = model.teacher_objective(batch, context, history)
        loss.backward()
        assert all(p.grad is None for p in model.student.parameters())
        model.zero_grad(set_to_none=True)
        model.freeze_teacher()
        keep = torch.rand(len(MISSING_RATES), 3, 12, context.num_nodes) > .5
        terms = model.student_objective(batch, context, history, keep, epoch=5)
        terms['loss'].backward()
        assert all(p.grad is None for p in model.teacher.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.parameters())
    else:
        model.train()
        out = model.forward_v4(batch, context)
        if name == 'STONE':
            env = out['stone_environments']
            assert env.shape == (2, 3, 12, context.num_nodes, 1)
            loss = env.abs().mean() - sum(out['stone_log_p'])
        else:
            assert out['pred'].shape == (3, 12, context.num_nodes, 1)
            loss = out['pred'].abs().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    model.eval()
    with torch.no_grad():
        reference = model.forward_v4(batch, context)['pred']
        poisoned = dict(batch, target=torch.full_like(batch['target'], 1e6),
                        lifecycle_context=torch.full_like(batch['lifecycle_context'], 7.),
                        loss_mask=torch.zeros_like(batch['loss_mask']),
                        decoder_available_mask=torch.zeros_like(batch['decoder_available_mask']))
        assert torch.equal(model.forward_v4(poisoned, context)['pred'], reference)
    assert reference.shape == (3, 12, context.num_nodes, 1)


def test_structural_node_identity_replaces_learned_tables():
    pytest.importorskip('torch_sparse')
    context = synthetic_context()
    merlin = build('Merlin', context)
    for network in (merlin.teacher, merlin.student):
        assert not network.parametrizations.node_emb.original.requires_grad
        assert network.node_emb.shape == (context.num_nodes, 64)
    hdtts = build('HDTTS', context)
    from layers.structural_node_encoder import StructuralNodeEncoder
    assert isinstance(hdtts.core.encoder_time.embeddings, StructuralNodeEncoder)
    assert not any('NodeEmbedding' in type(m).__name__ for m in hdtts.modules())


def test_stone_side_information_uses_topology_and_known_nodes_only():
    from model.comparison_model.STONE.adapter import frechet_embedding
    from model.comparison_model.source_loader import load_released
    from model.comparison_model.STONE.adapter import SOURCE
    ssi = load_released(SOURCE, ['src.utils.spatial_side_information'])[0][
        'src.utils.spatial_side_information'].Spatial_Side_Information
    context = synthetic_context()
    a = frechet_embedding(ssi, context.adjacency.numpy(), context.known_nodes, 32)
    b = frechet_embedding(ssi, context.adjacency.numpy(), context.known_nodes, 32)
    assert torch.equal(a, b) and a.shape[0] == context.num_nodes
    # known-node rows have zero distance to anchor sets containing themselves only sometimes;
    # the embedding must not depend on anything but adjacency, K and the seed.
    other = synthetic_context(seed=5)
    assert not torch.equal(frechet_embedding(ssi, other.adjacency.numpy(), other.known_nodes, 32), a)


def test_stone_objective_matches_released_formula_on_dense_mask():
    from experiments.extension_baselines_20261103 import compute_losses
    env = torch.randn(2, 3, 12, 4, 1, requires_grad=True)
    target = torch.randn(3, 12, 4, 1)
    log_p = [torch.tensor(-2., requires_grad=True), torch.tensor(-3., requires_grad=True)]
    batch = {'loss_mask': torch.ones(3, 12, 4, dtype=torch.bool), 'target': target}
    out = compute_losses({'pred': env[0], 'stone_environments': env, 'stone_log_p': log_p,
                          'source_objective': 'stone_environments'}, batch, 'mae', {'std': 2.})
    errors = ((env - target.unsqueeze(0)).abs() * 2.)
    # released masked_mae_train with an all-true mask: mean and var over all elements
    assert torch.allclose(out['base'], errors.mean())
    assert torch.allclose(out['optimize'], errors.mean() + errors.var() + 5. * errors.var().detach())
    out['optimize'].backward()
    assert torch.allclose(log_p[0].grad, -errors.var().detach())


def test_clipped_optimizer_clips_only_listed_groups():
    from experiments.extension_baselines_20261103 import clipped
    a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(3))
    opt = clipped(torch.optim.SGD, (0,))([{'params': [a]}, {'params': [b]}], lr=1.)
    a.grad, b.grad = torch.full((3,), 100.), torch.full((3,), 100.)
    opt.step()
    assert torch.allclose(a.norm(), torch.tensor(5.)) and torch.allclose(b, torch.full((3,), -100.))
    assert isinstance(opt, torch.optim.SGD)
