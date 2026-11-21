"""Study-only interventions (2026-11-16 supplement). Production ALFNet unchanged.

Three keys extend the earlier adapters, which stay byte for byte as they were.

``no_role``  -- w/o Observation Role. The four role channels of ALFNet's assembled
    input are replaced by the observation mask (the substitution already used by
    ``mask_only`` for the role channels and by ``no_age`` for the progress
    channel), so SAHC, ALFTemporal and ALFGraph (encoder and router) receive
    state progress and mask but no role. Output refinement uses the
    role-independent constant multiplier ``mean(role_scale)``, as in
    ``mask_only``/``no_os``. Batch roles, teacher forcing and the role-balanced
    objective keep their true labels.

``no_mask``  -- w/o Observation Mask. No learnable computation receives the mask:
    the mask reaching the GRIN spatial decoders, the first GRU cells and the
    bidirectional merge MLP is zeroed, and the mask channel handed to
    ALFTemporal and ALFGraph is zeroed after completion (the ``no_os``
    convention for the mask, applied to the mask alone). Role and state progress
    are untouched. The true mask still decides which values are kept as
    observations and still drives the non-learned interpolation/centering, the
    loss, the evaluation and teacher forcing.

``bfs_layout`` (context, not a model key) -- the spatial partition of a dataset
    with verified coordinates is built by the graph-native component BFS used
    for the flow datasets instead of the KD-tree. Only the partition membership
    changes: coordinates still feed the structural node features, and
    ``recur_times`` and the partition capacity are the anchor's.

Construct the complete model first (paired initial values), then apply.
"""
import dataclasses
import hashlib
from contextlib import contextmanager
from unittest.mock import patch

import torch

from experiments.alfnet_extension_ablation_adapter import KEYS as EXTENSION_KEYS
from experiments.alfnet_extension_ablation_adapter import apply_intervention as apply_extension_intervention
from experiments.alfnet_extension_ablation_adapter import constant_refiner_scale
from experiments.alfnet_ablation_adapter import KEYS as BASE_KEYS

KEYS = ('no_role', 'no_mask')


def remove_role(module, args):
    args = list(args)
    x = args[1].clone()
    mask = x[..., module.mask_index:module.mask_index + 1]
    x[..., module.role_start:module.role_end] = mask
    args[1] = x
    return tuple(args)


def _blind_decoder_mask(module, args, kwargs):
    kwargs = dict(kwargs)
    kwargs['m'] = torch.zeros_like(kwargs['m'])
    return args, kwargs


def _zero_slice(start, stop):
    def hook(module, args):
        x = args[0]
        x = torch.cat([x[:, :start], torch.zeros_like(x[:, start:stop]), x[:, stop:]], dim=1)
        return (x,) + tuple(args[1:])
    return hook


def blind_grin_mask(imputer, raw_dim):
    """Zero every learned use of the mask inside the shared BiGRIL; keep role/progress."""
    bigril = imputer.bigril
    handles = []
    for rnn in (bigril.fwd_rnn, bigril.bwd_rnn):
        handles.append(rnn.spatial_decoder.register_forward_pre_hook(_blind_decoder_mask, with_kwargs=True))
        if len(rnn.cells) != 1:
            raise ValueError('no_mask expects the final one-layer GRIL')
        # update_state input: cat([x_s (raw), m_s (raw), u_s]); zero m_s only.
        handles.append(rnn.cells[0].register_forward_pre_hook(_zero_slice(raw_dim, 2 * raw_dim)))
    if not isinstance(bigril.out, torch.nn.Sequential) or bigril.emb is not None:
        raise ValueError('no_mask expects the final mlp merge without node embedding')
    hidden = bigril.fwd_rnn.hidden_size
    # merge input: cat([fwd_repr (2h), bwd_repr (2h), mask]).
    handles.append(bigril.out.register_forward_pre_hook(_zero_slice(4 * hidden, 4 * hidden + raw_dim)))
    return handles


def zero_mask_after_completion(mask_index):
    def hook(module, args, filled):
        assert filled.shape[-1] == mask_index + 1
        return torch.cat([filled[..., :mask_index], torch.zeros_like(filled[..., mask_index:])], dim=-1)
    return hook


def apply_intervention(session):
    recipe = session.args.model_family_cfg.get('recipe', {})
    key = recipe.get('study_ablation', 'reference')
    if key in BASE_KEYS or key in EXTENSION_KEYS or key == 'reference':
        return apply_extension_intervention(session)
    assert session.model_name == 'ALFNet' and key in KEYS, key
    model = session.model
    assert not hasattr(model, 'study_ablation'), 'intervention must only be applied once'
    assert model.observation_state_mode == 'full' and model.graph_expert.router_mode != 'static'
    model.study_ablation = key
    if key == 'no_role':
        model._supplement_hooks = [model.register_forward_pre_hook(remove_role),
                                   model.graph_refiner.register_forward_pre_hook(constant_refiner_scale)]
    else:
        imputer = model.shared_input_imputer
        assert imputer.observation_mask_idx == model.mask_index
        assert model.mask_index not in imputer.extra_feature_indices
        model._supplement_hooks = blind_grin_mask(imputer, model.raw_enc_in)
        model._supplement_hooks.append(imputer.register_forward_hook(zero_mask_after_completion(model.mask_index)))
    return session


# --- BFS layout on a dataset that has verified coordinates -------------------------


def bfs_context(original):
    """Wrap ``build_runtime_context`` so the partition is the graph-native BFS one."""
    from lib.graph_native_layout import LAYOUT_ALGORITHM, LAYOUT_VERSION, layout_parts

    def build(dataset, phase, seed=32, manifest_dir=None):
        ctx = original(dataset, phase, seed=seed, manifest_dir=manifest_dir)
        assert ctx.positions is not None and ctx.layout_algorithm == 'kdtree_on_verified_wgs84', \
            'the BFS study applies only to datasets partitioned by the KD-tree'
        recur_times = (len(ctx.layout_parts) - 1).bit_length()
        assert 2 ** recur_times == len(ctx.layout_parts)
        parts = [[int(n) for n in part] for part in layout_parts(ctx.adjacency.cpu().numpy(), recur_times)]
        assert sorted(n for part in parts for n in part) == list(range(ctx.num_nodes))
        lines = [f'{LAYOUT_ALGORITHM}/v{LAYOUT_VERSION}', f'recur_times={recur_times}']
        lines += [f'part{i}:' + ','.join(str(n) for n in part) for i, part in enumerate(parts)]
        membership = hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()
        return dataclasses.replace(ctx, layout_parts=tuple(tuple(part) for part in parts),
                                   layout_algorithm=LAYOUT_ALGORITHM, layout_sha256=membership, _cache={})
    return build


@contextmanager
def bfs_layout():
    from experiments import v4_runner
    from lib import runtime_context
    wrapped = bfs_context(runtime_context.build_runtime_context)
    with patch.object(runtime_context, 'build_runtime_context', wrapped), \
            patch.object(v4_runner, 'build_runtime_context', wrapped):
        yield wrapped
