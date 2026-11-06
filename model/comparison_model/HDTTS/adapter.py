"""HD-TTS (ICML 2024): released HDTTSModel, protocol adapter only.

``source/lib`` is a byte copy of references/hdtts/lib (SOURCE_MANIFEST.json).
The released package is literally named ``lib``, which collides with this
project's ``lib``; it is therefore imported inside ``source_namespace()``, which
swaps the ``lib`` entries of ``sys.modules`` for the duration of the import and
restores the project's afterwards. The source files are not edited.

What this adapter changes, and only this:

* input handling: the released ``run_realworld.py`` fills the whole masked
  series with ``ffill().bfill()`` before windowing, which reads values outside
  -- and after -- the input window. Here the same forward-then-backward fill is
  applied inside the current 12-step window only; a window without any visible
  reading keeps the common mean fill (normalized 0);
* the per-node ``NodeEmbedding`` of the temporal encoder is replaced by the
  shared ``StructuralNodeEncoder`` (same width, hidden_size // 2);
* connectivity is the common frozen graph, without self-loops
  (``include_self: False`` as in the released traffic configs);
* the k-MIS coarsened graphs, which the source caches at its first forward, are
  computed once on CPU at construction so every phase uses the same hierarchy.

Requires the released dependencies (torch-spatiotemporal 0.9.5, torch_scatter,
torch_sparse) on ``sys.path`` before torch_geometric is first imported.
"""
import contextlib
import hashlib
import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn

from layers.structural_node_encoder import StructuralNodeEncoder
from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common

SOURCE = Path(__file__).resolve().parent / 'source'
# config/model/hd_tts.yaml + hd_tts_imp.yaml (the released default model).
HPARAMS = dict(hidden_size=64, dilation=3, rnn_layers=4, cell='gru',
               mp_method=['diffconv', 'propconv'], mp_kernel_size=1,
               pooling_layers=3, mp_stage='both', recursive_lifting=True,
               fully_connected_readout=True, multi_step_scores=True,
               dropout=0., activation='elu')
# config/dataset/{la,bay}.yaml covariates: day (sin, cos) + weekday one-hot.
EXOG_SIZE = 2 + 7


SOURCE_SUBPACKAGES = ('nn', 'pooling', 'utils', 'datasets')


@contextlib.contextmanager
def source_namespace():
    """Import the released ``lib`` package without displacing this project's.

    The released subpackages (nn, pooling, utils, datasets) do not collide with
    any project ``lib`` module, so after the import they stay registered under
    their own names -- torch_geometric resolves annotations through
    ``sys.modules[cls.__module__]`` -- while the top-level ``lib`` entry and
    every project submodule are restored.
    """
    saved = {k: v for k, v in sys.modules.items() if k == 'lib' or k.startswith('lib.')}
    clash = [k for k, v in saved.items() if k.split('.')[1:2] and k.split('.')[1] in SOURCE_SUBPACKAGES
             and not str(getattr(v, '__file__', '') or '').startswith(str(SOURCE))]
    if clash:
        raise RuntimeError(f'project modules collide with released HD-TTS subpackages: {clash}')
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(SOURCE))
    try:
        yield
    finally:
        sys.path.remove(str(SOURCE))
        loaded = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == 'lib' or k.startswith('lib.')]}
        sys.modules.update(saved)
        for key, module in loaded.items():
            if key != 'lib':
                # PyG writes generated `<module>_<Class>_propagate` helpers next to them.
                origin = str(getattr(module, '__file__', '') or '')
                if not (origin.startswith(str(SOURCE)) or key.endswith('_propagate')):
                    raise RuntimeError(f'{key} did not load from the released source copy')
                sys.modules[key] = module


def _generic_propagate(self, edge_index, size=None, **kwargs):
    from torch_geometric.nn import MessagePassing
    return MessagePassing.propagate(self, edge_index, size=size, **kwargs)


def _use_generic_propagate(modules):
    """PyG >= 2.5 compiles ``propagate`` from the subclass's own source; the
    released subclasses (written for PyG 2.4, which had no such compilation)
    inherit their ``self.propagate(...)`` call from tsl, so the compiled
    signature loses ``x``/``weight``. Pointing them at the generic
    ``MessagePassing.propagate`` restores the PyG 2.4 code path; the computed
    messages are the same."""
    from torch_geometric.nn import MessagePassing
    patched = []
    for module in modules:
        for value in vars(module).values():
            if (isinstance(value, type) and issubclass(value, MessagePassing)
                    and value.__module__ == module.__name__ and 'propagate' not in value.__dict__):
                value.propagate = _generic_propagate
                patched.append(f'{value.__module__}.{value.__name__}')
    return patched


def load_source_model_class():
    existing = sys.modules.get('lib.nn.models.hdtts_model')
    if existing is not None and str(existing.__file__).startswith(str(SOURCE)):
        return existing.HDTTSModel
    with source_namespace():
        module = importlib.import_module('lib.nn.models.hdtts_model')
    import torch_geometric.typing as pyg_typing
    if not (pyg_typing.WITH_TORCH_SPARSE and pyg_typing.WITH_TORCH_SCATTER):
        raise RuntimeError('HD-TTS needs torch_sparse/torch_scatter visible before torch_geometric import')
    layers = [sys.modules[k] for k in sorted(sys.modules) if k.startswith('lib.nn.layers.')]
    global PROPAGATE_COMPATIBILITY
    PROPAGATE_COMPATIBILITY = _use_generic_propagate(layers)
    return module.HDTTSModel


PROPAGATE_COMPATIBILITY = None


def window_ffill_bfill(x, mask):
    """pandas ``ffill().bfill()`` restricted to each [B, L, N] input window."""
    if x.ndim != 4 or mask.shape != x.shape[:3]:
        raise ValueError('window fill needs x [B,L,N,C] and mask [B,L,N]')
    mask = mask.bool()
    length = x.shape[1]
    index = torch.arange(length, device=x.device).view(1, length, 1).expand_as(mask)
    left = torch.where(mask, index, torch.full_like(index, -1)).cummax(1).values
    right = torch.where(mask, index, torch.full_like(index, length)).flip(1).cummin(1).values.flip(1)
    source = torch.where(left >= 0, left, right)
    found = source < length
    safe = torch.where(mask[..., None], x, torch.zeros_like(x))
    filled = safe.gather(1, source.clamp(0, length - 1)[..., None].expand_as(x))
    return torch.where(found[..., None], filled, torch.zeros_like(x))


def calendar_covariates(marks, num_nodes):
    """``datetime_encoded('day')`` then ``datetime_onehot('weekday')``."""
    minute, hour, weekday = marks[..., 0], marks[..., 1], marks[..., 2]
    if not ((weekday >= 0) & (weekday < 7)).all():
        raise ValueError('weekday marks must be 0..6')
    phase = 2 * math.pi * (hour * 60. + minute) / 1440.
    day = torch.stack([torch.sin(phase), torch.cos(phase)], -1)
    week = torch.nn.functional.one_hot(weekday.long(), 7).to(day.dtype)
    u = torch.cat([day, week], -1)
    return u.unsqueeze(2).expand(-1, -1, num_nodes, -1)


def _move_select_output(select, device):
    from torch_sparse import SparseTensor
    moved = select.__class__.__new__(select.__class__)
    for key, value in vars(select).items():
        moved.__dict__[key] = value.to(device) if isinstance(value, (torch.Tensor, SparseTensor)) else value
    return moved


def hierarchy_fingerprint(graphs):
    digest = hashlib.sha256()
    for select in graphs:
        digest.update(select.cluster_index.detach().cpu().numpy().tobytes())
        digest.update(str((select.num_nodes, select.num_clusters)).encode())
    return digest.hexdigest()


class Model(nn.Module):
    def __init__(self, configs, runtime_context):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('HD-TTS adaptation reads one traffic channel')
        from torch_sparse import SparseTensor
        self.raw_enc_in = 1
        self.num_nodes = runtime_context.num_nodes
        self.input_len, self.pred_len = int(configs.input_len), int(configs.pred_len)
        HDTTSModel = load_source_model_class()
        self.propagate_compatibility = list(PROPAGATE_COMPATIBILITY or ())
        self.core = HDTTSModel(input_size=1, n_nodes=self.num_nodes, horizon=self.pred_len,
                               exog_size=EXOG_SIZE, mask_size=1, **HPARAMS)
        emb_size = self.core.encoder_time.emb_size
        self.core.encoder_time.embeddings = StructuralNodeEncoder(
            runtime_context.adjacency, runtime_context.positions, out_dim=emb_size)
        adjacency = runtime_context.adjacency.detach().cpu().float().clone()
        adjacency.fill_diagonal_(0.)
        self._adj_cpu = SparseTensor.from_dense(adjacency)
        self._graphs_cpu = self.core.spatial_conv.compute_coarsened_graphs(
            self._adj_cpu, None, num_nodes=self.num_nodes)
        self.hierarchy_sha256 = hierarchy_fingerprint(self._graphs_cpu)
        self.hierarchy_sizes = [int(s.num_clusters) for s in self._graphs_cpu]
        self._device_graph = None

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'HDTTS')
        return bind_static_assets(cls(configs, runtime_context), configs, runtime_context, 'HDTTS')

    def _graph_on(self, device):
        if self._device_graph is None or self._device_graph[0] != str(device):
            graphs = [_move_select_output(s, device) for s in self._graphs_cpu]
            self.core.spatial_conv._graphs = graphs
            self._device_graph = (str(device), self._adj_cpu.to(device))
            for module in self.core.modules():
                # tsl DiffConv caches its normalized supports at first use.
                if hasattr(module, '_support'):
                    module._support = None
        return self._device_graph[1]

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        common = read_common(batch, runtime_context, self.static_asset_hash, 1, 'HDTTS',
                             self._v4_adjacency_cache, require_target=False)
        base = common['base_visible_input']
        mask = batch['input_available_mask'].bool()
        x = window_ffill_bfill(base, mask)
        u = calendar_covariates(common['encoder_time_marks'], self.num_nodes)
        adj = self._graph_on(base.device)
        prediction = self.core.predict(x, adj, None, input_mask=mask.unsqueeze(-1), u=u)
        # Released objective: MaskedMAE after inverse scaling (scale_target False).
        return {'pred': prediction.contiguous(), 'source_raw_mae': True}
