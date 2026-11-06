"""MCPST-FSL (arXiv 2602.01936, 2026): released MCPST_FSL network, protocol adapter only.

``source/models/model.py`` and ``source/datasets.py`` are byte copies of
references/MCPST (SOURCE_MANIFEST.json). Only the source-domain training stage is
run: the released few-shot FOMAML meta-learning and target-domain fine-tuning
update parameters on target-city data, which the fixed-parameter inductive
protocol does not have (inference never updates parameters).

Input handling, as in ``datasets.TrafficDataset.load_dataset`` but inside the
current window only:

* channels ``[value, time-of-day]`` (two original channels: the released LFS
  payload holds 16 bytes per (time, node), i.e. two float64 channels; the
  second is the DCRNN time-of-day channel). The released code normalises every
  channel with one global scalar mean/std and clips to [-5, 5]; here the common
  train-K scalar scaler replaces the whole-data statistics, applied to both
  channels, then the same clip;
* the four physics features come from the released
  ``enhanced_physics_features`` applied to the 12-step window, never to the whole
  series: the released whole-series call reads values outside the window and,
  through ``np.gradient``'s central difference, the first future step;
* the adjacency is the common frozen graph, normalised by the released
  ``robust_normalize_adjacency``.

The spectral module decomposes the normalised Laplacian once per batch element
and time step; with a fixed graph the copies are the same matrix. On CPU they are
bit-identical; on GPU the batched products differ by float32 rounding (measured
1.2e-7 on PEMS-BAY, which moves the released per-copy eigenvalues by up to 3.4e-5).
The decomposition of the first copy is therefore computed once and reused when
every copy agrees within ``SPECTRAL_TOLERANCE``; otherwise the released loop runs.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common

SOURCE = Path(__file__).resolve().parent / 'source'
# config.yaml (active, uncommented section) model/physics settings.
MODEL = dict(name='MCPST_FSL', input_dim=6, hidden_dim=64, pred_len=12, spatial_dim=2,
             num_diffusion_steps=6, num_oscillator_steps=10, num_eigen_vectors=8, dropout=0.1,
             node_minibatch_cpu=4096, node_minibatch_gpu=1024,
             physics=dict(alpha_flow=2.0, alpha_spatial=0.08, alpha_temporal=0.08, alpha_physics=0.03,
                          alpha_consistency=0.015, alpha_uncertainty=0.01, alpha_coupling=0.05))
CLIP = 5.
SPECTRAL_TOLERANCE = 1e-5
STEPS_PER_DAY = 288


def _load(name, relative):
    key = f'mcpst_released_{name}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, SOURCE / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def released():
    return _load('model', 'models/model.py'), _load('datasets', 'datasets.py')


class CachedSpectralDecomposition:
    """Same eigendecomposition, computed once for the fixed graph."""

    def __init__(self, module):
        self.module = module
        self.original = module.spectral_decomposition
        self.cache = None

    @staticmethod
    def _close(a, b):
        return a.shape[-2:] == b.shape[-2:] and a.device == b.device and \
            float((a - b).abs().amax()) <= SPECTRAL_TOLERANCE

    def __call__(self, laplacian):
        first = laplacian[:1]
        if not self._close(laplacian, first):
            return self.original(laplacian)
        if self.cache is None or not self._close(self.cache[0], first):
            values, vectors = self.original(first)
            self.cache = (first.detach().clone(), values.detach(), vectors.detach())
        _, values, vectors = self.cache
        b = laplacian.shape[0]
        return values.expand(b, -1), vectors.expand(b, -1, -1)


class Model(nn.Module):
    def __init__(self, configs, runtime_context):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('MCPST adaptation reads one traffic channel')
        self.raw_enc_in = 1
        self.num_nodes = runtime_context.num_nodes
        self.pred_len = int(configs.pred_len)
        model_module, data_module = released()
        self._physics_features = data_module.enhanced_physics_features
        config = {'model': dict(MODEL, pred_len=self.pred_len),
                  'data': {'v4': {'node_num': self.num_nodes}}, 'training': {'test_dataset': 'v4'}}
        self.core = model_module.MCPST_FSL(config)
        self.core.spectral_module.spectral_decomposition = CachedSpectralDecomposition(self.core.spectral_module)
        raw = runtime_context.adjacency.detach().cpu().numpy().astype(np.float32)
        self._raw_adjacency = raw
        self.register_buffer('adjacency', torch.from_numpy(data_module.robust_normalize_adjacency(raw)))
        self.register_buffer('scaler_mean_std', torch.zeros(2, dtype=torch.float64))

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'MCPST')
        return bind_static_assets(cls(configs, runtime_context), configs, runtime_context, 'MCPST')

    def set_scaler(self, scaler):
        """The common train-K scalar scaler (also used for the time channel)."""
        self.scaler_mean_std.copy_(torch.tensor([float(scaler['mean']), float(scaler['std'])], dtype=torch.float64))

    def features(self, base, marks):
        mean, std = (float(v) for v in self.scaler_mean_std)
        if std <= 0:
            raise RuntimeError('MCPST requires the common scaler (set_scaler) before use')
        minute, hour = marks[..., 0].double(), marks[..., 1].double()
        tod = (hour * (STEPS_PER_DAY // 24) + torch.floor(minute / (1440 // STEPS_PER_DAY))) / STEPS_PER_DAY
        tod = ((tod - mean) / std).to(base.dtype).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, base.shape[2], -1)
        window = torch.clamp(torch.cat([base, tod], dim=-1), -CLIP, CLIP).detach().cpu().numpy().astype(np.float32)
        out = np.stack([self._physics_features(w, self._raw_adjacency) for w in window])
        return torch.from_numpy(out).to(base.device)

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        common = read_common(batch, runtime_context, self.static_asset_hash, 1, 'MCPST',
                             self._v4_adjacency_cache, require_target=False)
        x = self.features(common['base_visible_input'], common['encoder_time_marks'])
        outputs = self.core(x, self.adjacency)
        pred = outputs['flow_predictions'].permute(0, 2, 1).unsqueeze(-1).contiguous()
        return {'pred': pred, 'source_objective': 'mcpst_physics_informed',
                'mcpst_uncertainty': outputs['uncertainty'].permute(0, 2, 1).unsqueeze(-1),
                'mcpst_order_parameter': outputs['order_parameter'],
                'mcpst_spectral_gap': outputs['spectral_gap'],
                'mcpst_physics_weights': outputs['physics_weights']}
