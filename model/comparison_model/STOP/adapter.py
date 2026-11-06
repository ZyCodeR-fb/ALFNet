"""STOP (ICML 2025): released MLP backbone + Core_Adaptive, protocol adapter only.

``source/src`` holds byte copies of references/STOP/TrafficStream/src/{models/
stop.py, base/model.py}; TrafficStream (PEMS district 3, 5-minute data) is the
released configuration closest to the four benchmark datasets, so its
``tod_size=288`` is used. The released code has no node-identity parameter and
no graph input; the adapter only builds the released ``[value, tod, dow]``
input from the common batch. Inference never updates parameters.

The released training loop (stop_engine.py) contains no message perturbation or
distributionally robust objective -- those code paths are commented out and the
loss is the masked MAE of the corrected prediction -- so none is added here.
"""
from pathlib import Path

import torch
from torch import nn

from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common
from model.comparison_model.source_loader import load_released

SOURCE = Path(__file__).resolve().parent / 'source'
# TrafficStream/experiments/stop/main.py and src/utils/args.py defaults.
MLP_ARGS = dict(input_dim=3, output_dim=1, num_layer=3, model_dim=64, prompt_dim=32,
                tod_size=288, kernel_size=3)
STOP_ARGS = dict(extra_type=1, same=0, hid_dim=256)
CORE, HEAD = 8, 8
STEPS_PER_DAY = 288


def calendar_features(raw, marks):
    """LargeST-style channels: value, time-of-day fraction, day-of-week / 7."""
    if raw.ndim != 4 or raw.shape[-1] != 1 or marks.shape[:2] != raw.shape[:2]:
        raise ValueError('STOP requires [B,L,N,1] traffic and matching calendar marks')
    minute, hour, dow = (marks[..., i].double() for i in range(3))
    if not ((dow >= 0) & (dow < 7)).all():
        raise ValueError('weekday marks must be 0..6')
    step = hour * (STEPS_PER_DAY // 24) + torch.floor(minute / (1440 // STEPS_PER_DAY))
    calendar = torch.stack([step / STEPS_PER_DAY, dow / 7.], -1).to(raw.dtype)
    return torch.cat([raw, calendar.unsqueeze(2).expand(-1, -1, raw.shape[2], -1)], -1)


class Model(nn.Module):
    def __init__(self, configs, runtime_context):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('STOP adaptation reads one traffic channel')
        if int(configs.input_len) != 12:
            raise ValueError('the released MLP is built for a 12-step input')
        self.raw_enc_in = 1
        modules, self.source_import_side_effects = load_released(SOURCE, ['src.models.stop'])
        stop = modules['src.models.stop']
        base = stop.MLP(**MLP_ARGS)
        model_dim = MLP_ARGS['model_dim'] + 2 * MLP_ARGS['prompt_dim']
        self.core = stop.STOP(input_dim=MLP_ARGS['input_dim'], output_dim=MLP_ARGS['output_dim'],
                              model_args=dict(STOP_ARGS, horizon=int(configs.pred_len)),
                              stmodel=base, dim=[model_dim, model_dim], core=CORE, head=HEAD)

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'STOP')
        return bind_static_assets(cls(configs, runtime_context), configs, runtime_context, 'STOP')

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        common = read_common(batch, runtime_context, self.static_asset_hash, 1, 'STOP',
                             self._v4_adjacency_cache, require_target=False)
        x = calendar_features(common['base_visible_input'], common['encoder_time_marks'])
        prediction = self.core(x)
        # Released objective: masked MAE after inverse scaling.
        return {'pred': prediction.contiguous(), 'source_raw_mae': True}
