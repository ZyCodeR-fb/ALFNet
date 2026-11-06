"""MetaDG (AAAI 2026): released MetaDG network, protocol adapter only.

``source/model/MetaDG.py`` and ``MetaDG.yaml`` are byte copies of references/MetaDG
(SOURCE_MANIFEST.json). The adapter:

* builds the released ``x = [value, tod, dow]`` and decoder covariates
  ``y_cov = [tod, dow]`` from the common masked input and calendar marks;
* keeps the released curriculum teacher forcing (inverse-sigmoid schedule over
  ``batches_seen``, one ``np.random.uniform`` draw per decoder step), but the
  label fed back at a forced step is the common legal target only where
  ``decoder_available_mask`` allows it and the model's own previous output
  elsewhere -- the released code feeds the full label tensor;
* replaces the static node-identity table ``node_emb`` by the shared
  ``StructuralNodeEncoder`` (``NODE_IDENTITY='structural'``) or keeps the
  released table (``'released'``), selected per run;
* creates ``MetaDGCRU``'s initial hidden state on the model device (the
  released ``init_hidden_state`` builds it on CPU while every other tensor is on
  ``self.device``, which fails on GPU; the values are the same zeros).

Hyperparameters: the released per-dataset block for PEMS04/PEMS08; METR-LA uses
the PEMS08 block and PEMS-BAY the PEMS04 block (closest node counts; the release
has no speed block), with ``node_num`` set to the dataset.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn.utils import parametrize

from layers.structural_node_encoder import StructuralNodeEncoder
from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common

SOURCE = Path(__file__).resolve().parent / 'source'
BLOCK = {'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08', 'METRLA': 'PEMS08', 'PEMSBAY': 'PEMS04'}
STEPS_PER_DAY = 288


def released():
    key = 'metadg_released_model'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, SOURCE / 'model/MetaDG.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def released_config(dataset):
    blocks = yaml.safe_load((SOURCE / 'model/MetaDG.yaml').read_text())
    return blocks[BLOCK[dataset]]


class StructuralNodeTable(nn.Module):
    def __init__(self, context, width):
        super().__init__()
        self.encoder = StructuralNodeEncoder(context.adjacency, context.positions, out_dim=width)

    def forward(self, original):
        return self.encoder()


class LegalTeacher:
    """``labels[:, t, ...]`` for the released curriculum: legal target or own output."""

    def __init__(self, target, available):
        self.target, self.available, self.last = target, available.bool(), None

    def remember(self, module, args, output):
        self.last = output

    def __getitem__(self, key):
        _, t, _ = key
        return torch.where(self.available[:, t, :, None], self.target[:, t], self.last)


def calendar(marks):
    minute, hour, dow = marks[..., 0].double(), marks[..., 1].double(), marks[..., 2].double()
    if not ((dow >= 0) & (dow < 7)).all():
        raise ValueError('weekday marks must be 0..6')
    tod = (hour * (STEPS_PER_DAY // 24) + torch.floor(minute / (1440 // STEPS_PER_DAY))) / STEPS_PER_DAY
    return tod, dow


class Model(nn.Module):
    decoder_training_policy_name = 'source_curriculum_legal_target_teacher_input'

    def __init__(self, configs, runtime_context, node_identity='structural'):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('MetaDG adaptation reads one traffic channel')
        if node_identity not in ('structural', 'released'):
            raise ValueError(node_identity)
        self.raw_enc_in = 1
        self.num_nodes = runtime_context.num_nodes
        self.node_identity = node_identity
        module = released()
        block = released_config(str(configs.dloader_name))
        args = dict(block['model_args'], node_num=self.num_nodes, out_steps=int(configs.pred_len))
        self.released_block = BLOCK[str(configs.dloader_name)]
        self.released_hyperparameters = {k: block[k] for k in ('lr', 'eps', 'weight_decay', 'clip_grad', 'batch_size')}
        self.core = module.MetaDG(**args, device=torch.device(configs.device))
        if node_identity == 'structural':
            self.core.node_emb.requires_grad_(False)
            parametrize.register_parametrization(
                self.core, 'node_emb', StructuralNodeTable(runtime_context, args['node_emb_dim']))
        for cell in list(self.core.encoder.cells) + list(self.core.decoder.cells):
            cell.init_hidden_state = self._hidden_state_factory(cell)
        self.register_buffer('batches_seen', torch.zeros((), dtype=torch.long))

    @staticmethod
    def _hidden_state_factory(cell):
        def init_hidden_state(batch_size):
            device = next(cell.parameters()).device
            return torch.zeros(batch_size, cell.node_num, cell.hidden_dim, device=device)
        return init_hidden_state

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'MetaDG')
        identity = (getattr(configs, 'model_family_cfg', {}) or {}).get('model', {}).get('node_identity', 'structural')
        return bind_static_assets(cls(configs, runtime_context, identity), configs, runtime_context, 'MetaDG')

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        common = read_common(batch, runtime_context, self.static_asset_hash, 1, 'MetaDG',
                             self._v4_adjacency_cache, require_target=self.training)
        base = common['base_visible_input']
        n = base.shape[2]
        tod, dow = calendar(common['encoder_time_marks'])
        ftod, fdow = calendar(common['decoder_time_marks'])
        expand = (lambda v: v.to(base.dtype).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, -1))
        x = torch.cat([base, expand(tod), expand(dow)], dim=-1)
        y_cov = torch.cat([expand(ftod), expand(fdow)], dim=-1)
        if not self.training:
            return {'pred': self.core(x, y_cov).contiguous()}
        teacher = LegalTeacher(batch['target'][..., :1], batch['decoder_available_mask'])
        handle = self.core.fc_final.register_forward_hook(teacher.remember)
        try:
            out = self.core(x, y_cov, teacher, int(self.batches_seen))
        finally:
            handle.remove()
        self.batches_seen += 1
        return {'pred': out.contiguous(), 'source_objective': 'metadg_released_loss',
                'metadg_loss': 'huber' if str(runtime_context.target) == 'q' else 'mae'}
