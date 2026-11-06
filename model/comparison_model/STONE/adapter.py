"""STONE (KDD 2024): released model and graph editors, protocol adapter only.

``source/src`` holds byte copies of references/STONE-KDD-2024/src/{base/stone.py,
base/model.py, utils/graph_edite.py, utils/spatial_side_information.py,
utils/metrics.py}. The adapter:

* runs the released model on the full static node axis in canonical order
  (``node_num_ob = N``, ``node_num_un = 0``): the released observed/unobserved
  split only orders nodes and weights the two loss halves by node count, and the
  common loss mask already decides which positions are supervised;
* computes the spatial side information (Frechet embedding) exactly as the
  released ``Spatial_Embedding`` does -- Manhattan distances between adjacency
  rows, anchor sets drawn from the observed nodes -- with the common frozen graph
  (self-loops removed, as ``adj_mx - eye``) and the training-known nodes K as the
  observed set. No dynamic signal enters it;
* holds the two released ``Graph_Editer_Delete_Row`` modules (KK=2, 1% of nodes)
  as submodules so their parameters are optimised and checkpointed. Their mask
  ``M`` is a plain tensor attribute in the source and, as released, accumulates
  deleted columns across iterations; it is moved to the input device lazily.
  The editors act only in training; inference uses no edit and updates nothing.

Training returns every environment's prediction; the objective (mean + variance
of the absolute errors plus the editors' score-function terms) is assembled by
experiments/extension_baselines_20261103.py.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn

from lib.lcbt_v4 import assert_v4_strategy
from lib.v4_batch import bind_static_assets, read_common
from model.comparison_model.STOP.adapter import calendar_features
from model.comparison_model.source_loader import load_released

SOURCE = Path(__file__).resolve().parent / 'source'
# experiments/stone/main.py and src/utils/args.py; c=10 is the SD setting
# (716 nodes) in run.sh, the released configuration closest in size.
TBLOCKS_NUM, SBLOCKS_NUM = 5, 2
KT, KS_S, KS_T, DROPOUT = 3, 1, 3, .3
X_OUT, SEM_OUT, GATE_OUT, ADP_S, ADP_T = 128, 64, 128, 20, 20
KK, NUM_SAMPLE, EPSILON, ANCHOR_ROUNDS = 2, .01, .01, 10
SEM_NOISE_STD = 1.


def frechet_embedding(ssi_class, adjacency, known_nodes, seed):
    """Released Spatial_Embedding.SSI_func on (observed=K, unobserved=rest),
    returned in canonical node order."""
    adjacency = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(adjacency, 0.)
    num_nodes = adjacency.shape[0]
    observed = np.sort(np.asarray(known_nodes, dtype=np.int64))
    unobserved = np.setdiff1d(np.arange(num_nodes), observed)
    order = np.concatenate([observed, unobserved])
    sub = adjacency[order][:, order]
    dist = cdist(sub, sub, metric='cityblock')
    sem, _ = ssi_class([observed, unobserved], EPSILON, ANCHOR_ROUNDS, int(seed), 'cpu') \
        .spatial_emb_matrix(adj=None, dist=dist)
    canonical = torch.empty_like(sem)
    canonical[torch.as_tensor(order)] = sem
    return canonical.float()


class Model(nn.Module):
    def __init__(self, configs, runtime_context):
        super().__init__()
        if int(getattr(configs, 'raw_enc_in', 1)) != 1:
            raise ValueError('STONE adaptation reads one traffic channel')
        if int(configs.input_len) != 12:
            raise ValueError('the released STONE head is sized for a 12-step input (Linear(31, .))')
        self.raw_enc_in = 1
        self.num_nodes = runtime_context.num_nodes
        modules, self.source_import_side_effects = load_released(
            SOURCE, ['src.base.stone', 'src.utils.graph_edite', 'src.utils.spatial_side_information'])
        stone = modules['src.base.stone']
        editor = modules['src.utils.graph_edite'].Graph_Editer_Delete_Row
        ssi = modules['src.utils.spatial_side_information'].Spatial_Side_Information
        self.register_buffer('sem', frechet_embedding(
            ssi, runtime_context.adjacency.cpu().numpy(), runtime_context.known_nodes, configs.seed))
        sblocks = [[64, 16, 64] for _ in range(SBLOCKS_NUM)]
        tblocks = [[3, 128] if l == 0 else [128, 128] for l in range(TBLOCKS_NUM)]
        self.core = stone.STONE(input_dim=3, output_dim=1, node_num=self.num_nodes,
                                SBlocks=sblocks, TBlocks=tblocks, node_num_un=0,
                                node_num_ob=self.num_nodes, sem_dim=int(self.sem.shape[-1]),
                                has_shallow_encode=[l == 0 for l in range(SBLOCKS_NUM)],
                                Kt=KT, Ks_s=KS_S, Ks_t=KS_T, dropout=DROPOUT,
                                adp_s_dim=ADP_S, adp_t_dim=ADP_T, x_output_dim=X_OUT,
                                sem_output_dim=SEM_OUT, gate_output_dim=GATE_OUT,
                                horizon=int(configs.pred_len))
        # mask_s edits the semantic-graph adjacency, mask_t the series one.
        self.mask_s = editor(KK, self.num_nodes, NUM_SAMPLE, 'cpu')
        self.mask_t = editor(KK, self.num_nodes, NUM_SAMPLE, 'cpu')

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        assert_v4_strategy(configs, 'STONE')
        return bind_static_assets(cls(configs, runtime_context), configs, runtime_context, 'STONE')

    def editor_parameters(self):
        return list(self.mask_s.parameters()) + list(self.mask_t.parameters())

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        common = read_common(batch, runtime_context, self.static_asset_hash, 1, 'STONE',
                             self._v4_adjacency_cache, require_target=False)
        x = calendar_features(common['base_visible_input'], common['encoder_time_marks'])
        sem = torch.stack([self.sem] * x.shape[0], dim=0)
        if not self.training:
            pred, _, _ = self.core(x, sem)
            return {'pred': pred.squeeze(0).contiguous()}
        # Released train_batch: N(0, 1) noise on the side information, clamped at 0.
        sem = torch.clamp(sem + torch.normal(0., SEM_NOISE_STD, sem.shape).to(sem), min=0)
        for editor in (self.mask_s, self.mask_t):
            if editor.M.device != x.device:
                editor.M = editor.M.to(x.device)
        environments, _, log_p = self.core(x, sem, [self.mask_s, self.mask_t])
        return {'pred': environments[0].contiguous(), 'source_objective': 'stone_environments',
                'stone_environments': environments, 'stone_log_p': log_p,
                'source_raw_mae': True}
