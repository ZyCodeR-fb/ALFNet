import numpy as np
import torch
import torch.nn as nn

from model.comparison_model.family_utils import build_patchstg_cfg
from .patchstg import PatchSTG as PatchSTGBackbone


# Availability one-hot (4) + state_age (1) + observation mask (1). Every one of
# these is known at prediction time, so feeding them leaks nothing.
LIFECYCLE_CHANNELS = 6
CALENDAR_CHANNELS = 2


def _augment_align(dist_matrix, auglen):
    sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(int(idx))
        if len(augidx) == auglen:
            break
    return np.asarray(augidx, dtype=int)


def _reorder_data(parts_idx, adj, patch_size):
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if patch_size - part_idx.shape[0] > 0:
            local_part_idx = _augment_align(
                part_dist, patch_size - part_idx.shape[0]
            )
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([
            reo_parts_idx,
            np.arange(part_idx.shape[0]) + patch_size * i,
        ])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])
    return ori_parts_idx, reo_parts_idx, reo_all_idx


class Model(nn.Module):

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        """The v4 entry point. Explicit, and it resolves nothing from a path.

        This takes every graph fact from the context: node count and order, the
        adjacency, coordinates or their absence, and the already-resolved layout
        membership. In particular it does not decide for itself whether to build
        a KD-tree or a graph-native layout -- that decision lives in
        `resolve_layout` and is recorded in the manifest, and a second builder
        here is how the two silently disagreed before.
        """
        if runtime_context is None:
            raise ValueError(
                "ALFTemporal.from_runtime_context requires a RuntimeDataContext; "
                "v4 fails closed rather than falling back to data_path")
        from lib.context_binding import assert_config_agrees_with_context

        # The nested expert's own `input_features` sizes `expert_input_dim`, so
        # a nested YAML naming the wrong quantity is not merely documentation.
        assert_config_agrees_with_context(configs, runtime_context, "ALFTemporal")
        return cls(configs, runtime_context=runtime_context)

    def __init__(self, configs, runtime_context=None):
        super().__init__()
        if runtime_context is None:
            raise ValueError("construct ALFTemporal through from_runtime_context")
        self.runtime_context = runtime_context
        self.static_asset_hash = runtime_context.static_asset_hash()
        self.cfg = build_patchstg_cfg(configs, runtime_context=runtime_context)
        model_cfg = self.cfg["model"]
        self.full_num_nodes = int(self.cfg["node_num"])
        self.known_nodes = list(self.cfg["known_nodes"])
        self.output_dim = configs.c_out
        # Ablation switch: with lifecycle_input off the expert sees only the raw
        # series, which is the pre-fix behaviour.
        self.lifecycle_input = bool(model_cfg.get("lifecycle_input", True))
        # "features": the branch stops decoding its own forecast and emits one
        # condition vector per node instead. "prediction" is the original
        # two-expert behaviour, kept for the fallback variant.
        self.emit = str(model_cfg.get("emit", "features"))
        if self.emit != "features":
            raise ValueError("the final ALFTemporal branch emits features only")
        retired = [name for name in ("use_timefilter", "mixer", "spatial_tokenizer")
                   if name in model_cfg and model_cfg[name] not in (False, "crossformer", "kdtree")]
        if retired:
            raise ValueError(f"retired ALFTemporal options are not supported: {retired}")
        self.local_context_only = bool(model_cfg.get("local_context_only", False))
        self.expert_input_dim = int(configs.raw_enc_in) + (
            LIFECYCLE_CHANNELS if self.lifecycle_input else 0
        )

        full_adj = runtime_context.adjacency.cpu().numpy()
        positions = runtime_context.positions
        self._assert_context_assets(runtime_context, full_adj, model_cfg)
        parts_idx = [np.asarray(part, dtype=np.int64)
                     for part in runtime_context.layout_parts]
        ori_parts_idx, reo_parts_idx, reo_all_idx = _reorder_data(
            parts_idx, full_adj, int(model_cfg["spa_patchsize"]))
        ori_parts_idx = ori_parts_idx.tolist()
        reo_parts_idx = reo_parts_idx.tolist()
        reo_all_idx = reo_all_idx.tolist()
        spa_patchnum = len(parts_idx)
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx = reo_all_idx
        # `_reorder_data` only *pads* a KD-tree part up to spa_patchsize; it
        # never splits one. So a patch size smaller than the largest part
        # silently produces the wrong token count, and the model builds fine and
        # then fails deep inside WindowAttBlock's reshape. Porting to a second
        # dataset is exactly when that happens -- PEMS-BAY's 325 nodes give
        # parts of 10-11 where METR-LA's 207 give 6-7 -- so check it here, where
        # the message can name the value to put in the YAML.
        expected_tokens = int(spa_patchnum) * int(model_cfg["spa_patchsize"])
        if len(reo_all_idx) != expected_tokens:
            largest = max(len(part) for part in parts_idx)
            raise ValueError(
                f"spa_patchsize={model_cfg['spa_patchsize']} cannot hold this "
                f"dataset's KD-tree partition: {self.full_num_nodes} nodes split "
                f"into {spa_patchnum} parts of up to {largest}. Set "
                f"spa_patchsize >= {largest} (the final configs use the exact "
                f"largest resolved part; only smaller parts are padded)."
            )

        tem_patchsize = int(model_cfg["tem_patchsize"])
        tem_patchnum = int(model_cfg["tem_patchnum"])
        if tem_patchsize * tem_patchnum != configs.pred_len:
            raise ValueError(
                "PatchSTG requires tem_patchsize * tem_patchnum == pred_len "
                "under the current adapter"
            )
        self.backbone = PatchSTGBackbone(
            tem_patchsize=tem_patchsize,
            tem_patchnum=tem_patchnum,
            node_num=self.full_num_nodes,
            spa_patchsize=int(model_cfg["spa_patchsize"]),
            spa_patchnum=spa_patchnum,
            tod=int(model_cfg["tod"]),
            dow=int(model_cfg["dow"]),
            layers=int(model_cfg["layers"]),
            factors=int(model_cfg["factors"]),
            input_dims=int(model_cfg["input_dims"]),
            node_dims=int(model_cfg["node_dims"]),
            tod_dims=int(model_cfg["tod_dims"]),
            dow_dims=int(model_cfg["dow_dims"]),
            ori_parts_idx=self.ori_parts_idx,
            reo_parts_idx=self.reo_parts_idx,
            reo_all_idx=self.reo_all_idx,
            adjacency=torch.as_tensor(full_adj, dtype=torch.float32),
            positions=positions,
            input_channels=self.expert_input_dim + CALENDAR_CHANNELS,
            crossformer_router_factor=int(
                model_cfg.get("crossformer_router_factor", 4)
            ),
            crossformer_dropout=float(model_cfg.get("crossformer_dropout", 0.1)),
        )
        self.backbone.local_context_only = self.local_context_only
        self.feature_dim = self.backbone.feature_dim

    @staticmethod
    def _assert_context_assets(runtime_context, adjacency, model_cfg):
        """Fail closed on a context whose assets are not the declared ones."""
        identity = runtime_context.static_asset_identity()
        nodes = identity["full_num_nodes"]
        if adjacency.shape != (nodes, nodes):
            raise ValueError(
                f"context adjacency is {adjacency.shape}, expected ({nodes}, {nodes})")
        flat = [int(node) for part in runtime_context.layout_parts for node in part]
        if sorted(flat) != list(range(nodes)):
            raise ValueError(
                "context layout does not cover every node exactly once; "
                "node order or membership has been tampered with")
        if identity["layout_algorithm"] not in (
                "graph_native_component_bfs", "kdtree_on_verified_wgs84"):
            raise ValueError(f"unknown layout algorithm {identity['layout_algorithm']}")

    def _build_temporal_features(self, x_enc, x_t_mark):
        if x_t_mark is None or x_t_mark.shape[-1] < 3:
            raise ValueError(
                "PatchSTG requires x_t_mark with minute/hour/day-of-week features"
            )
        minute = x_t_mark[..., 0].to(x_enc.device, dtype=x_enc.dtype)
        hour = x_t_mark[..., 1].to(x_enc.device, dtype=x_enc.dtype)
        dow = x_t_mark[..., 2].to(x_enc.device, dtype=x_enc.dtype)
        tod = hour * 12.0 + torch.floor(minute / 5.0)
        return torch.stack([tod, dow], dim=-1).unsqueeze(2).expand(
            -1, -1, self.full_num_nodes, -1
        )

    def forward_v4(self, x_enc, x_t_mark):
        """The v4 path: full N in, full N out, and no scatter anywhere.

        The retired K-to-N scatter is absent: the full static node axis is the
        only accepted representation.

        A node axis that is not the context's N is refused rather than padded.
        Zero-padding a short axis is exactly the defect being removed, and it
        would be invisible: the shapes downstream would all be right.
        """
        if x_enc.dim() != 4:
            raise ValueError(
                f"ALFTemporal v4 expects [batch, time, node, channel], got "
                f"{tuple(x_enc.shape)}")
        if x_enc.shape[2] != self.full_num_nodes:
            raise ValueError(
                f"ALFTemporal v4 received {x_enc.shape[2]} nodes but the context "
                f"has {self.full_num_nodes}; the full static node axis is the "
                "only accepted input and a short axis is never padded")
        if x_enc.shape[-1] < self.expert_input_dim:
            raise ValueError(
                f"ALFTemporal expects at least {self.expert_input_dim} input "
                f"channels (raw + lifecycle context), got {x_enc.shape[-1]}")
        full_x = x_enc[..., :self.expert_input_dim]
        full_te = self._build_temporal_features(full_x, x_t_mark)
        out = self.backbone(full_x, full_te).contiguous()
        return {"features": out} if self.emit == "features" else {"pred": out}
