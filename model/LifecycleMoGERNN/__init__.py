import importlib
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from args import load_args_from_config
from lib.utils import load_project_adjacency, resolve_known_unknown_nodes
from model.STBP.stbp import STBP_Model
from utils.model_output import unpack_model_output


epsilon = 1e-8


class SpatialConvOrderK(nn.Module):
    """
    Spatial convolution of order K with possibly different diffusion matrices (useful for directed graphs)

    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, support_len=3, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order
        self.c_out = c_out

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        support = [adj_fwd, adj_bwd]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support, support_diag=None, pattern=None):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)

        if support_diag is not None:
            out_diag = [x] if self.include_self else []
            if (type(support_diag) is not list):
                support_diag = [support_diag]
            for a in support_diag:
                x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
                out_diag.append(x1)
                for k in range(2, self.order + 1):
                    x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                    out_diag.append(x2)
                    x1 = x2
            out_diag = torch.cat(out_diag, dim=1)

        # out => b t*d n s, w/o self-loop and temporal
        # out_diag => b t*d n s, w/ self-loop and temporal
        # suppose t=0,1 (current),2

        if pattern is not None:
            t = out.size(1) // self.c_out
            mid = t // 2

            if pattern == "wo_self_loop_w_temporal":
                # if w/o self-loop, but w/ temporal
                out[:, :mid, :, :] = out_diag[:, :mid, :, :]
                out[:, mid+1:, :, :] = out_diag[:, mid+1:, :, :]
            elif pattern == "w_self_loop_wo_temporal":
                # if w/ self-loop, but w/o temporal
                out[:, mid, :, :] = out_diag[:, mid, :, :]
            elif pattern == "wo_self_loop_wo_temporal":
                out = out_diag

        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out


class KITS(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 args
                 ):
        super(KITS, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dataset_name = args.dataset_name

        self.t_dim = 3
        self.register_buffer('adj', torch.tensor(adj).float(), persistent=False)
        self.fc_1 = nn.Linear(1, d_hidden)

        self.gcn_1 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_2 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_3 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)

        self.smooth = nn.Linear(2 * d_hidden, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, 1)

        self.relu = nn.ReLU(inplace=True)
        self.supp = None
        self.adj_aug = None
        self.obs_neighbors = None

        if args.use_adj_drop:
            print("use adj dropout...")
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Identity()

        if args.use_init:
            print("use init...")
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)

    def adj_drop(self, supp, mask):
        # supp: list, fwd and bwd adj - (n, n)
        # mask: b, s, n, 1
        mask = rearrange(mask, 'b s n 1 -> (b s) n')
        mask = mask.sum(0)  # n
        obs_index = mask > 0  # n
        # unobs_index = mask == 0  # n
        supp_update = []
        for i in range(len(supp)):
            s = supp[i].clone().detach()

            s_hor = s[obs_index, :]  # n1, n
            s_ver = s[:, obs_index]  # n, n1

            s_hor = self.dropout(s_hor)
            s_ver = self.dropout(s_ver)

            s[obs_index, :] = s_hor
            s[:, obs_index] = s_ver

            supp_update.append(s)
        return supp_update

    def forward(self, x, mask=None, known_set=None, sub_entry_num=None, reset=False):
        adj = self.adj.clone()  # adjacency matrix
        mask_bool = mask.bool()
        mask = mask.to(dtype=x.dtype)

        if self.training:
            if reset and known_set is not None and sub_entry_num is not None:
                # ========================================
                # Obtain 1-hop neighbors of each observed entry
                # ========================================
                # preserve observed entries
                adj = adj[known_set, :]
                adj = adj[:, known_set]  # n1, n1
                n1 = adj.size(0)

                # get the 1-hop neighbors of each observed entry.
                if self.obs_neighbors is None:
                    obs_neighbors = {}
                    for i in range(n1):
                        row_nonzero = set(torch.where(adj[i, :] > 0)[0].detach().cpu().numpy().tolist())
                        col_nonzero = set(torch.where(adj[:, i] > 0)[0].detach().cpu().numpy().tolist())
                        nonzero = row_nonzero.union(col_nonzero)
                        obs_neighbors[i] = list(nonzero)  # 1-hop neighbors
                    self.obs_neighbors = obs_neighbors
                else:
                    obs_neighbors = self.obs_neighbors  # n1, n1, note that cannot use copy!!!

                # ========================================
                # Create dynamic adjacency matrix
                # ========================================
                # initialize dynamic adjacency matrix
                n2 = n1 + sub_entry_num
                adj_aug = torch.rand(n2, n2).to(adj.device)  # n2, n2

                # preserve original observed parts in newly-created adj
                adj_aug[:n1, :n1] = adj

                # remove self-loop
                adj_diag = 1. - torch.eye(n2).to(adj.device)  # n2, n2
                adj_aug = adj_aug * adj_diag  # n2, n2

                # for each newly-created virtual entry, randomly connect it to one observed entry
                neighbors = {key: value.copy() for key, value in obs_neighbors.items()}  # initially has n1 entries' 1-hop neighbors
                adj_aug_mask = torch.zeros_like(adj_aug)  # n2, n2
                adj_aug_mask[:n1, :n1] = 1
                for i in range(n1, n2):
                    n_current = range(len(neighbors.keys()))  # number of current entries (obs and already added virtual)
                    rand_entry = random.sample(n_current, 1)[0]  # randomly sample 1 entry (obs or already added virtual)
                    rand_neighbors = neighbors[rand_entry]  # get 1-hop neighbors of sampled entry
                    p = np.random.rand(1)  # randomly generate a probability

                    # randomly select neighbors
                    valid_neighbors = (np.random.rand(len(rand_neighbors)) < p).astype(int)
                    valid_neighbors = np.where(valid_neighbors == 1)[0].tolist()
                    valid_neighbors = [rand_neighbors[idx] for idx in valid_neighbors]
                    all_entries = [rand_entry]
                    all_entries.extend(valid_neighbors)

                    # add current virtual entry to the 1-hop neighbors of selected entries
                    for entry in all_entries:
                        neighbors[entry].append(i)

                    # add selected entries to the 1-hop neighbors of current virtual entry
                    neighbors[i] = all_entries

                    options = [0, 1, 2]  # 0: forward; 1: backward; 2: bi-direction
                    connected_conditions = [random.choice(options) for _ in range(len(all_entries))]
                    for j in range(len(all_entries)):
                        entry = all_entries[j]
                        condition = connected_conditions[j]

                        if condition == 0 or condition == 2:
                            adj_aug_mask[entry, i] = 1
                        if condition == 1 or condition == 2:
                            adj_aug_mask[i, entry] = 1

                adj_aug = adj_aug * adj_aug_mask

                if self.dataset_name in ["sea_loop_point"]:
                    adj_aug[adj_aug > 0] = 1  # only for sea-loop, because their adj are binary

                self.adj_aug = adj_aug
            elif self.adj_aug is not None and known_set is not None:
                adj_aug = self.adj_aug
                adj = adj_aug.detach()

        supp = SpatialConvOrderK.compute_support(adj, x.device)

        imputation = self.impute(x, mask, supp)
        if not self.training:
            imputation = torch.where(mask_bool, x, imputation)
            return imputation
        else:
            y = torch.where(mask_bool, x, imputation)
            x = imputation * (1 - mask)
            imputation_cyc = self.impute(x, 1 - mask, supp)
            return imputation, imputation_cyc, y

    def impute(self, x, mask, supp):
        b, s, n, c = x.size()
        imputation = self.relu(self.fc_1(x))  # bs, s, n, dim
        imputation = rearrange(imputation, 'b s n d -> b d n s')
        d = imputation.size(1)

        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_1(imputation, supp_drop))

        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_2(imputation, supp_drop))

        # cross reference
        # b d n s
        feat = imputation.clone()
        feat = rearrange(feat, 'b d n s -> (b s) n d')
        feat_mask = rearrange(mask, 'b s n d -> (b s) n d')

        cosine_eps = 1e-7
        q = feat.clone()  # b n d
        k = feat.clone().transpose(-2, -1)  # b n d
        q_norm = torch.norm(q, 2, 2, True)
        k_norm = torch.norm(k, 2, 1, True)

        # ========================================
        # Hard Transfer
        # Align each obs road with the most similar unobs road
        # Align each unobs road with the most similar obs road
        # ========================================
        cos_sim = torch.bmm(q, k) / (torch.bmm(q_norm, k_norm) + cosine_eps)  # b n n, [-1, 1]
        cos_sim = (cos_sim + 1.) / 2.  # [0, 1]

        cos_sim_max = cos_sim * feat_mask  # observed positions
        cos_sim_max_score, cos_sim_max_index = torch.max(cos_sim_max, dim=1)  # b n
        cos_sim_min = cos_sim * (1. - feat_mask)  # unobserved positions
        cos_sim_min_score, cos_sim_min_index = torch.max(cos_sim_min, dim=1)  # b n

        v = feat.clone().transpose(-2, -1)  # b d n
        v_unobs = self.bis(v, 2, cos_sim_max_index)  # find the most similar observed road for each unobserved road
        v_obs = self.bis(v, 2, cos_sim_min_index)  # find the most dissimilar unobserved road for each observed road
        v_unobs = v_unobs * cos_sim_max_score.unsqueeze(1)  # b d n
        v_obs = v_obs * cos_sim_min_score.unsqueeze(1)  # b d n

        v_unobs = rearrange(v_unobs, '(b s) d n -> b d n s', b=b, s=s)
        v_obs = rearrange(v_obs, '(b s) d n -> b d n s', b=b, s=s)

        feat_mask = rearrange(feat_mask, '(b s) n d -> b d n s', b=b, s=s)
        feat_transfer = v_unobs * (1. - feat_mask) + v_obs * feat_mask  # b d n s

        imputation = torch.cat([imputation, feat_transfer], dim=1)
        imputation = rearrange(imputation, 'b d n s -> b s n d')
        imputation = self.relu(self.smooth(imputation))
        imputation = rearrange(imputation, 'b s n d -> b d n s')

        # ========================================
        # Output
        # ========================================
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_3(imputation, supp_drop))

        imputation = rearrange(imputation, 'b d n s -> b s n d')
        imputation = self.fc_2(imputation)  # b s n d
        return imputation

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


def _resolve_model_module_name(model_name):
    if model_name == "GWNET":
        return "GWNet"
    return model_name


def _load_backbone_from_spec(spec, device):
    model_name = spec["model"]
    config_path = spec["config_path"]
    checkpoint_path = spec["checkpoint_path"]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LifecycleMoGERNN backbone config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LifecycleMoGERNN backbone checkpoint not found: {checkpoint_path}")
    backbone_args = load_args_from_config(config_path, overrides={"device": device})
    if backbone_args.model != model_name:
        backbone_args.model = model_name
    module_name = _resolve_model_module_name(model_name)
    module = importlib.import_module(f"model.{module_name}")
    backbone = module.Model(backbone_args).float()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Backbone checkpoint at {checkpoint_path} must contain a 'state_dict'")
    backbone.load_state_dict(checkpoint["state_dict"])
    return backbone, backbone_args, os.path.normpath(checkpoint_path)


class _SilentLogger:

    def info(self, *args, **kwargs):
        return None


class STBPResidualAdapter(nn.Module):

    def __init__(self, configs, num_nodes, recipe_cfg):
        super().__init__()
        hidden_channel = int(recipe_cfg.get("hidden_channel", max(192, configs.d_model * 3)))
        dropout = float(recipe_cfg.get("dropout", 0.0))
        self.residual_scale = float(recipe_cfg.get("residual_scale", 0.2))
        self.output_dim = configs.c_out
        self.pred_len = configs.pred_len
        self.model_args = SimpleNamespace(
            dropout=dropout,
            y_len=self.pred_len,
            base_node_size=num_nodes,
            year=0,
            model={
                "in_channel": self.pred_len,
                "out_channel": self.pred_len,
                "hidden_channel": hidden_channel,
            },
            logger=_SilentLogger(),
        )
        self.backbone = STBP_Model(self.model_args)

    def forward(self, pred, adj, current_node_ids=None):
        if pred.size(-1) != 1:
            raise ValueError("STBPResidualAdapter currently supports single-channel target_features=v")
        N = adj.shape[0]
        if N > self.backbone.num_nodes:
            self.backbone.expand_adaptive_params(N)
        x = pred[..., 0].permute(0, 2, 1).reshape(-1, self.pred_len).reshape((-1, N, self.backbone.args.model["in_channel"]))
        B, N, T = x.shape
        x = F.relu(self.backbone.fconv1(x))

        if current_node_ids is None:
            current_node_ids = list(range(N))
        adaptive_params = self.backbone.pattern_bank[current_node_ids]

        node_emb = adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape).permute(0,2,1).unsqueeze(-1)  # [bs, N, feature]
        x = x.permute(0,2,1).unsqueeze(-1)

        x = self.backbone.stmodule(x, node_emb).squeeze(-1)
        x = x.permute(0,2,1).squeeze(-1)
        x = self.backbone.fconv2(x)
        x = x.reshape((-1, self.backbone.args.model["out_channel"]))
        x = x + pred[..., 0].permute(0, 2, 1).reshape(-1, self.pred_len)
        residual = self.backbone.fc(self.backbone.activation(x))
        residual = F.dropout(residual, p=self.backbone.dropout, training=self.backbone.training)
        residual = residual.reshape(pred.shape[0], pred.shape[2], self.pred_len).permute(0, 2, 1).unsqueeze(-1).contiguous()
        return pred + self.residual_scale * residual


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.cfg = getattr(configs, "model_family_cfg", {}) or {}
        self.recipe = configs.model_recipe
        if self.recipe not in {"kits_comp", "stbp_residual", "kits_stbp"}:
            raise ValueError(f"Unsupported LifecycleMoGERNN recipe: {self.recipe}")
        backbone_cfg = self.cfg.get("backbone", {})
        if not backbone_cfg:
            raise ValueError("LifecycleMoGERNN family.backbone must define model/config_path/checkpoint_path")
        self.backbone, self.backbone_args, self.backbone_checkpoint_path = _load_backbone_from_spec(backbone_cfg, configs.device)
        self.freeze_backbone = bool(self.cfg.get("recipe", {}).get("freeze_backbone", True))
        self.raw_enc_in = configs.raw_enc_in
        self.total_enc_in = configs.enc_in
        self.output_dim = configs.c_out
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + 4
        self.age_idx = self.role_end
        self.kits_lambda_cycle = float(self.cfg.get("recipe", {}).get("kits_lambda_cycle", 0.0))

        full_adj = load_project_adjacency({"data_path": configs.data_path}).cpu().numpy()
        self.full_num_nodes = int(full_adj.shape[0])
        self.known_nodes, _ = resolve_known_unknown_nodes(
            {
                "data_path": configs.data_path,
                "unknown_nodes_path": configs.unknown_nodes_path,
                "unknown_nodes_list": configs.unknown_nodes_list,
                "unknown_nodes_num": configs.unknown_nodes_num,
                "seed": configs.seed,
            }
        )
        recipe_cfg = self.cfg.get("recipe", {})
        d_hidden = int(recipe_cfg.get("kits_hidden_dim", configs.d_model))
        kits_args = SimpleNamespace(
            dataset_name=configs.dloader_name,
            use_adj_drop=bool(recipe_cfg.get("kits_use_adj_drop", False)),
            use_init=bool(recipe_cfg.get("kits_use_init", True)),
        )
        self.kits = None
        if self.recipe in {"kits_comp", "kits_stbp"}:
            self.kits = KITS(full_adj, configs.raw_enc_in, d_hidden, kits_args)
        self.stbp_residual = None
        if self.recipe in {"stbp_residual", "kits_stbp"}:
            self.stbp_residual = STBPResidualAdapter(configs, full_adj.shape[0], recipe_cfg)

        self._configure_backbone_trainability()

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _configure_backbone_trainability(self):
        if self.freeze_backbone:
            self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze_backbone

    def train(self, mode=True):
        super().train(mode)
        self._configure_backbone_trainability()
        return self

    def _visible_mask(self, x_enc):
        if x_enc.size(-1) >= self.role_end:
            role_seq = x_enc[..., self.role_start:self.role_end]
            return ((role_seq[..., 0:1] + role_seq[..., 1:2]) > 0.5)
        return torch.abs(x_enc[..., : self.raw_enc_in]) > 0

    def _compensate_input(self, adj, x_enc):
        if self.kits is None:
            return x_enc, {}
        raw_x = x_enc[..., : self.raw_enc_in]
        extra_x = x_enc[..., self.raw_enc_in:] if x_enc.size(-1) > self.raw_enc_in else None
        mask = self._visible_mask(x_enc).expand_as(raw_x)
        self.kits.adj = adj.detach().clone()
        kits_output = self.kits(raw_x, mask=mask)
        aux_losses = {}
        if isinstance(kits_output, tuple):
            imputation, imputation_cyc, y = kits_output
            comp_x = y
            if self.kits_lambda_cycle > 0 and mask.any():
                aux_losses["kits_cycle"] = self.kits_lambda_cycle * F.mse_loss(imputation_cyc[mask], raw_x[mask])
        else:
            comp_x = kits_output
        if extra_x is None:
            return comp_x, aux_losses
        return torch.cat([comp_x, extra_x], dim=-1), aux_losses

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        x_enc, aux_losses = self._compensate_input(adj, x_enc)
        backbone_x = x_enc[..., : self.raw_enc_in]
        backbone_x_dec = None if x_dec is None else x_dec[..., : self.output_dim]
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_output = self.backbone(adj, backbone_x, x_t_mark, pos_mark, backbone_x_dec, dec_t_mark, epoch)
        else:
            backbone_output = self.backbone(adj, backbone_x, x_t_mark, pos_mark, backbone_x_dec, dec_t_mark, epoch)
        pred, backbone_meta = unpack_model_output(backbone_output)
        if self.stbp_residual is not None:
            current_node_ids = self._resolve_current_node_ids(pred.shape[2])
            pred = self.stbp_residual(pred, adj, current_node_ids=current_node_ids)
        if backbone_meta.get("aux_losses"):
            aux_losses.update(backbone_meta["aux_losses"])
        return {"pred": pred, "aux_losses": aux_losses}
