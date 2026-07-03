import math

from layers.gnn import nn, torch, calculate_random_walk_matrix
import torch.nn.functional as F
from einops import rearrange
from layers.agg import MeanAggregator, WMeanAggregator, MaxPoolingAggregator, MinPoolingAggregator, DiffusionAggregator
from layers.DCRNNCell import AdaptiveGatedFFNDCRNNCell, DCRNNCell, DiffConv, GatedFFNDCRNNCell, HighwayDCRNNCell, LayerNormDCRNNCell, LocalGateDCRNNCell, MGDCRNNCell, MixHopDCRNNCell, ResidualDCRNNCell, SourceMixHopDCRNNCell, StateDecayDCRNNCell
from layers.source_kan import KANLayer
from layers.source_grin import GRINInputImputer
from lib.utils import infer_project_num_nodes


class Expert(nn.Module):

    def __init__(self, completion: nn.Module):
        super(Expert, self).__init__()
        self.completion = completion

    def forward(self, adj, x_enc, batch_unknown_nodes):
        return self.completion(adj, x_enc, batch_unknown_nodes)


class GatingNetwork(nn.Module):

    def __init__(self, configs, input_dim=None):
        super(GatingNetwork, self).__init__()
        raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        gate_input_dim = configs.input_len * (raw_enc_in if input_dim is None else input_dim)
        self.fc1 = nn.Linear(gate_input_dim, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SparseGatingNetwork(nn.Module):

    def __init__(self, configs, input_dim=None, recipe_cfg=None):
        super(SparseGatingNetwork, self).__init__()
        recipe_cfg = recipe_cfg or {}
        raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        gate_input_dim = configs.input_len * (raw_enc_in if input_dim is None else input_dim)
        self.fc1 = nn.Linear(gate_input_dim, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.num_experts)
        self.k = max(1, min(int(configs.num_used_experts), int(configs.num_experts)))
        self.temperature = max(float(recipe_cfg.get("mogernn_router_temperature", 1.0)), 1e-4)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，维度为 [batch_size, input_dim]

        Returns:
            torch.Tensor: 每个专家的权重，维度为 [batch_size, num_experts]
        """
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        top_logits, topk_indices = torch.topk(logits, self.k, dim=-1)
        zeros = torch.zeros_like(logits)
        weights = zeros.scatter(2, topk_indices, F.softmax(top_logits / self.temperature, dim=-1))
        return weights


class SparseKANGatingNetwork(nn.Module):

    def __init__(self, configs, input_dim=None, recipe_cfg=None):
        super(SparseKANGatingNetwork, self).__init__()
        recipe_cfg = recipe_cfg or {}
        raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        gate_input_dim = configs.input_len * (raw_enc_in if input_dim is None else input_dim)
        hidden_dim = int(recipe_cfg.get("kan_router_hidden_dim", configs.d_model))
        self.norm = nn.LayerNorm(gate_input_dim)
        self.kan = KANLayer(
            in_dim=gate_input_dim,
            out_dim=hidden_dim,
            num=int(recipe_cfg.get("kan_router_grid_size", 8)),
            k=int(recipe_cfg.get("kan_router_spline_order", 3)),
            noise_scale=float(recipe_cfg.get("kan_router_noise_scale", 0.5)),
            scale_sp=float(recipe_cfg.get("kan_router_spline_scale", 0.10)),
            grid_range=[
                float(recipe_cfg.get("kan_router_grid_min", -2.0)),
                float(recipe_cfg.get("kan_router_grid_max", 2.0)),
            ],
            sparse_init=bool(recipe_cfg.get("kan_router_sparse_init", False)),
        )
        self.fc = nn.Linear(hidden_dim, configs.num_experts)
        self.k = max(1, min(int(configs.num_used_experts), int(configs.num_experts)))
        self.temperature = max(float(recipe_cfg.get("mogernn_router_temperature", 1.0)), 1e-4)

    def forward(self, x):
        hidden, _, _, _ = self.kan(self.norm(x).reshape(-1, x.size(-1)))
        hidden = F.gelu(hidden.reshape(*x.shape[:-1], -1))
        logits = self.fc(hidden)
        top_logits, topk_indices = torch.topk(logits, self.k, dim=-1)
        zeros = torch.zeros_like(logits)
        weights = zeros.scatter(2, topk_indices, F.softmax(top_logits / self.temperature, dim=-1))
        return weights


class DataPreparation(nn.Module):

    def __init__(self, configs):
        super(DataPreparation, self).__init__()
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.total_enc_in = configs.enc_in
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + 4
        self.age_idx = self.role_end
        recipe_cfg = (getattr(configs, "model_family_cfg", {}) or {}).get("recipe", {})
        self.use_mask_aware_gate = bool(recipe_cfg.get("use_mask_aware_gate", False))
        self.use_observed_mask_gate = bool(recipe_cfg.get("use_observed_mask_gate", True))
        self.use_role_gate_features = bool(recipe_cfg.get("use_role_gate_features", True))
        self.use_age_gate_feature = bool(recipe_cfg.get("use_age_gate_feature", True))
        self.use_temporal_carry_imputer = bool(recipe_cfg.get("use_temporal_carry_imputer", False))
        self.temporal_carry_mix = float(recipe_cfg.get("temporal_carry_mix", 0.50))
        self.use_temporal_carry_gate = bool(recipe_cfg.get("use_temporal_carry_gate", False))
        self.temporal_carry_gate_max_mix = float(recipe_cfg.get("temporal_carry_gate_max_mix", 0.50))
        gate_feature_dim = self.raw_enc_in
        if self.use_mask_aware_gate:
            if self.use_observed_mask_gate:
                gate_feature_dim += 1
            if self.use_role_gate_features and self.total_enc_in >= self.role_end:
                gate_feature_dim += 4
            if self.use_age_gate_feature and self.total_enc_in > self.age_idx:
                gate_feature_dim += 1
        flattened_input_dim = configs.input_len * self.raw_enc_in
        experts = []
        if configs.mean_expert:
            experts.append(MeanAggregator(flattened_input_dim,
                                          flattened_input_dim))
        if configs.weight_expert:
            experts.append(
                WMeanAggregator(flattened_input_dim, flattened_input_dim))
        if configs.max_expert:
            experts.append(
                MaxPoolingAggregator(flattened_input_dim, flattened_input_dim))
        if configs.min_expert:
            experts.append(
                MinPoolingAggregator(flattened_input_dim, flattened_input_dim))
        if configs.diffusion_expert:
            experts.append(
                DiffusionAggregator(flattened_input_dim, flattened_input_dim))
        self.experts = nn.ModuleList([Expert(expert) for expert in experts])
        router_type = str(recipe_cfg.get("mogernn_router_type", recipe_cfg.get("router_type", "sparse_mlp"))).lower()
        if router_type in {"kan", "source_kan", "rbf_kan", "sparse_kan"}:
            self.gating = SparseKANGatingNetwork(configs, input_dim=gate_feature_dim, recipe_cfg=recipe_cfg)
        else:
            self.gating = SparseGatingNetwork(configs, input_dim=gate_feature_dim, recipe_cfg=recipe_cfg)
        if self.use_temporal_carry_gate:
            self.temporal_carry_gate = nn.Linear(self.raw_enc_in * 4, self.raw_enc_in)
            nn.init.zeros_(self.temporal_carry_gate.weight)
            init_gate = float(recipe_cfg.get("temporal_carry_gate_init", 0.10))
            init_gate = min(max(init_gate, 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.temporal_carry_gate.bias, math.log(init_gate / (1.0 - init_gate)))
        else:
            self.temporal_carry_gate = None

    def _observed_mask(self, raw_x, extra_x):
        if extra_x is not None and self.total_enc_in >= self.role_end:
            role_seq = extra_x[..., :4]
            return ((role_seq[..., 0:1] + role_seq[..., 1:2]) > 0.5).to(dtype=raw_x.dtype)
        return (torch.abs(raw_x).sum(dim=-1, keepdim=True) > 0).to(dtype=raw_x.dtype)

    def _build_gate_input(self, raw_x, extra_x):
        if not self.use_mask_aware_gate:
            return raw_x
        features = [raw_x]
        if self.use_observed_mask_gate:
            features.append(self._observed_mask(raw_x, extra_x))
        if extra_x is not None and self.use_role_gate_features and self.total_enc_in >= self.role_end:
            features.append(extra_x[..., :4])
        if extra_x is not None and self.use_age_gate_feature and self.total_enc_in > self.age_idx:
            features.append(extra_x[..., 4:5])
        return torch.cat(features, dim=-1)

    def _temporal_carry_completion(self, raw_x):
        batch_size, _, num_nodes, raw_dim = raw_x.shape
        last_value = raw_x.new_zeros(batch_size, num_nodes, raw_dim)
        has_value = raw_x.new_zeros(batch_size, num_nodes, raw_dim)
        carry_steps = []
        available_steps = []
        for t in range(raw_x.size(1)):
            carry_steps.append(last_value)
            available_steps.append(has_value)
            observed = (torch.abs(raw_x[:, t]) > 0).to(dtype=raw_x.dtype)
            last_value = torch.where(observed.bool(), raw_x[:, t], last_value)
            has_value = torch.maximum(has_value, observed)
        return torch.stack(carry_steps, dim=1), torch.stack(available_steps, dim=1)

    def _temporal_carry_gate_completion(self, spatial_completed, carry_completed, carry_available):
        gate_features = torch.cat(
            [spatial_completed, carry_completed, spatial_completed - carry_completed, carry_available],
            dim=-1,
        )
        max_mix = max(0.0, min(1.0, self.temporal_carry_gate_max_mix))
        gate = max_mix * carry_available * torch.sigmoid(self.temporal_carry_gate(gate_features))
        return (1.0 - gate) * spatial_completed + gate * carry_completed

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None):
        if x_enc.size(-1) != self.total_enc_in:
            raise ValueError(
                f"Lifecycle input channel mismatch: model expects {self.total_enc_in} channels "
                f"(raw={self.raw_enc_in}, extra={self.total_enc_in - self.raw_enc_in}), "
                f"but dataloader produced {x_enc.size(-1)}. "
                "Ensure train/test use the same lifecycle feature flags and dataset branch."
            )
        raw_x = x_enc[..., :self.raw_enc_in]
        extra_x = x_enc[..., self.raw_enc_in:] if self.total_enc_in > self.raw_enc_in else None
        gate_x = self._build_gate_input(raw_x, extra_x)
        batch_unknown_nodes = torch.where(
            torch.sum(torch.abs(raw_x), dim=(0, 1, 3)) == 0)[0].cpu().tolist()
        raw_seq = raw_x
        raw_x = rearrange(raw_seq, 'b t n c -> b n (t c)')
        gate_x = rearrange(gate_x, 'b t n c -> b n (t c)')
        gate_weights = self.gating(gate_x)
        adj_noselfloop = adj.clone()
        adj_noselfloop[torch.eye(adj.shape[0]).bool()] = 0
        expert_outputs = [
            expert(adj_noselfloop, raw_x, batch_unknown_nodes)
            for expert in self.experts
        ]
        expert_outputs_stack = torch.stack(expert_outputs, dim=2)
        combied_out = torch.sum(gate_weights.unsqueeze(-1) *
                                expert_outputs_stack,
                                dim=2)
        spatial_completed = rearrange(combied_out, 'b n (t c) -> b t n c', t=x_enc.shape[1], c=self.raw_enc_in)
        fill_completed = spatial_completed
        if self.use_temporal_carry_gate:
            carry_completed, carry_available = self._temporal_carry_completion(raw_seq)
            fill_completed = self._temporal_carry_gate_completion(spatial_completed, carry_completed, carry_available)
        elif self.use_temporal_carry_imputer:
            carry_completed, carry_available = self._temporal_carry_completion(raw_seq)
            carry_mix = max(0.0, min(1.0, self.temporal_carry_mix))
            carry_weight = carry_available * carry_mix
            fill_completed = (1.0 - carry_weight) * spatial_completed + carry_weight * carry_completed
        miss_ix = (raw_seq == 0).to(dtype=raw_seq.dtype)
        raw_x = raw_seq * (1.0 - miss_ix) + fill_completed * miss_ix
        if extra_x is None:
            return raw_x
        return torch.cat([raw_x, extra_x], dim=-1)


class TemporalInputMixer(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation=1, scale=0.05):
        super(TemporalInputMixer, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        batch_size, seq_len, num_nodes, input_dim = x.shape
        residual = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, input_dim, seq_len)
        residual = self.net(residual)
        residual = residual.reshape(batch_size, num_nodes, input_dim, seq_len).permute(0, 3, 1, 2)
        return x + self.scale * residual


class AdaptiveGraphSupport(nn.Module):

    def __init__(self, num_nodes, embed_dim=8, mix=0.05, temperature=1.0, symmetric=False):
        super(AdaptiveGraphSupport, self).__init__()
        self.mix = float(mix)
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.node_emb_src = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.node_emb_dst = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb_src)
        nn.init.xavier_uniform_(self.node_emb_dst)

    def _select_node_embeddings(self, num_nodes, node_ids, device):
        if node_ids is not None:
            node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=device)
            if node_ids.numel() == num_nodes and node_ids.max().item() < self.node_emb_src.size(0):
                return self.node_emb_src[node_ids], self.node_emb_dst[node_ids]
        if num_nodes <= self.node_emb_src.size(0):
            return self.node_emb_src[:num_nodes], self.node_emb_dst[:num_nodes]
        raise ValueError(
            f"AdaptiveGraphSupport was initialized for {self.node_emb_src.size(0)} nodes, "
            f"but received a {num_nodes}-node adjacency."
        )

    def forward(self, A_q, A_h, node_ids=None):
        num_nodes = int(A_q.size(0))
        node_emb_src, node_emb_dst = self._select_node_embeddings(num_nodes, node_ids, A_q.device)
        logits = torch.relu(torch.matmul(node_emb_src, node_emb_dst.transpose(0, 1)))
        logits = logits / max(self.temperature, 1e-6)
        adaptive = torch.softmax(logits, dim=-1)
        if self.symmetric:
            adaptive = 0.5 * (adaptive + adaptive.transpose(0, 1))
            adaptive = adaptive / adaptive.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mix = max(0.0, min(1.0, self.mix))
        return (1.0 - mix) * A_q + mix * adaptive, (1.0 - mix) * A_h + mix * adaptive.transpose(0, 1)


class LifecycleOutputAdapter(nn.Module):

    def __init__(self, raw_dim, output_dim, hidden_dim, scale=0.05):
        super(LifecycleOutputAdapter, self).__init__()
        self.raw_dim = int(raw_dim)
        self.output_dim = int(output_dim)
        self.scale = float(scale)
        state_dim = 4 + 1 + 1 + self.raw_dim
        self.net = nn.Sequential(
            nn.Linear(self.output_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pred, x_enc):
        if x_enc.size(-1) < self.raw_dim + 5:
            return pred
        raw_x = x_enc[..., :self.raw_dim]
        role_seq = x_enc[..., self.raw_dim:self.raw_dim + 4]
        last_role = role_seq[:, -1]
        last_age = x_enc[:, -1, :, self.raw_dim + 4:self.raw_dim + 5]
        visibility_ratio = (role_seq[..., 0:1] + role_seq[..., 1:2]).mean(dim=1)
        last_raw = raw_x[:, -1]
        state = torch.cat([last_role, last_age, visibility_ratio, last_raw], dim=-1)
        state = state.unsqueeze(1).expand(-1, pred.size(1), -1, -1)
        residual = torch.tanh(self.net(torch.cat([pred, state], dim=-1)))
        return pred + self.scale * residual


class RoleOutputCalibrator(nn.Module):

    def __init__(
        self,
        raw_dim,
        output_dim,
        scale=0.05,
        use_multiplicative=True,
        use_bias=True,
        bias_weight=1.0,
        scale_weight=1.0,
        learned_component_gate=False,
        component_gate_max=2.0,
        out_seq_len=None,
        horizon_conditioned=False,
        state_aware=False,
        use_age=True,
        use_visibility=True,
    ):
        super(RoleOutputCalibrator, self).__init__()
        self.raw_dim = int(raw_dim)
        self.output_dim = int(output_dim)
        self.scale = float(scale)
        self.use_multiplicative = bool(use_multiplicative)
        self.use_bias = bool(use_bias)
        self.bias_weight = float(bias_weight)
        self.scale_weight = float(scale_weight)
        self.learned_component_gate = bool(learned_component_gate)
        self.component_gate_max = float(component_gate_max)
        self.horizon_conditioned = bool(horizon_conditioned)
        self.state_aware = bool(state_aware)
        self.use_age = bool(use_age)
        self.use_visibility = bool(use_visibility)
        self.out_seq_len = int(out_seq_len or 0)
        if self.state_aware and self.horizon_conditioned:
            raise ValueError("RoleOutputCalibrator does not combine state_aware and horizon_conditioned yet.")
        if self.state_aware:
            state_dim = 4
            if self.use_age:
                state_dim += 1
            if self.use_visibility:
                state_dim += 1
            if self.use_bias:
                self.state_bias = nn.Linear(state_dim, self.output_dim, bias=False)
                nn.init.zeros_(self.state_bias.weight)
            else:
                self.state_bias = None
            if self.use_multiplicative:
                self.state_scale = nn.Linear(state_dim, self.output_dim, bias=False)
                nn.init.zeros_(self.state_scale.weight)
            else:
                self.state_scale = None
            self.register_parameter("role_bias", None)
            self.register_parameter("role_scale", None)
        elif self.horizon_conditioned:
            if self.out_seq_len <= 0:
                raise ValueError("RoleOutputCalibrator needs out_seq_len when horizon_conditioned=True.")
            if self.use_bias:
                self.role_bias = nn.Parameter(torch.zeros(self.out_seq_len, 4, self.output_dim))
            else:
                self.register_parameter("role_bias", None)
            if self.use_multiplicative:
                self.role_scale = nn.Parameter(torch.zeros(self.out_seq_len, 4, self.output_dim))
            else:
                self.register_parameter("role_scale", None)
        else:
            if self.use_bias:
                self.role_bias = nn.Parameter(torch.zeros(4, self.output_dim))
            else:
                self.register_parameter("role_bias", None)
            if self.use_multiplicative:
                self.role_scale = nn.Parameter(torch.zeros(4, self.output_dim))
            else:
                self.register_parameter("role_scale", None)
        gate_init = 1.0 / max(self.component_gate_max, 1e-6)
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        if self.learned_component_gate and self.use_bias:
            self.bias_gate_logits = nn.Parameter(torch.full((4, self.output_dim), gate_logit))
        else:
            self.register_parameter("bias_gate_logits", None)
        if self.learned_component_gate and self.use_multiplicative:
            self.scale_gate_logits = nn.Parameter(torch.full((4, self.output_dim), gate_logit))
        else:
            self.register_parameter("scale_gate_logits", None)

    def _state_features(self, x_enc, role):
        features = [role]
        if self.use_age:
            if x_enc.size(-1) >= self.raw_dim + 5:
                age = torch.tanh(x_enc[:, -1, :, self.raw_dim + 4:self.raw_dim + 5])
            else:
                age = role.new_zeros(*role.shape[:-1], 1)
            features.append(age)
        if self.use_visibility:
            role_seq = x_enc[..., self.raw_dim:self.raw_dim + 4]
            visibility_ratio = (role_seq[..., 0:1] + role_seq[..., 1:2]).mean(dim=1).clamp(0.0, 1.0)
            features.append(visibility_ratio)
        return torch.cat(features, dim=-1)

    def _component_gate(self, role, logits):
        if logits is None:
            return None
        gate_values = self.component_gate_max * torch.sigmoid(logits)
        return torch.matmul(role, gate_values).unsqueeze(1)

    def forward(self, pred, x_enc):
        if x_enc.size(-1) < self.raw_dim + 4:
            return pred
        role = x_enc[:, -1, :, self.raw_dim:self.raw_dim + 4]
        bias = None
        if self.state_aware:
            state = self._state_features(x_enc, role)
            if self.state_bias is not None:
                bias = self.bias_weight * self.state_bias(state).unsqueeze(1)
        elif self.horizon_conditioned:
            if self.role_bias is not None:
                bias = self.bias_weight * torch.einsum("bnr,hrc->bhnc", role, self.role_bias)
        elif self.role_bias is not None:
            bias = self.bias_weight * torch.matmul(role, self.role_bias).unsqueeze(1)
        bias_gate = self._component_gate(role, self.bias_gate_logits)
        if bias is not None and bias_gate is not None:
            bias = bias * bias_gate
        residual = bias
        if self.state_aware and self.state_scale is not None:
            scale = torch.tanh(self.state_scale(state)).unsqueeze(1)
            scale_residual = self.scale_weight * pred * scale
            scale_gate = self._component_gate(role, self.scale_gate_logits)
            if scale_gate is not None:
                scale_residual = scale_residual * scale_gate
            residual = scale_residual if residual is None else residual + scale_residual
        elif self.role_scale is not None:
            if self.horizon_conditioned:
                scale = torch.tanh(torch.einsum("bnr,hrc->bhnc", role, self.role_scale))
            else:
                scale = torch.tanh(torch.matmul(role, self.role_scale)).unsqueeze(1)
            scale_residual = self.scale_weight * pred * scale
            scale_gate = self._component_gate(role, self.scale_gate_logits)
            if scale_gate is not None:
                scale_residual = scale_residual * scale_gate
            residual = scale_residual if residual is None else residual + scale_residual
        if residual is None:
            return pred
        return pred + self.scale * residual


def _build_recurrent_cell(cell_type, input_size, hidden_size, k):
    cell_type = str(cell_type).lower()
    if cell_type in {"dcrnn", "dcgru", "graph_gru"}:
        return DCRNNCell(input_size, hidden_size, k)
    if cell_type in {"ln_dcrnn", "layernorm_dcrnn", "norm_dcrnn"}:
        return LayerNormDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"res_dcrnn", "residual_dcrnn", "skip_dcrnn"}:
        return ResidualDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"mgu_dcrnn", "mg_dcrnn", "minimal_dcrnn"}:
        return MGDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"localgate_dcrnn", "local_gate_dcrnn", "lg_dcrnn"}:
        return LocalGateDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"highway_dcrnn", "hw_dcrnn", "carry_dcrnn"}:
        return HighwayDCRNNCell(input_size, hidden_size, k)
    if cell_type.startswith("highway_dcrnn"):
        max_scale = 0.10
        init_gate = 0.10
        for token in cell_type.split("_"):
            if token.startswith("s") and len(token) > 1:
                scale_token = token[1:]
                max_scale = float(scale_token) if "." in scale_token else float(scale_token) / 100.0
            elif token.startswith("g") and len(token) > 1:
                gate_token = token[1:]
                init_gate = float(gate_token) if "." in gate_token else float(gate_token) / 100.0
        return HighwayDCRNNCell(input_size, hidden_size, k, max_scale=max_scale, init_gate=init_gate)
    if cell_type in {"state_decay_dcrnn", "statedecay_dcrnn", "grud_dcrnn"}:
        return StateDecayDCRNNCell(input_size, hidden_size, k)
    if cell_type.startswith("state_decay_dcrnn") or cell_type.startswith("statedecay_dcrnn") or cell_type.startswith("grud_dcrnn"):
        max_decay_scale = 0.10
        init_decay = 0.30
        for token in cell_type.split("_"):
            if token.startswith("s") and len(token) > 1 and token[1:].replace(".", "", 1).isdigit():
                scale_token = token[1:]
                max_decay_scale = float(scale_token) if "." in scale_token else float(scale_token) / 100.0
            elif token.startswith("g") and len(token) > 1 and token[1:].replace(".", "", 1).isdigit():
                gate_token = token[1:]
                init_decay = float(gate_token) if "." in gate_token else float(gate_token) / 100.0
        return StateDecayDCRNNCell(input_size, hidden_size, k, max_decay_scale=max_decay_scale, init_decay=init_decay)
    if cell_type in {"gffn_dcrnn", "swiglu_dcrnn", "grn_dcrnn"}:
        return GatedFFNDCRNNCell(input_size, hidden_size, k)
    if cell_type.startswith("gffn_dcrnn_s"):
        scale_token = cell_type.rsplit("_s", 1)[-1]
        transition_scale = float(scale_token) if "." in scale_token else float(scale_token) / 100.0
        return GatedFFNDCRNNCell(input_size, hidden_size, k, transition_scale=transition_scale)
    if cell_type.startswith("agffn_dcrnn"):
        max_scale = 0.10
        init_gate = 0.30
        for token in cell_type.split("_"):
            if token.startswith("s") and len(token) > 1:
                scale_token = token[1:]
                max_scale = float(scale_token) if "." in scale_token else float(scale_token) / 100.0
            elif token.startswith("g") and len(token) > 1:
                gate_token = token[1:]
                init_gate = float(gate_token) if "." in gate_token else float(gate_token) / 100.0
        return AdaptiveGatedFFNDCRNNCell(input_size, hidden_size, k, max_scale=max_scale, init_gate=init_gate)
    if cell_type in {"mixhop_dcrnn", "mh_dcrnn", "multiorder_dcrnn"}:
        return MixHopDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"source_mixhop_dcrnn", "src_mixhop_dcrnn", "mixhop_source_dcrnn"}:
        return SourceMixHopDCRNNCell(input_size, hidden_size, k)
    raise ValueError(f"Unsupported MoGERNN recurrent cell_type: {cell_type}")


class GraphImputerDirection(nn.Module):

    def __init__(self, raw_dim, extra_dim, hidden_dim, k=1, cell_type="ln_dcrnn"):
        super(GraphImputerDirection, self).__init__()
        self.raw_dim = int(raw_dim)
        self.extra_dim = int(extra_dim)
        self.hidden_dim = int(hidden_dim)
        self.rnn_input_dim = self.raw_dim + 1 + self.extra_dim
        self.first_stage = nn.Linear(self.hidden_dim, self.raw_dim)
        self.spatial_decoder = DiffConv(self.raw_dim + 1 + self.extra_dim + self.hidden_dim, self.raw_dim, k)
        self.cell = _build_recurrent_cell(cell_type, self.rnn_input_dim, self.hidden_dim, k)
        self.norm = nn.LayerNorm(self.raw_dim)

    def forward(self, raw_x, mask, extra_x, A_q, A_h, reverse=False):
        batch_size, seq_len, num_nodes, _ = raw_x.shape
        h = raw_x.new_zeros(batch_size, num_nodes, self.hidden_dim)
        pred_steps = [None for _ in range(seq_len)]
        state_steps = [None for _ in range(seq_len)]
        time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in time_indices:
            raw_t = raw_x[:, t]
            mask_t = mask[:, t]
            extra_t = None if extra_x is None else extra_x[:, t]
            first_pred = self.first_stage(h)
            x_t = torch.where(mask_t.expand_as(raw_t).bool(), raw_t, first_pred)
            decoder_inputs = [x_t, mask_t]
            if extra_t is not None:
                decoder_inputs.append(extra_t)
            decoder_inputs.append(h)
            spatial_pred = self.spatial_decoder(torch.cat(decoder_inputs, dim=-1), A_q, A_h)
            spatial_pred = self.norm(spatial_pred)
            x_t = torch.where(mask_t.expand_as(raw_t).bool(), raw_t, spatial_pred)
            cell_inputs = [x_t, mask_t]
            if extra_t is not None:
                cell_inputs.append(extra_t)
            h = self.cell(torch.cat(cell_inputs, dim=-1), h, A_q, A_h)
            pred_steps[t] = spatial_pred
            state_steps[t] = h
        return torch.stack(pred_steps, dim=1), torch.stack(state_steps, dim=1)


class GraphBidirInputImputer(nn.Module):

    def __init__(self, configs, recipe_cfg):
        super(GraphBidirInputImputer, self).__init__()
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.total_enc_in = configs.enc_in
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + 4
        self.age_idx = self.role_end
        self.use_role_mask = bool(recipe_cfg.get("graph_imputer_use_role_mask", True))
        self.use_extra_features = bool(recipe_cfg.get("graph_imputer_use_extra_features", True))
        extra_dim = self.total_enc_in - self.raw_enc_in if self.use_extra_features else 0
        hidden_dim = int(recipe_cfg.get("graph_imputer_hidden_dim", configs.d_model))
        k_hop = int(recipe_cfg.get("graph_imputer_k_hop", configs.k_hop))
        cell_type = recipe_cfg.get("graph_imputer_cell_type", "ln_dcrnn")
        self.merge = str(recipe_cfg.get("graph_imputer_merge", "gate")).lower()
        self.fwd_imputer = GraphImputerDirection(self.raw_enc_in, extra_dim, hidden_dim, k_hop, cell_type)
        self.bwd_imputer = GraphImputerDirection(self.raw_enc_in, extra_dim, hidden_dim, k_hop, cell_type)
        merge_input_dim = 2 * hidden_dim + self.raw_enc_in + 1 + extra_dim
        self.merge_gate = nn.Linear(merge_input_dim, self.raw_enc_in)
        self.merge_mlp = nn.Sequential(
            nn.Linear(merge_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.raw_enc_in),
        )

    def _observed_mask(self, raw_x, extra_x):
        if self.use_role_mask and extra_x is not None and self.total_enc_in >= self.role_end:
            role_seq = extra_x[..., :4]
            return ((role_seq[..., 0:1] + role_seq[..., 1:2]) > 0.5).to(dtype=raw_x.dtype)
        return (torch.abs(raw_x).sum(dim=-1, keepdim=True) > 0).to(dtype=raw_x.dtype)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None):
        if x_enc.size(-1) != self.total_enc_in:
            raise ValueError(
                f"Lifecycle input channel mismatch: model expects {self.total_enc_in} channels "
                f"(raw={self.raw_enc_in}, extra={self.total_enc_in - self.raw_enc_in}), "
                f"but dataloader produced {x_enc.size(-1)}."
            )
        raw_x = x_enc[..., :self.raw_enc_in]
        extra_x = x_enc[..., self.raw_enc_in:] if self.total_enc_in > self.raw_enc_in else None
        mask = self._observed_mask(raw_x, extra_x)
        imputer_extra = extra_x if self.use_extra_features else None
        A_q = calculate_random_walk_matrix(adj)
        A_h = calculate_random_walk_matrix(adj.T)
        fwd_pred, fwd_state = self.fwd_imputer(raw_x, mask, imputer_extra, A_q, A_h, reverse=False)
        bwd_pred, bwd_state = self.bwd_imputer(raw_x, mask, imputer_extra, A_q, A_h, reverse=True)
        if self.merge == "mean":
            imputed_raw = 0.5 * (fwd_pred + bwd_pred)
        else:
            merge_inputs = [fwd_state, bwd_state, raw_x, mask]
            if imputer_extra is not None:
                merge_inputs.append(imputer_extra)
            merge_input = torch.cat(merge_inputs, dim=-1)
            if self.merge == "mlp":
                imputed_raw = self.merge_mlp(merge_input)
            elif self.merge == "gate":
                gate = torch.sigmoid(self.merge_gate(merge_input))
                imputed_raw = gate * fwd_pred + (1.0 - gate) * bwd_pred
            else:
                raise ValueError(f"Unsupported graph_imputer_merge: {self.merge}")
        filled_raw = torch.where(mask.expand_as(raw_x).bool(), raw_x, imputed_raw)
        if extra_x is None:
            return filled_raw
        return torch.cat([filled_raw, extra_x], dim=-1)


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k=2, cell_type="dcrnn"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = nn.ModuleList([
            _build_recurrent_cell(cell_type, input_size if i == 0 else hidden_size, hidden_size, k)
            for i in range(num_layers)
        ])

    def forward(self, x, A_q, A_h, h0=None):
        # x: [batch_size, seq_len, num_nodes, input_size]
        batch_size, seq_len, num_nodes, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers,
                             batch_size,
                             num_nodes,
                             self.hidden_size,
                             device=x.device)

        hidden_state = h0
        for t in range(seq_len):
            hs = []
            input = x[:, t, :, :]
            for l in range(self.num_layers):
                next_hidden = self.rnn_cells[l](input, hidden_state[l], A_q,
                                                A_h)
                input = next_hidden
                hs.append(next_hidden)
            hidden_state = torch.stack(hs, dim=0)
        return hidden_state


class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 out_seq_len,
                 output_size=None,
                 num_layers=1,
                 k=2,
                 cell_type="dcrnn",
                 output_mode="direct",
                 residual_scale=1.0):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = input_size if output_size is None else output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len
        self.output_mode = str(output_mode).lower()
        self.residual_scale = float(residual_scale)

        self.rnn_cells = nn.ModuleList([
            _build_recurrent_cell(cell_type, input_size if i == 0 else hidden_size, hidden_size, k)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self,
                intial_input,
                h,
                A_q,
                A_h,
                tearcher_input,
                teacher_forcing_ratio=0.5):
        # label_seq: [batch_size, seq_len, num_nodes, input_size]
        # h: [num_layers, batch_size, num_nodes, hidden_size]
        input_at_t = intial_input
        outputs = [None for _ in range(self.out_seq_len)]
        for t in range(self.out_seq_len):
            decoder_input = input_at_t
            hs = []
            for l in range(self.num_layers):
                next_h = self.rnn_cells[l](input_at_t, h[l], A_q, A_h)
                input_at_t = next_h
                hs.append(next_h)
            h = torch.stack(hs, dim=0)
            raw_output = self.fc(next_h)
            if self.output_mode in {"residual_delta", "delta", "residual"}:
                output = decoder_input[..., : self.output_size] + self.residual_scale * raw_output
            else:
                output = raw_output
            outputs[t] = output
            # Teacher forcing
            use_teacher_forcing = True if (
                torch.rand(1).item() < teacher_forcing_ratio
                and self.training) else False
            if use_teacher_forcing:
                input_at_t = tearcher_input[:, t]
            else:
                input_at_t = outputs[t]
        outputs = torch.stack(outputs, dim=1)  #type:ignore
        return outputs


class Seq2Seq(nn.Module):

    def __init__(
        self,
        enc_input_size,
        dec_input_size,
        hidden_size,
        num_layers,
        out_seq_len,
        k=2,
        cell_type="dcrnn",
        encoder_cell_type=None,
        decoder_cell_type=None,
        decoder_output_mode="direct",
        decoder_residual_scale=1.0,
    ):
        super(Seq2Seq, self).__init__()
        encoder_cell_type = cell_type if encoder_cell_type is None else encoder_cell_type
        decoder_cell_type = cell_type if decoder_cell_type is None else decoder_cell_type
        self.encoder = Encoder(enc_input_size, hidden_size, num_layers, k, cell_type=encoder_cell_type)
        self.decoder = Decoder(dec_input_size, hidden_size, out_seq_len,
                               output_size=dec_input_size, num_layers=num_layers, k=k, cell_type=decoder_cell_type,
                               output_mode=decoder_output_mode, residual_scale=decoder_residual_scale)

    def forward(self, x_enc, x_dec, A_q, A_h, initial_input, teacher_forcing_ratio=0.5):
        h = self.encoder(x_enc, A_q, A_h)
        dec_out = self.decoder(initial_input, h, A_q, A_h, x_dec,
                               teacher_forcing_ratio)
        return dec_out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.out_len = configs.pred_len if configs.pred_len > 0 else configs.look_back
        self.output_dim = configs.c_out
        recipe_cfg = (getattr(configs, "model_family_cfg", {}) or {}).get("recipe", {})
        if bool(recipe_cfg.get("use_grin_input_imputer", False)):
            self.data_pre = GRINInputImputer(configs, recipe_cfg)
        elif bool(recipe_cfg.get("use_graph_bidir_input_imputer", False)):
            self.data_pre = GraphBidirInputImputer(configs, recipe_cfg)
        else:
            self.data_pre = DataPreparation(configs)
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.use_temporal_input_mixer = bool(recipe_cfg.get("use_temporal_input_mixer", False))
        if self.use_temporal_input_mixer:
            self.temporal_input_mixer = TemporalInputMixer(
                self.raw_enc_in,
                int(recipe_cfg.get("temporal_mixer_hidden_dim", configs.d_model)),
                int(recipe_cfg.get("temporal_mixer_kernel_size", 3)),
                int(recipe_cfg.get("temporal_mixer_dilation", 1)),
                float(recipe_cfg.get("temporal_mixer_scale", 0.05)),
            )
        else:
            self.temporal_input_mixer = None
        self.current_node_ids = None
        self.adaptive_support = None
        if bool(recipe_cfg.get("use_adaptive_graph_support", False)):
            num_nodes = int(recipe_cfg.get("num_nodes", 0) or infer_project_num_nodes({"data_path": configs.data_path}))
            self.adaptive_support = AdaptiveGraphSupport(
                num_nodes,
                int(recipe_cfg.get("adaptive_graph_embed_dim", 8)),
                float(recipe_cfg.get("adaptive_graph_mix", 0.05)),
                float(recipe_cfg.get("adaptive_graph_temperature", 1.0)),
                bool(recipe_cfg.get("adaptive_graph_symmetric", False)),
            )
        self.lifecycle_output_adapter = None
        if bool(recipe_cfg.get("use_lifecycle_output_adapter", False)):
            self.lifecycle_output_adapter = LifecycleOutputAdapter(
                self.raw_enc_in,
                self.output_dim,
                int(recipe_cfg.get("lifecycle_output_adapter_hidden_dim", configs.d_model)),
                float(recipe_cfg.get("lifecycle_output_adapter_scale", 0.05)),
            )
        self.role_output_calibrator = None
        if bool(recipe_cfg.get("use_role_output_calibrator", False)):
            self.role_output_calibrator = RoleOutputCalibrator(
                raw_dim=self.raw_enc_in,
                output_dim=self.output_dim,
                scale=float(recipe_cfg.get("role_output_calibrator_scale", 0.05)),
                use_multiplicative=bool(recipe_cfg.get("role_output_calibrator_multiplicative", True)),
                use_bias=bool(recipe_cfg.get("role_output_calibrator_use_bias", True)),
                bias_weight=float(recipe_cfg.get("role_output_calibrator_bias_weight", 1.0)),
                scale_weight=float(recipe_cfg.get("role_output_calibrator_scale_weight", 1.0)),
                learned_component_gate=bool(recipe_cfg.get("role_output_calibrator_learned_component_gate", False)),
                component_gate_max=float(recipe_cfg.get("role_output_calibrator_component_gate_max", 2.0)),
                out_seq_len=self.out_len,
                horizon_conditioned=bool(recipe_cfg.get("role_output_calibrator_horizon_conditioned", False)),
                state_aware=bool(recipe_cfg.get("role_output_calibrator_state_aware", False)),
                use_age=bool(recipe_cfg.get("role_output_calibrator_use_age", True)),
                use_visibility=bool(recipe_cfg.get("role_output_calibrator_use_visibility", True)),
            )
        cell_type = recipe_cfg.get("cell_type", recipe_cfg.get("mogernn_cell_type", "dcrnn"))
        encoder_cell_type = recipe_cfg.get("encoder_cell_type", recipe_cfg.get("mogernn_encoder_cell_type", None))
        decoder_cell_type = recipe_cfg.get("decoder_cell_type", recipe_cfg.get("mogernn_decoder_cell_type", None))
        self.seq2seq = Seq2Seq(configs.enc_in, configs.c_out, configs.d_model,
                               configs.e_layers, self.out_len, configs.k_hop,
                               cell_type=cell_type,
                               encoder_cell_type=encoder_cell_type,
                               decoder_cell_type=decoder_cell_type,
                               decoder_output_mode=recipe_cfg.get("decoder_output_mode", "direct"),
                               decoder_residual_scale=float(recipe_cfg.get("decoder_residual_scale", 1.0)))

    def set_current_node_ids(self, node_ids):
        self.current_node_ids = node_ids

    def _mix_temporal_input(self, x_enc):
        if self.temporal_input_mixer is None:
            return x_enc
        raw_x = x_enc[..., :self.raw_enc_in]
        mixed_raw = self.temporal_input_mixer(raw_x)
        if x_enc.size(-1) == self.raw_enc_in:
            return mixed_raw
        return torch.cat([mixed_raw, x_enc[..., self.raw_enc_in:]], dim=-1)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, *args):
        x_dec = args[0]
        #输入的adj应该转一下
        A_q = calculate_random_walk_matrix(adj)
        A_h = calculate_random_walk_matrix(adj.T)
        if self.adaptive_support is not None:
            A_q, A_h = self.adaptive_support(A_q, A_h, node_ids=self.current_node_ids)
            self.current_node_ids = None
        x_enc = self.data_pre(adj, x_enc, x_t_mark, pos_mark)
        x_enc = self._mix_temporal_input(x_enc)
        epoch = args[-1]
        teacher_forcing_ratio = 1 - (epoch / 40)
        initial_input = x_enc[:, -1, :, : self.output_dim]
        dec_out = self.seq2seq(x_enc, x_dec, A_q, A_h, initial_input, teacher_forcing_ratio)
        if self.lifecycle_output_adapter is not None:
            dec_out = self.lifecycle_output_adapter(dec_out, x_enc)
        if self.role_output_calibrator is not None:
            dec_out = self.role_output_calibrator(dec_out, x_enc)
        return dec_out
