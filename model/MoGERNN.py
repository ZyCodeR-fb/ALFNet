from layers.gnn import nn, torch, calculate_random_walk_matrix
import torch.nn.functional as F
from einops import rearrange
from layers.agg import MeanAggregator, WMeanAggregator, MaxPoolingAggregator, MinPoolingAggregator, DiffusionAggregator
from layers.DCRNNCell import DCRNNCell, LayerNormDCRNNCell, MGDCRNNCell


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

    def __init__(self, configs, input_dim=None):
        super(SparseGatingNetwork, self).__init__()
        raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        gate_input_dim = configs.input_len * (raw_enc_in if input_dim is None else input_dim)
        self.fc1 = nn.Linear(gate_input_dim, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.num_experts)
        self.k = configs.num_used_experts

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
        weights = zeros.scatter(2, topk_indices, F.softmax(top_logits, dim=-1))
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
        self.gating = SparseGatingNetwork(configs, input_dim=gate_feature_dim)

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
        raw_x = rearrange(raw_x, 'b t n c -> b n (t c)')
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
        miss_ix = (raw_x == 0).int()
        raw_x = raw_x * (1 - miss_ix) + combied_out * miss_ix
        raw_x = rearrange(raw_x, 'b n (t c) -> b t n c', t=x_enc.shape[1], c=self.raw_enc_in)
        if extra_x is None:
            return raw_x
        return torch.cat([raw_x, extra_x], dim=-1)


def _build_recurrent_cell(cell_type, input_size, hidden_size, k):
    cell_type = str(cell_type).lower()
    if cell_type in {"dcrnn", "dcgru", "graph_gru"}:
        return DCRNNCell(input_size, hidden_size, k)
    if cell_type in {"ln_dcrnn", "layernorm_dcrnn", "norm_dcrnn"}:
        return LayerNormDCRNNCell(input_size, hidden_size, k)
    if cell_type in {"mgu_dcrnn", "mg_dcrnn", "minimal_dcrnn"}:
        return MGDCRNNCell(input_size, hidden_size, k)
    raise ValueError(f"Unsupported MoGERNN recurrent cell_type: {cell_type}")


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
                 cell_type="dcrnn"):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = input_size if output_size is None else output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len

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
            hs = []
            for l in range(self.num_layers):
                next_h = self.rnn_cells[l](input_at_t, h[l], A_q, A_h)
                input_at_t = next_h
                hs.append(next_h)
            h = torch.stack(hs, dim=0)
            output = self.fc(next_h)
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

    def __init__(self, enc_input_size, dec_input_size, hidden_size, num_layers, out_seq_len, k=2, cell_type="dcrnn"):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(enc_input_size, hidden_size, num_layers, k, cell_type=cell_type)
        self.decoder = Decoder(dec_input_size, hidden_size, out_seq_len,
                               output_size=dec_input_size, num_layers=num_layers, k=k, cell_type=cell_type)

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
        self.data_pre = DataPreparation(configs)
        recipe_cfg = (getattr(configs, "model_family_cfg", {}) or {}).get("recipe", {})
        cell_type = recipe_cfg.get("cell_type", recipe_cfg.get("mogernn_cell_type", "dcrnn"))
        self.seq2seq = Seq2Seq(configs.enc_in, configs.c_out, configs.d_model,
                               configs.e_layers, self.out_len, configs.k_hop, cell_type=cell_type)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, *args):
        x_dec = args[0]
        #输入的adj应该转一下
        A_q = calculate_random_walk_matrix(adj)
        A_h = calculate_random_walk_matrix(adj.T)
        x_enc = self.data_pre(adj, x_enc, x_t_mark, pos_mark)
        epoch = args[-1]
        teacher_forcing_ratio = 1 - (epoch / 40)
        initial_input = x_enc[:, -1, :, : self.output_dim]
        dec_out = self.seq2seq(x_enc, x_dec, A_q, A_h, initial_input, teacher_forcing_ratio)
        return dec_out
