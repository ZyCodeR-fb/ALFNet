import torch
from torch import nn
from torch.distributions.normal import Normal


class MovingAvg(nn.Module):
    """Copied from DUET Autoformer_EncDec moving average decomposition."""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Copied from DUET Autoformer_EncDec series_decomp."""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DUETRevIN(nn.Module):
    """Minimal local copy of DUET RevIN for [batch, time, variables] tensors."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError


class DUETDistributionEncoder(nn.Module):
    """Copied from references/DUET/.../distributional_router_encoder.py::encoder."""

    def __init__(self, seq_len, num_experts, hidden_size):
        super(DUETDistributionEncoder, self).__init__()
        self.distribution_fit = nn.Sequential(
            nn.Linear(seq_len, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts, bias=False),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        return self.distribution_fit(mean)


class DUETLinearExtractor(nn.Module):
    """Copied from references/DUET/.../linear_pattern_extractor.py."""

    def __init__(self, seq_len, output_len, enc_in, moving_avg=3, individual=False):
        super(DUETLinearExtractor, self).__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(output_len)
        self.decompsition = SeriesDecomp(int(moving_avg))
        self.individual = bool(individual)
        self.channels = int(enc_in)
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.channels), device=x_enc.device, dtype=x_enc.dtype)
        return self.encoder(x_enc)[:, -self.pred_len :, :]


class SparseDispatcher(object):
    """Copied from references/DUET/.../linear_extractor_cluster.py."""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device, dtype=stitched.dtype)
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class DUETLinearExtractorCluster(nn.Module):
    """Copied from references/DUET/.../linear_extractor_cluster.py with config removed."""

    def __init__(
        self,
        seq_len,
        enc_in,
        num_experts=2,
        k=1,
        hidden_size=64,
        moving_avg=3,
        noisy_gating=True,
    ):
        super(DUETLinearExtractorCluster, self).__init__()
        self.noisy_gating = bool(noisy_gating)
        self.num_experts = int(num_experts)
        self.input_size = int(seq_len)
        self.k = int(k)
        self.experts = nn.ModuleList([
            DUETLinearExtractor(seq_len=seq_len, output_len=seq_len, enc_in=enc_in, moving_avg=moving_avg)
            for _ in range(self.num_experts)
        ])
        self.W_h = nn.Parameter(torch.eye(self.num_experts))
        self.gate = DUETDistributionEncoder(seq_len, self.num_experts, hidden_size)
        self.noise = DUETDistributionEncoder(seq_len, self.num_experts, hidden_size)
        self.n_vars = int(enc_in)
        self.revin = DUETRevIN(self.n_vars)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        if self.k > self.num_experts:
            raise ValueError("DUET k must be <= num_experts")

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            noisy_logits = clean_logits
            noise_stddev = None
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        x_norm = self.revin(x, "norm")
        expert_inputs = dispatcher.dispatch(x_norm)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class DUETPatchRouter(nn.Module):
    """PatchSTG adapter around copied DUET temporal distribution router."""

    def __init__(
        self,
        hidden_dim,
        patch_num,
        num_experts=2,
        k=1,
        router_hidden_size=64,
        moving_avg=3,
        residual_scale=0.10,
        noisy_gating=True,
    ):
        super().__init__()
        self.patch_num = int(patch_num)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.cluster = DUETLinearExtractorCluster(
            seq_len=self.patch_num,
            enc_in=self.hidden_dim,
            num_experts=int(num_experts),
            k=int(k),
            hidden_size=int(router_hidden_size),
            moving_avg=int(moving_avg),
            noisy_gating=bool(noisy_gating),
        )
        self.last_importance_loss = None

    def forward(self, x):
        B, T, P, N, D = x.shape
        series = x.permute(0, 1, 3, 2, 4).reshape(B * T * N, P, D)
        routed, loss = self.cluster(series)
        routed = self.cluster.revin(routed, "denorm")
        self.last_importance_loss = loss
        routed = routed.reshape(B, T, N, P, D).permute(0, 1, 3, 2, 4).contiguous()
        return x + self.residual_scale * (routed - x)
