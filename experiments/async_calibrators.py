import math

import torch
import torch.nn as nn


class FRPlusModule(nn.Module):
    """Amplitude/phase calibration module copied from ST-TTC."""

    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = freq_bins // groups
        self.lambda_amp = nn.Parameter(torch.zeros(groups, 1, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, 1, 1))

    def forward(self, y_pred):
        # y_pred: [B,1,N,T]
        _, _, _, T = y_pred.shape
        y = y_pred[:, 0]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)

        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T // 2 + 1 if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ADCSDModule(nn.Module):
    def __init__(self, output_dim, output_window, num_nodes, moving_avg=5):
        super(ADCSDModule, self).__init__()
        self.decomp = series_decomp(moving_avg)
        hidden_ratio = 128
        FWL_list_1 = [
            nn.Linear(output_dim, hidden_ratio),
            nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
            nn.GELU(),
            nn.Linear(hidden_ratio, output_dim),
        ]
        self.FWL_1 = nn.Sequential(*FWL_list_1)
        self.learned_lambda_1 = nn.Parameter(torch.zeros(num_nodes, 1))
        FWL_list_2 = [
            nn.Linear(output_dim, hidden_ratio),
            nn.LayerNorm([output_window, num_nodes, hidden_ratio]),
            nn.GELU(),
            nn.Linear(hidden_ratio, output_dim),
        ]
        self.FWL_2 = nn.Sequential(*FWL_list_2)
        self.learned_lambda_2 = nn.Parameter(torch.zeros(num_nodes, 1))

    def forward(self, x):
        output_1, output_2 = self.decomp(x)
        output = x + self.learned_lambda_1 * self.FWL_1(output_1) + self.learned_lambda_2 * self.FWL_2(output_2)
        return output


class SimpleOutputAdapter(nn.Module):
    def __init__(self, pred_len: int, buffer_context_size: int = 5, n_vars: int = 1, var_wise_gating: bool = False, num_layers: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.pred_len = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len

        if self.var_wise:
            if self.num_layers == 1:
                self.fc_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(n_vars)])
            else:
                self.fc_layers = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim, output_dim),
                        )
                        for _ in range(n_vars)
                    ]
                )
            self.gate = nn.Parameter(torch.zeros(n_vars))
        else:
            if self.num_layers == 1:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim),
                )
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.var_wise:
            for fc_module in self.fc_layers:
                if self.num_layers == 1:
                    nn.init.xavier_uniform_(fc_module.weight, gain=0.1)
                    nn.init.zeros_(fc_module.bias)
                else:
                    for layer in fc_module:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.1)
                            nn.init.zeros_(layer.bias)
        else:
            if self.num_layers == 1:
                nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
                nn.init.zeros_(self.fc.bias)
            else:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y

        batch_size, pred_len, n_vars = y.shape

        if self.var_wise:
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]
                combined_input = torch.cat([y_var, context_data], dim=-1)
                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))

            correction = torch.cat(corrections, dim=-1)
            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)

        return y + gating_factor * correction


class CorrCoefLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds.reshape(-1)
        y = target.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        std_x = x.std(unbiased=False)
        std_y = y.std(unbiased=False)
        if std_x <= self.eps or std_y <= self.eps:
            return preds.new_tensor(0.0)
        corr_xy = torch.mean((x / (std_x + self.eps)) * (y / (std_y + self.eps)))
        return -corr_xy


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise

        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        weight = torch.einsum("ik,kjl->ijl", self.lora_A, self.lora_B)
        x_1 = torch.tanh(self.gating * x)
        if self.var_wise:
            new_x = torch.einsum("biv,iov->bov", x_1, weight) + self.bias
        else:
            new_x = torch.einsum("biv,io->bov", x_1, weight) + self.bias

        x = x + new_x
        return x


class Calibration(nn.Module):
    def __init__(self, seq_len, pred_len, n_var, hidden_dim=128, gating_init=0.01, var_wise=True, low_rank=16):
        super(Calibration, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_var = n_var
        self.hidden_dim = hidden_dim
        self.gating_init = gating_init
        self.var_wise = var_wise
        self.low_rank = low_rank

        self.in_cali = GCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)
        self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, self.low_rank)

    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = inputs
        batch_size, seq_len, num_nodes, channels = enc_window.shape
        enc_window = enc_window.reshape(batch_size, seq_len, num_nodes * channels)
        enc_window = self.in_cali(enc_window)
        enc_window = enc_window.reshape(batch_size, seq_len, num_nodes, channels)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs):
        batch_size, pred_len, num_nodes, channels = outputs.shape
        outputs = outputs.reshape(batch_size, pred_len, num_nodes * channels)
        outputs = self.out_cali(outputs)
        return outputs.reshape(batch_size, pred_len, num_nodes, channels)
