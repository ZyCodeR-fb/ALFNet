from .layers.gnn import nn, torch, calculate_random_walk_matrix
import torch.nn.functional as F
from einops import rearrange
from .layers.agg import MeanAggregator, WMeanAggregator, MaxPoolingAggregator, MinPoolingAggregator, DiffusionAggregator
from .layers.DCRNNCell import DCRNNCell


class Expert(nn.Module):

    def __init__(self, completion: nn.Module):
        super(Expert, self).__init__()
        self.completion = completion

    def forward(self, adj, x_enc, batch_unknown_nodes):
        return self.completion(adj, x_enc, batch_unknown_nodes)


class GatingNetwork(nn.Module):

    def __init__(self, configs):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(configs.input_len, configs.d_model)
        self.fc2 = nn.Linear(configs.d_model, configs.num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SparseGatingNetwork(nn.Module):

    def __init__(self, configs):
        super(SparseGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(configs.input_len, configs.d_model)
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
        experts = []
        if configs.mean_expert:
            experts.append(MeanAggregator(configs.input_len, configs.input_len))
        if configs.weight_expert:
            experts.append(WMeanAggregator(configs.input_len, configs.input_len))
        if configs.max_expert:
            experts.append(MaxPoolingAggregator(configs.input_len, configs.input_len))
        if configs.min_expert:
            experts.append(MinPoolingAggregator(configs.input_len, configs.input_len))
        if configs.diffusion_expert:
            experts.append(DiffusionAggregator(configs.input_len, configs.input_len))
        self.experts = nn.ModuleList([Expert(expert) for expert in experts])
        self.gating = SparseGatingNetwork(configs)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None):
        """Complete the unobserved sensors, one sample at a time.

        The source detects unknown nodes with ``sum(x_enc, dim=(0,1)) == 0``,
        i.e. a node counts as unknown only if it is zero across the *whole
        batch*. That is exactly right in the setting MoGERNN was written for,
        where one fixed sensor set is unobserved for the entire dataset.

        LCBT hides a different cohort in every sample, so the batch-wide
        intersection is usually empty: measured on METR-LA with disjoint
        cohorts, B=1 finds 8 unknown nodes and B>=4 finds 0, which switches the
        aggregator completion off for the whole of training and validation while
        leaving it on at test time (the async evaluator runs at batch size 1).

        Running the original rule per sample restores the source semantics under
        this protocol: at B=1 it is bit-identical to the reference, and a mixed
        batch now equals the concatenation of its samples. Nothing about the
        model changes -- no lifecycle feature and no ALFNet module is consulted,
        only the unknown-node set that the reference already computed.
        """
        x_enc = x_enc.squeeze(-1)
        x_enc = rearrange(x_enc, "b t n -> b n t")
        gate_weights = self.gating(x_enc)
        adj_noselfloop = adj.clone()
        adj_noselfloop[torch.eye(adj.shape[0], device=adj.device).bool()] = 0

        completed = []
        for index in range(x_enc.shape[0]):
            sample = x_enc[index:index + 1]
            # `sum(dim=(0,1))` over a single-sample slice is the reference rule
            # applied to that sample alone.
            unknown_nodes = torch.where(
                torch.sum(sample, dim=(0, 2)) == 0
            )[0].cpu().tolist()
            outputs = [
                expert(adj_noselfloop, sample, unknown_nodes) for expert in self.experts
            ]
            stacked = torch.stack(outputs, dim=2)
            completed.append(
                torch.sum(gate_weights[index:index + 1].unsqueeze(-1) * stacked, dim=2)
            )
        combied_out = torch.cat(completed, dim=0)

        miss_ix = (x_enc == 0).int()
        x_enc = x_enc * (1 - miss_ix) + combied_out * miss_ix
        x_enc = rearrange(x_enc, "b n t -> b t n 1")
        return x_enc


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = nn.ModuleList([DCRNNCell(input_size if i == 0 else hidden_size, hidden_size, k) for i in range(num_layers)])

    def forward(self, x, A_q, A_h, h0=None):
        # x: [batch_size, seq_len, num_nodes, input_size]
        batch_size, seq_len, num_nodes, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, num_nodes, self.hidden_size, device=x.device)

        hidden_state = h0
        for t in range(seq_len):
            hs = []
            input = x[:, t, :, :]
            for l in range(self.num_layers):
                next_hidden = self.rnn_cells[l](input, hidden_state[l], A_q, A_h)
                input = next_hidden
                hs.append(next_hidden)
            hidden_state = torch.stack(hs, dim=0)
        return hidden_state


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, out_seq_len, num_layers=1, k=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len

        self.rnn_cells = nn.ModuleList([DCRNNCell(input_size if i == 0 else hidden_size, hidden_size, k) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, intial_input, h, A_q, A_h, tearcher_input, teacher_forcing_ratio=0.5):
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
            use_teacher_forcing = True if (torch.rand(1).item() < teacher_forcing_ratio and self.training) else False
            if use_teacher_forcing:
                input_at_t = tearcher_input[:, t]
            else:
                input_at_t = outputs[t]
        outputs = torch.stack(outputs, dim=1)  # type: ignore
        return outputs


class Seq2Seq(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, out_seq_len, k=2):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, k)
        self.decoder = Decoder(input_size, hidden_size, out_seq_len, num_layers, k)

    def forward(self, x_enc, x_dec, A_q, A_h, teacher_forcing_ratio=0.5):
        h = self.encoder(x_enc, A_q, A_h)
        dec_out = self.decoder(x_enc[:, -1], h, A_q, A_h, x_dec, teacher_forcing_ratio)
        return dec_out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.out_len = configs.pred_len if configs.pred_len > 0 else configs.look_back
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.data_pre = DataPreparation(configs)
        self.seq2seq = Seq2Seq(self.raw_enc_in, configs.d_model, configs.e_layers, self.out_len, configs.k_hop)

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, *args):
        x_dec = args[0]
        # 输入的adj应该转一下
        A_q = calculate_random_walk_matrix(adj)
        A_h = calculate_random_walk_matrix(adj.T)
        x_enc = self.data_pre(
            adj,
            x_enc[..., :self.raw_enc_in],
            x_t_mark,
            pos_mark,
        )
        epoch = args[-1]
        teacher_forcing_ratio = 1 - (epoch / 40)
        dec_out = self.seq2seq(x_enc, x_dec, A_q, A_h, teacher_forcing_ratio)
        return dec_out
