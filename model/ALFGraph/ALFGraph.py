import torch
from torch import nn

from layers.DCRNNCell import DCRNNCell
from layers.gnn import calculate_random_walk_matrix
from layers.source_grin import GRINInputImputer


def _build_recurrent_cell(cell_type, input_size, hidden_size, k):
    cell_type = str(cell_type).lower()
    if cell_type not in {"dcrnn", "dcgru", "graph_gru"}:
        raise ValueError(
            f"ALFGraph only retains the final DCRNN cell, got cell_type={cell_type}."
        )
    return DCRNNCell(input_size, hidden_size, k)


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k=2, cell_type="dcrnn"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = nn.ModuleList([
            _build_recurrent_cell(
                cell_type,
                input_size if i == 0 else hidden_size,
                hidden_size,
                k,
            )
            for i in range(num_layers)
        ])

    def forward(self, x, A_q, A_h, h0=None):
        batch_size, seq_len, num_nodes, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers,
                batch_size,
                num_nodes,
                self.hidden_size,
                device=x.device,
            )

        hidden_state = h0
        for t in range(seq_len):
            states = []
            layer_input = x[:, t, :, :]
            for layer_idx in range(self.num_layers):
                next_hidden = self.rnn_cells[layer_idx](
                    layer_input,
                    hidden_state[layer_idx],
                    A_q,
                    A_h,
                )
                layer_input = next_hidden
                states.append(next_hidden)
            hidden_state = torch.stack(states, dim=0)
        return hidden_state


class Decoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        out_seq_len,
        output_size=None,
        num_layers=1,
        k=2,
        cell_type="dcrnn",
        output_mode="direct",
        residual_scale=1.0,
    ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = input_size if output_size is None else output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_seq_len = out_seq_len
        self.output_mode = str(output_mode).lower()
        self.residual_scale = float(residual_scale)
        self.rnn_cells = nn.ModuleList([
            _build_recurrent_cell(
                cell_type,
                input_size if i == 0 else hidden_size,
                hidden_size,
                k,
            )
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(
        self,
        initial_input,
        h,
        A_q,
        A_h,
        teacher_input,
        teacher_forcing_ratio=0.5,
    ):
        input_at_t = initial_input
        outputs = [None for _ in range(self.out_seq_len)]
        for t in range(self.out_seq_len):
            decoder_input = input_at_t
            states = []
            for layer_idx in range(self.num_layers):
                next_hidden = self.rnn_cells[layer_idx](
                    input_at_t,
                    h[layer_idx],
                    A_q,
                    A_h,
                )
                input_at_t = next_hidden
                states.append(next_hidden)
            h = torch.stack(states, dim=0)
            raw_output = self.fc(next_hidden)
            if self.output_mode in {"residual_delta", "delta", "residual"}:
                output = (
                    decoder_input[..., :self.output_size]
                    + self.residual_scale * raw_output
                )
            else:
                output = raw_output
            outputs[t] = output
            use_teacher_forcing = (
                torch.rand(1).item() < teacher_forcing_ratio and self.training
            )
            input_at_t = teacher_input[:, t] if use_teacher_forcing else output
        return torch.stack(outputs, dim=1)


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
        encoder_cell_type = (
            cell_type if encoder_cell_type is None else encoder_cell_type
        )
        decoder_cell_type = (
            cell_type if decoder_cell_type is None else decoder_cell_type
        )
        self.encoder = Encoder(
            enc_input_size,
            hidden_size,
            num_layers,
            k,
            cell_type=encoder_cell_type,
        )
        self.decoder = Decoder(
            dec_input_size,
            hidden_size,
            out_seq_len,
            output_size=dec_input_size,
            num_layers=num_layers,
            k=k,
            cell_type=decoder_cell_type,
            output_mode=decoder_output_mode,
            residual_scale=decoder_residual_scale,
        )

    def forward(
        self,
        x_enc,
        x_dec,
        A_q,
        A_h,
        initial_input,
        teacher_forcing_ratio=0.5,
    ):
        hidden = self.encoder(x_enc, A_q, A_h)
        return self.decoder(
            initial_input,
            hidden,
            A_q,
            A_h,
            x_dec,
            teacher_forcing_ratio,
        )


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.out_len = configs.pred_len if configs.pred_len > 0 else configs.look_back
        self.output_dim = configs.c_out
        recipe_cfg = (getattr(configs, "model_family_cfg", {}) or {}).get(
            "recipe", {}
        )
        self.use_grin_input_imputer = bool(
            recipe_cfg.get("use_grin_input_imputer", False)
        )

        # Keep these names stable so existing rf_alf_sd checkpoints load unchanged.
        self.data_pre = (
            GRINInputImputer(configs, recipe_cfg)
            if self.use_grin_input_imputer
            else None
        )
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.lifecycle_observation_mask_feature = bool(
            getattr(configs, "lifecycle_observation_mask_feature", False)
        )
        backbone_enc_in = int(configs.enc_in) - int(
            self.lifecycle_observation_mask_feature
        )
        self.current_node_ids = None
        cell_type = recipe_cfg.get(
            "cell_type", recipe_cfg.get("mogernn_cell_type", "dcrnn")
        )
        encoder_cell_type = recipe_cfg.get(
            "encoder_cell_type", recipe_cfg.get("mogernn_encoder_cell_type", None)
        )
        decoder_cell_type = recipe_cfg.get(
            "decoder_cell_type", recipe_cfg.get("mogernn_decoder_cell_type", None)
        )
        self.seq2seq = Seq2Seq(
            backbone_enc_in,
            configs.c_out,
            configs.d_model,
            configs.e_layers,
            self.out_len,
            configs.k_hop,
            cell_type=cell_type,
            encoder_cell_type=encoder_cell_type,
            decoder_cell_type=decoder_cell_type,
            decoder_output_mode=recipe_cfg.get("decoder_output_mode", "direct"),
            decoder_residual_scale=float(
                recipe_cfg.get("decoder_residual_scale", 1.0)
            ),
        )

    def set_current_node_ids(self, node_ids):
        # Retained for the common expert wrapper; the final graph has fixed support.
        self.current_node_ids = node_ids

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, *args):
        x_dec = args[0]
        A_q = calculate_random_walk_matrix(adj)
        A_h = calculate_random_walk_matrix(adj.T)
        if self.data_pre is not None:
            x_enc = self.data_pre(adj, x_enc, x_t_mark, pos_mark)
            x_enc = self.data_pre.without_observation_mask(x_enc)
        epoch = args[-1]
        teacher_forcing_ratio = 1 - (epoch / 40)
        initial_input = x_enc[:, -1, :, :self.output_dim]
        return self.seq2seq(
            x_enc,
            x_dec,
            A_q,
            A_h,
            initial_input,
            teacher_forcing_ratio,
        )
