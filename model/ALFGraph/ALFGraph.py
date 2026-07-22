import torch
from torch import nn

from layers.DCRNNCell import DCRNNCell
from layers.gnn import calculate_random_walk_matrix


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.cells = nn.ModuleList(
            [
                DCRNNCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    k,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, sequence, forward_support, backward_support):
        batch, steps, nodes, _ = sequence.shape
        hidden = torch.zeros(
            self.num_layers,
            batch,
            nodes,
            self.hidden_size,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        for step in range(steps):
            layer_input = sequence[:, step]
            next_states = []
            for layer, cell in enumerate(self.cells):
                layer_input = cell(
                    layer_input,
                    hidden[layer],
                    forward_support,
                    backward_support,
                )
                next_states.append(layer_input)
            hidden = torch.stack(next_states, dim=0)
        return hidden


class ResidualDeltaDecoder(nn.Module):

    def __init__(self, output_size, hidden_size, num_layers, out_steps, k, scale):
        super().__init__()
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.out_steps = int(out_steps)
        self.scale = float(scale)
        self.cells = nn.ModuleList(
            [
                DCRNNCell(
                    output_size if layer == 0 else hidden_size,
                    hidden_size,
                    k,
                )
                for layer in range(num_layers)
            ]
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        initial_input,
        hidden,
        forward_support,
        backward_support,
        teacher_input,
        teacher_forcing_ratio,
    ):
        current = initial_input
        outputs = []
        for step in range(self.out_steps):
            anchor = current
            next_states = []
            for layer, cell in enumerate(self.cells):
                current = cell(
                    current,
                    hidden[layer],
                    forward_support,
                    backward_support,
                )
                next_states.append(current)
            hidden = torch.stack(next_states, dim=0)
            output = anchor[..., :self.output_size] + self.scale * self.projection(current)
            outputs.append(output)
            use_teacher = self.training and torch.rand(1).item() < teacher_forcing_ratio
            current = teacher_input[:, step] if use_teacher else output
        return torch.stack(outputs, dim=1)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        if not configs.lifecycle_observation_mask_feature:
            raise ValueError("ALFGraph requires the explicit observation-mask channel")
        recipe = (getattr(configs, "model_family_cfg", {}) or {}).get("recipe", {})
        if bool(recipe.get("use_grin_input_imputer", False)):
            raise ValueError("ALFNet uses one shared GRIN, not an ALFGraph-local GRIN")
        output_steps = configs.pred_len if configs.pred_len > 0 else configs.look_back
        backbone_input = int(configs.enc_in) - 1
        self.output_dim = int(configs.c_out)
        self.encoder = Encoder(
            backbone_input,
            configs.d_model,
            configs.e_layers,
            configs.k_hop,
        )
        self.decoder = ResidualDeltaDecoder(
            configs.c_out,
            configs.d_model,
            configs.e_layers,
            output_steps,
            configs.k_hop,
            recipe.get("decoder_residual_scale", 0.3),
        )

    def set_current_node_ids(self, node_ids):
        return None

    def forward(
        self,
        adjacency,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
    ):
        forward_support = calculate_random_walk_matrix(adjacency)
        backward_support = calculate_random_walk_matrix(adjacency.T)
        backbone_input = x_enc[..., :-1]
        hidden = self.encoder(backbone_input, forward_support, backward_support)
        initial_input = backbone_input[:, -1, :, :self.output_dim]
        teacher_forcing_ratio = 1.0 - float(epoch) / 40.0
        return self.decoder(
            initial_input,
            hidden,
            forward_support,
            backward_support,
            x_dec,
            teacher_forcing_ratio,
        )
