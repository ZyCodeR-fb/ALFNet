"""Study-only interventions. The production ALFNet implementation is unchanged.

Construct the complete model first so retained tensors keep their paired initial
values. Apply this adapter before optimizer construction, including when restoring
a study checkpoint. Never apply it to comparison algorithms.
"""
import torch
from torch import nn
from experiments.v4_runner import build_v4_session as build_base_session

KEYS = ('mask_only', 'no_age', 'static_router', 'local_context',
        'no_imputer', 'no_temporal_film', 'no_refiner', 'plain_mae', 'no_role_balance')


class ObservedOnly(nn.Module):
    def __init__(self, raw_dim, mask_index):
        super().__init__()
        self.raw_dim, self.mask_index = raw_dim, mask_index

    def forward(self, adjacency, x, *unused):
        observed = x[..., self.mask_index:self.mask_index + 1] > .5
        raw = torch.where(observed, x[..., :self.raw_dim], torch.zeros_like(x[..., :self.raw_dim]))
        return torch.cat((raw, x[..., self.raw_dim:]), dim=-1)


class ZeroTemporal(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward_v4(self, x, marks):
        return {'features': x.new_zeros(x.shape[0], x.shape[2], self.feature_dim)}


class ZeroFiLM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, features):
        return features.new_zeros(*features.shape[:-1], 2 * self.hidden_dim)


class IdentityRefiner(nn.Module):
    def forward(self, prediction, adjacency, last_availability):
        return prediction


def remove_age(module, args):
    # Only ALFNet's assembled method input changes. Batch roles, target masks,
    # teacher forcing and role-balanced supervision retain their true labels.
    args = list(args)
    x = args[1].clone()
    x[..., module.age_index] = x[..., module.mask_index]
    args[1] = x
    return tuple(args)


def apply_intervention(session):
    recipe = session.args.model_family_cfg.get('recipe', {})
    key = recipe.get('study_ablation', 'reference')
    if key == 'reference':
        return session
    assert session.model_name == 'ALFNet' and key in KEYS
    model = session.model
    assert not hasattr(model, 'study_ablation'), 'intervention must only be applied once'
    model.study_ablation = key
    if key == 'no_age':
        model.register_forward_pre_hook(remove_age)
    elif key == 'no_imputer':
        model.shared_input_imputer = ObservedOnly(model.raw_enc_in, model.mask_index)
    elif key == 'no_temporal_film':
        model.patch_expert = ZeroTemporal(model.patch_expert.feature_dim)
        model.film_head = ZeroFiLM(model.graph_args.d_model)
    elif key == 'no_refiner':
        model.graph_refiner = IdentityRefiner()
    elif key == 'mask_only':
        assert model.observation_state_mode == 'mask_only'
    elif key == 'static_router':
        assert model.graph_expert.router_mode == 'static'
    elif key == 'local_context':
        assert model.patch_expert.local_context_only
    elif key == 'plain_mae':
        assert session.args.loss == 'mae'
    elif key == 'no_role_balance':
        assert model.forecast_role_balanced_beta == 0
    return session


def build_study_session(*args, **kwargs):
    return apply_intervention(build_base_session(*args, **kwargs))


def install_training_adapter():
    import scripts.train_v4 as runner
    runner.build_v4_session = build_study_session
