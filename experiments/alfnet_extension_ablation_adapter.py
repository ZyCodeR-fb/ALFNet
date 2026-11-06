"""Study-only whole-mechanism interventions (2026-11-03). Production ALFNet unchanged.

Two keys extend experiments/alfnet_ablation_adapter.py, which is left byte for
byte as it was so that earlier evidence keeps its code identity:

``no_sahc``  -- w/o SAHC. The graph-recurrent completion is replaced by the
    observed-only fill (unavailable positions = 0 = the train-K mean, as in the
    existing ``no_imputer``), and the run's config removes window interpolation
    and reference centering (``support_transfer_variant: delta1`` keeps only the
    anchor's decoder residual scale on PEMS04/PEMS08; METR-LA/PEMS-BAY never had
    either). Lifecycle context S is still provided; the objective is unchanged.

``no_os``    -- w/o OS conditioning. No learnable computation receives the
    observation role, the state progress or the mask:
    * SAHC: the mask and the role/progress covariates reaching the GRIN spatial
      decoders, the first GRU cells and the bidirectional merge MLP are zeroed;
      the true mask still decides which values are kept as observations (the
      ``torch.where`` selections and the final fill), and the non-learned window
      interpolation / reference centering still locate observed readings;
    * the six OS channels handed to ALFTemporal and to the ALFGraph encoder are
      zeroed after completion (a zero input is exactly a removed input column);
    * ALFGraph routing is the input-independent global mixture (the config sets
      ``moge_router_mode: static``, identical to ``static_router``);
    * output refinement uses the role-independent constant multiplier
      ``mean(role_scale)`` (the ``mask_only`` convention, 0.5625 here).
    The mask's remaining uses are non-learned: keeping observations during
    completion, and the loss/evaluation/teacher-forcing target masks. The
    training objective, including the role-balanced term, is unchanged.

Construct the complete model first (paired initial values), then apply.
"""
import torch

from experiments.alfnet_ablation_adapter import KEYS as BASE_KEYS
from experiments.alfnet_ablation_adapter import ObservedOnly
from experiments.alfnet_ablation_adapter import apply_intervention as apply_base_intervention

KEYS = ('no_sahc', 'no_os')
AVAILABILITY_COUNT = 4


def _zero_from(start):
    def hook(module, args):
        x = args[0]
        x = torch.cat([x[:, :start], torch.zeros_like(x[:, start:])], dim=1)
        return (x,) + tuple(args[1:])
    return hook


def _blind_spatial_decoder(module, args, kwargs):
    # SpatialDecoder.forward(x, m, h, u, adj, cached_support): m and u are learned inputs.
    kwargs = dict(kwargs)
    kwargs['m'] = torch.zeros_like(kwargs['m'])
    if kwargs.get('u') is not None:
        kwargs['u'] = torch.zeros_like(kwargs['u'])
    return args, kwargs


def blind_grin(imputer, raw_dim):
    """Zero every learned use of mask/role/progress inside the shared BiGRIL."""
    bigril = imputer.bigril
    handles = []
    for rnn in (bigril.fwd_rnn, bigril.bwd_rnn):
        handles.append(rnn.spatial_decoder.register_forward_pre_hook(
            _blind_spatial_decoder, with_kwargs=True))
        if len(rnn.cells) != 1:
            raise ValueError('no_os expects the final one-layer GRIL')
        # update_state input: cat([x_s, m_s, u_s]); keep x_s only.
        handles.append(rnn.cells[0].register_forward_pre_hook(_zero_from(raw_dim)))
    if not isinstance(bigril.out, torch.nn.Sequential) or bigril.emb is not None:
        raise ValueError('no_os expects the final mlp merge without node embedding')
    hidden = bigril.fwd_rnn.hidden_size
    # merge input: cat([fwd_repr (2h), bwd_repr (2h), mask]); keep the representations.
    handles.append(bigril.out.register_forward_pre_hook(_zero_from(4 * hidden)))
    return handles


def zero_os_after_completion(raw_dim):
    def hook(module, args, filled):
        return torch.cat([filled[..., :raw_dim], torch.zeros_like(filled[..., raw_dim:])], dim=-1)
    return hook


def constant_refiner_scale(module, args):
    prediction, adjacency, last_availability = args
    return prediction, adjacency, torch.full_like(last_availability, 1.0 / AVAILABILITY_COUNT)


def apply_intervention(session):
    recipe = session.args.model_family_cfg.get('recipe', {})
    key = recipe.get('study_ablation', 'reference')
    if key in BASE_KEYS or key == 'reference':
        return apply_base_intervention(session)
    assert session.model_name == 'ALFNet' and key in KEYS, key
    model = session.model
    assert not hasattr(model, 'study_ablation'), 'intervention must only be applied once'
    model.study_ablation = key
    if key == 'no_sahc':
        variant = getattr(model, 'support_transfer_variant', None)
        assert variant in (None, 'delta1') and not hasattr(model, 'unified_variant'), \
            'w/o SAHC removes interpolation and centering'
        model.shared_input_imputer = ObservedOnly(model.raw_enc_in, model.mask_index)
    elif key == 'no_os':
        assert model.observation_state_mode == 'full'
        assert model.graph_expert.router_mode == 'static', 'no_os requires the static router config'
        model._extension_hooks = blind_grin(model.shared_input_imputer, model.raw_enc_in)
        model._extension_hooks.append(model.shared_input_imputer.register_forward_hook(
            zero_os_after_completion(model.raw_enc_in)))
        model._extension_hooks.append(model.graph_refiner.register_forward_pre_hook(
            constant_refiner_scale))
    return session
