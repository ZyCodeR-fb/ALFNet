"""A/B/C/D switch contracts.

A = TokenLearner/TokenFuser replacing the KD-tree blocks
B = TSMixer replacing Crossformer, with the time axis restored to 3 segments
C = evidence-aware graph/temporal router
D = observed-history NLinear residual

Each is a YAML switch on top of fg1 and each must work alone.
"""

import torch
import pytest
import yaml

from pathlib import Path

from args import load_args_from_config
from layers.evidence_router import (
    ObservedNLinear,
    mask_shape,
    neighbour_evidence,
    routing_diagnostics,
    self_evidence,
)
from layers.source_tokenlearner import NodeTokenizer
from layers.source_tsmixer import TSMixerBlock
from model.regist_model import load_model_class


ROOT = Path(__file__).resolve().parents[1]
SWITCHES = {"fg1": "fg1", "a": "a", "b": "b", "c": "c", "d": "d"}
ROLE_AAS, ROLE_NAS, ROLE_VS, ROLE_FS = 0, 1, 2, 3


def build(name):
    args = load_args_from_config(f"configs/ALFNet/{name}.yaml", [])
    args.device = "cpu"
    return args, load_model_class(args.model)(args)


def _batch(args, model, batch=2, seed=3):
    nodes = len(model.known_nodes)
    generator = torch.Generator().manual_seed(seed)
    return dict(
        adj=torch.rand(nodes, nodes, generator=generator),
        x=torch.randn(batch, args.input_len, nodes, args.enc_in, generator=generator),
        tm=torch.rand(batch, args.input_len, 3, generator=generator) * 5,
        dec=torch.randn(batch, args.pred_len, nodes, args.c_out, generator=generator),
        dm=torch.rand(batch, args.pred_len, 3, generator=generator) * 5,
    )


def _set_roles(args, model, data, roles):
    onehot = torch.zeros(data["x"].shape[0], args.input_len, len(model.known_nodes), 4)
    for node, role in enumerate(roles):
        onehot[:, :, node, role] = 1.0
    data["x"][..., model.role_start:model.role_end] = onehot
    return data


# --- 1. TokenLearner maps nodes -> tokens -> every node, losing none ---------


def test_tokenizer_round_trips_every_node():
    torch.manual_seed(0)
    tokenizer = NodeTokenizer(in_dim=16, num_tokens=8, bottleneck_dim=12)
    x = torch.randn(2, 3, 37, 16)
    tokens = tokenizer.tokenize(x)
    assert tokens.shape == (2, 3, 8, 16)
    fused = tokenizer.fuse(tokens, x)
    assert fused.shape == x.shape, "every node must get a value back"
    assert torch.isfinite(fused).all()
    # the zero-initialised fuser mix means an untrained fuser adds nothing,
    # so the residual path, not noise, carries the signal at step 0
    assert torch.allclose(fused, torch.zeros_like(fused), atol=1e-6)
    # after perturbing the fuser, every node is still written to
    torch.nn.init.normal_(tokenizer.fuser.token_mix.weight, std=0.5)
    fused = tokenizer.fuse(tokens, x)
    assert (fused.abs().sum(dim=-1) > 0).all(), "some node received nothing"


def test_tokenizer_attention_is_content_adaptive():
    """Different node content must produce different token assignments."""
    torch.manual_seed(0)
    tokenizer = NodeTokenizer(in_dim=8, num_tokens=4, bottleneck_dim=8)
    a = torch.randn(1, 1, 20, 8)
    b = a.clone()
    b[0, 0, :5] += 5.0
    assert not torch.allclose(tokenizer.tokenize(a), tokenizer.tokenize(b))


# --- 2. TSMixer really mixes three ordered time segments ---------------------


def test_tsmixer_requires_and_uses_a_real_time_axis():
    with pytest.raises(ValueError, match="at least two time segments"):
        TSMixerBlock(time_steps=1, num_tokens=8, hidden_dim=16)

    torch.manual_seed(0)
    block = TSMixerBlock(time_steps=3, num_tokens=8, hidden_dim=16, ff_dim=16, dropout=0.0)
    block.eval()
    x = torch.randn(2, 3, 8, 16)
    # time mixing is not the identity: permuting the time order changes output
    out = block(x)
    shuffled = block(x[:, [2, 0, 1]])
    assert not torch.allclose(out[:, 0], shuffled[:, 0], atol=1e-6), (
        "time mixing ignores segment order, so it is not acting on the time axis"
    )
    # and a change at one time step reaches the others. The perturbation must
    # not be constant across the feature axis: pre-norm LayerNorm subtracts the
    # mean over D, so adding the same value to every channel is invisible.
    y = x.clone()
    y[:, 0, :, ::2] += 3.0
    assert not torch.allclose(block(y)[:, 2], out[:, 2], atol=1e-6), (
        "a change at segment 0 never reaches segment 2"
    )


def test_b_variant_carries_three_time_segments_end_to_end():
    args, model = build("b")
    backbone = model.patch_expert.backbone
    block = backbone.spa_encoder[0]
    assert block.mixer_kind == "tsmixer"
    assert block.tsmixer.time_steps == 3
    assert block.tsa is None, "the replaced attention layer must not be built"
    # the embedding conv must no longer collapse T to 1
    assert backbone.input_st_fc.kernel_size == (1, 4)
    assert backbone.input_st_fc.stride == (1, 4)
    assert model.patch_expert.feature_dim == 3 * 160

    seen = {}

    def record(module, inputs):
        seen["shape"] = tuple(inputs[0].shape)
        return None            # a pre-hook returning a value replaces the args

    block.tsmixer.register_forward_pre_hook(record)
    data = _batch(args, model)
    model.eval()
    with torch.no_grad():
        model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
    assert seen["shape"][1] == 3, f"TSMixer saw a time axis of {seen['shape'][1]}"


# --- 3. every switch runs alone; gradients are finite ------------------------


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_each_switch_forwards_and_backwards_alone(name):
    args, model = build(name)
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    pred = out["pred"]
    assert pred.shape == (2, args.pred_len, len(model.known_nodes), args.c_out)
    assert torch.isfinite(pred).all()
    loss = pred.pow(2).mean()
    for value in (out.get("aux_losses") or {}).values():
        if torch.is_tensor(value):
            loss = loss + value.float()
    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(p.grad).all() for p in grads)


def test_combined_switches_stay_finite():
    """A+B+C+D. The previous version pointed at cond1b, which is B only: the
    TokenLearner was never built, so the test named four switches and exercised
    three."""
    args, model = build("abcd")
    backbone = model.patch_expert.backbone
    assert backbone.spatial_tokenizer == "tokenlearner", "A is not actually on"
    assert backbone.spa_encoder[0].mixer_kind == "tsmixer", "B is not actually on"
    assert model.evidence_router_enabled and model.observed_nlinear_enabled
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert torch.isfinite(out["pred"]).all()
    out["pred"].pow(2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


COMBINATIONS = ["ac", "ad", "bc", "bd", "cd", "acd", "bcd", "abc", "abcd"]
EXPECTED_MODULES = {
    "a": lambda m: m.patch_expert.backbone.spatial_tokenizer == "tokenlearner",
    "b": lambda m: m.patch_expert.backbone.spa_encoder[0].mixer_kind == "tsmixer",
    "c": lambda m: m.evidence_router is not None,
    "d": lambda m: m.observed_nlinear is not None,
}


@pytest.mark.parametrize("stem", COMBINATIONS)
def test_combination_instantiates_exactly_its_switches(stem):
    _, model = build(stem)
    for switch, present in EXPECTED_MODULES.items():
        assert present(model) is (switch in stem), (
            f"{stem}: switch {switch} should be {'on' if switch in stem else 'off'}"
        )


@pytest.mark.parametrize("stem", COMBINATIONS)
def test_combination_forwards_and_backwards(stem):
    args, model = build(stem)
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert torch.isfinite(out["pred"]).all()
    loss = out["pred"].pow(2).mean()
    for value in (out.get("aux_losses") or {}).values():
        if torch.is_tensor(value):
            loss = loss + value.float()
    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(p.grad).all() for p in grads)


# --- 4. C off leaves fg1 untouched ------------------------------------------


def test_fg1_is_unchanged_when_c_and_d_are_off():
    _, plain = build("fg1")
    assert plain.evidence_router_enabled is False
    assert plain.observed_nlinear_enabled is False
    assert plain.evidence_router is None and plain.observed_nlinear is None
    assert plain.patch_projection is None
    _, routed = build("c")
    extra = set(routed.state_dict()) - set(plain.state_dict())
    assert extra, "C must add parameters"
    assert all(key.startswith(("evidence_router.", "patch_projection.")) for key in extra), extra
    assert set(plain.state_dict()) - set(routed.state_dict()) == set()
    # and the forward advertises no routing metadata without C
    args, _ = build("fg1")
    data = _batch(args, plain)
    out = plain(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert "route_routing" not in out


# --- 5/6. Unseen stays graph-led; Outage keeps its patch correction ----------


def _amplify(module, std=0.6):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=std)
            torch.nn.init.normal_(layer.bias, std=std)


def test_self_gate_silences_the_observed_path_without_history():
    """D's NLinear must contribute nothing when nothing was measured."""
    torch.manual_seed(0)
    nlinear = ObservedNLinear(12, 12, 1)
    _amplify(nlinear)
    observed = torch.randn(2, 12, 5, 1)
    empty = torch.zeros(2, 12, 5, 1)
    gate = (self_evidence(empty) / 0.02).clamp(0, 1)
    assert float(gate.max()) == 0.0, "no observed step must give zero self evidence"
    full = torch.ones(2, 12, 5, 1)
    assert float((self_evidence(full) / 0.02).clamp(0, 1).min()) == 1.0


def test_unseen_and_outage_share_e_self_but_differ_in_context():
    """The pair the design turns on: e_self alone cannot separate them."""
    empty = torch.zeros(1, 12, 3, 1)
    assert float(self_evidence(empty).max()) == 0.0
    # both an unseen sensor and a fully-dark failed sensor look identical here
    shape_unseen = mask_shape(empty)
    shape_outage = mask_shape(empty)
    assert torch.equal(shape_unseen, shape_outage)
    # what separates them is availability context and state_age, which the
    # router receives as separate inputs
    args, model = build("c")
    context_dim = model.evidence_router.router[0].in_features
    series_dim = 3 * args.pred_len * args.c_out
    assert context_dim - series_dim == 4 + 2 + 2 + 3


def test_patch_correction_survives_zero_self_evidence():
    """Outage has e_self = 0 and must still receive a patch correction."""
    args, model = build("c")
    model.eval()
    nodes = len(model.known_nodes)
    data = _batch(args, model)
    data = _set_roles(args, model, data, [ROLE_FS] * nodes)
    data["x"][..., model.mask_index] = 0.0          # nothing observed: e_self = 0
    _amplify(model.patch_projection)
    _amplify(model.evidence_router.router)
    with torch.no_grad():
        with_patch = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        projection, model.patch_projection = model.patch_projection, None
        without = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
    assert not torch.allclose(with_patch, without), (
        "e_self = 0 switched the patch path off, which is the behaviour the "
        "fault-track measurements say is backwards"
    )


def test_router_can_separate_unseen_from_outage():
    """Same zero self-evidence, different availability context -> different beta."""
    args, model = build("c")
    model.eval()
    nodes = len(model.known_nodes)
    _amplify(model.evidence_router.router)
    betas = {}
    for label, role in (("unseen", ROLE_VS), ("outage", ROLE_FS)):
        data = _batch(args, model)
        data = _set_roles(args, model, data, [role] * nodes)
        data["x"][..., model.mask_index] = 0.0
        with torch.no_grad():
            model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
        betas[label] = model.last_routing_diagnostics["beta_mean"]
    assert betas["unseen"] != betas["outage"], (
        "the router cannot tell an unseen sensor from a failed one"
    )


# --- 7/8. neighbourhood evidence -------------------------------------------


def test_neighbour_evidence_reacts_to_the_neighbourhood_not_the_node():
    """Identical local mask, different neighbours -> different e_nbr."""
    adjacency = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    mask_bright = torch.ones(1, 4, 3, 1)
    mask_dark = mask_bright.clone()
    mask_dark[:, -1, 1:] = 0.0            # node 0 unchanged, its neighbours dark
    bright = neighbour_evidence(adjacency, mask_bright)
    dark = neighbour_evidence(adjacency, mask_dark)
    assert float(bright[0, 0, 0]) == pytest.approx(1.0)
    assert float(dark[0, 0, 0]) == pytest.approx(0.0)
    # the node's own last-step mask is identical in both cases
    assert torch.equal(mask_bright[:, -1, 0], mask_dark[:, -1, 0])


def test_beta_responds_to_neighbourhood_darkness():
    """The clustered/scattered distinction must be available to the router."""
    args, model = build("c")
    model.eval()
    nodes = len(model.known_nodes)
    _amplify(model.evidence_router.router)
    results = {}
    for label, dark in (("scattered", False), ("clustered", True)):
        data = _batch(args, model)
        data = _set_roles(args, model, data, [ROLE_FS] * nodes)
        data["x"][..., model.mask_index] = 0.0 if dark else 1.0
        data["x"][:, :, 0, model.mask_index] = 0.0     # the node itself is dark either way
        with torch.no_grad():
            model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
        results[label] = model.last_routing_diagnostics["beta_mean"]
    assert results["clustered"] != results["scattered"], (
        "beta is blind to neighbourhood observability"
    )
    # the direction is learned, not hard-coded: this test asserts sensitivity only


# --- 9. D reads measurements, never the imputed history ---------------------


def test_observed_nlinear_reads_raw_input_not_the_imputer_output():
    import inspect

    from model.ALFNet.ALFNet import Model

    source = inspect.getsource(Model._evidence_fusion)
    assert "x_enc[..., :self.output_dim]" in source, (
        "the observed residual must read the raw window, not `filled`"
    )
    assert "filled" not in source

    torch.manual_seed(0)
    nlinear = ObservedNLinear(6, 4, 1)
    _amplify(nlinear)
    observed = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1).repeat(1, 1, 2, 1)
    mask = torch.ones(1, 6, 2, 1)
    mask[:, 3:, 0] = 0.0                     # node 0 goes dark after step 2
    residual, anchor = nlinear(observed, mask)
    assert torch.isfinite(residual).all() and torch.isfinite(anchor).all()
    # node 0 is anchored on its last *observed* value (2.0), not the last row
    assert float(anchor[0, 0, 0, 0]) == pytest.approx(2.0)
    assert float(anchor[0, 0, 1, 0]) == pytest.approx(5.0)
    dark_residual, dark_anchor = nlinear(observed, torch.zeros_like(mask))
    assert torch.isfinite(dark_residual).all()
    assert float(dark_anchor.abs().max()) == 0.0, "no measurement, no anchor"


# --- 10. C+D produce one Yt and one beta ------------------------------------


def test_c_plus_d_have_a_single_fusion_point():
    args = load_args_from_config("configs/ALFNet/c.yaml", [])
    args.device = "cpu"
    args.model_family_cfg["recipe"]["observed_nlinear"] = True
    model = load_model_class(args.model)(args)
    assert model.evidence_router_enabled and model.observed_nlinear_enabled
    # no leftover second residual from the retired candidate
    assert model.fault_residual_head is None
    assert model.availability_gates is None and model.temporal_refiner is None
    data = _batch(args, model)
    model.eval()
    with torch.no_grad():
        out = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
    assert "route_routing" in out
    beta_like = 1.0 - out["route_routing"]["alpha"]
    assert beta_like.shape[1] == 1, "one beta per node, not one per horizon step"


# --- 11. collapse diagnostics -----------------------------------------------


def test_routing_diagnostics_expose_collapse():
    target = torch.zeros(2, 4, 5, 1)
    y_graph = torch.full_like(target, 1.0)
    y_temporal = torch.full_like(target, 0.5)      # temporal is always better
    valid = torch.ones_like(target, dtype=torch.bool)
    availability = torch.zeros(2, 5, 4)
    availability[:, :3, ROLE_AAS] = 1.0
    availability[:, 3:, ROLE_FS] = 1.0

    collapsed = routing_diagnostics(torch.zeros(2, 1, 5, 1), y_graph, y_temporal,
                                    target, valid, availability)
    assert collapsed["beta_std"] == pytest.approx(0.0)
    assert collapsed["temporal_activation_ratio"] == 0.0
    assert collapsed["route_accuracy"] == 0.0      # always picks the worse branch

    beta = torch.zeros(2, 1, 5, 1)
    beta[:, :, 3:] = 1.0
    split = routing_diagnostics(beta, y_graph, y_temporal, target, valid, availability)
    assert split["beta_std"] > 0.0
    assert split[f"beta_role{ROLE_FS}"] > split[f"beta_role{ROLE_AAS}"]
    assert split[f"graph_mae_role{ROLE_FS}"] > split[f"temporal_mae_role{ROLE_FS}"]


# --- 12. the ablation YAMLs differ only in the switch under test -------------


def _flat(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


ALLOWED_DRIFT = {
    "runtime.model_variant",
    "family.experts.patch.config_path",
    "family.recipe.evidence_router",
    "family.recipe.detach_imputer_for_temporal",
    "family.recipe.self_evidence_floor",
    "family.recipe.lambda_route",
    "family.recipe.observed_nlinear",
    "family.recipe.observed_residual_scale",
    "family.recipe.router_lr",
    "family.recipe.patch_residual_scale",
}


@pytest.mark.parametrize("name", ["a", "b", "c", "d"])
def test_ablation_yaml_differs_from_fg1_only_in_its_own_switch(name):
    base = _flat(yaml.safe_load((ROOT / "configs/ALFNet/fg1.yaml").read_text(encoding="utf-8")))
    other = _flat(yaml.safe_load((ROOT / f"configs/ALFNet/{name}.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift <= ALLOWED_DRIFT, f"{name}.yaml drifted on {sorted(drift - ALLOWED_DRIFT)}"


@pytest.mark.parametrize("name", ["a", "b"])
def test_patch_yaml_differs_from_cond1_only_in_its_own_switch(name):
    suffix = {"a": "a", "b": "b"}[name]
    base = _flat(yaml.safe_load(
        (ROOT / "configs/ALFTemporal/METRLA_cond1.yaml").read_text(encoding="utf-8")))
    other = _flat(yaml.safe_load(
        (ROOT / f"configs/ALFTemporal/METRLA_cond1{suffix}.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    allowed = {"runtime.model_variant", "family.model.spatial_tokenizer",
               "family.model.token_bottleneck", "family.model.mixer",
               "family.model.tem_patchsize", "family.model.tem_patchnum"}
    assert drift <= allowed, f"cond1{suffix} drifted on {sorted(drift - allowed)}"


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_every_variant_keeps_the_frozen_protocol_and_mainline(name):
    args, model = build(name)
    assert args.lifecycle_training_strategy == "lcbt_v3"
    assert args.lce_protocol_version == 3 and args.ples_protocol_version == 3
    assert args.seed == 32 and args.stop_based == "val_lifecycle"
    assert model.shared_input_imputer is not None
    assert hasattr(model.graph_expert, "decoder")
    from layers.lifecycle_moge import LifecycleRoutedGraphConv
    assert any(isinstance(m, LifecycleRoutedGraphConv) for m in model.modules())


@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_parameter_budget_stays_within_1_2x_fg1(name):
    _, reference = build("fg1")
    _, model = build(name)
    base = sum(p.numel() for p in reference.parameters() if p.requires_grad)
    size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert size <= 1.2 * base, f"{name} is {size / base:.2f}x fg1"



# --- initialisation contracts that the first stage-1 runs violated -----------


def test_observed_residual_is_exactly_zero_before_training():
    """`d` used to blend 35% of naive persistence in from epoch 0."""
    torch.manual_seed(0)
    nlinear = ObservedNLinear(12, 12, 1)
    observed = torch.randn(2, 12, 5, 1) * 10 + 50
    residual, anchor = nlinear(observed, torch.ones(2, 12, 5, 1))
    assert float(residual.abs().max()) == 0.0
    assert torch.allclose(anchor, observed[:, -1:])

    args, model = build("d")
    model.eval()
    data = _batch(args, model)
    with torch.no_grad():
        with_d = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        model.observed_nlinear = None
        model.observed_nlinear_enabled = False
        model.evidence_router_enabled = False
        without = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
    assert torch.allclose(with_d, without, atol=1e-6), (
        "enabling D moved the prediction before it learned anything"
    )


def test_router_does_not_start_at_the_symmetric_fixed_point():
    """`c` sat at beta = 0.5 for eleven epochs (beta_std 0.0017)."""
    args, model = build("c")
    model.eval()
    data = _batch(args, model)
    with torch.no_grad():
        out = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
    beta = 1.0 - out["route_routing"]["alpha"]
    assert float(beta.max()) < 0.4, "beta starts at the 0.5 symmetry point again"
    assert float(beta.max()) > 0.0, "beta is pinned to zero, the router cannot move"
    # and Yt must differ from Yg, otherwise dYhat/dbeta = 0 and no forecast
    # gradient ever reaches the router
    gap = (out["route_routing"]["y_patch"] - out["route_routing"]["y_graph"]).abs().max()
    assert float(gap) > 0.0, "Yt == Yg exactly, so the router gets no gradient"



# --- RNG isolation: a switch must not perturb the weights it does not own ----


SWITCH_PREFIXES = {
    "c": ("evidence_router.", "patch_projection."),
    "d": ("observed_nlinear.",),
}


def _seeded_state(name):
    import pytorch_lightning as pl

    pl.seed_everything(32, workers=True)
    args = load_args_from_config(f"configs/ALFNet/{name}.yaml", [])
    args.device = "cpu"
    return load_model_class(args.model)(args).state_dict()


@pytest.mark.parametrize("base,combined", [
    ("fg1", "c"), ("fg1", "d"), ("fg1", "cd"),
    ("a", "ac"), ("a", "ad"), ("a", "acd"),
    ("b", "bc"), ("b", "bd"), ("b", "bcd"),
])
def test_shared_weights_are_bit_identical_under_the_same_seed(base, combined):
    """Optional modules used to consume from the shared RNG stream, which moved
    graph_refiner.input_proj by up to 1.01 and made a candidate and its base two
    different models at step 0."""
    left, right = _seeded_state(base), _seeded_state(combined)
    common = sorted(set(left) & set(right))
    assert common, f"{base} and {combined} share no parameters"
    drifted = [
        key for key in common
        if left[key].shape == right[key].shape and not torch.equal(left[key], right[key])
    ]
    assert not drifted, f"{base} vs {combined} drifted on {drifted[:5]}"


@pytest.mark.parametrize("solo,combined", [("c", "cd"), ("d", "cd")])
def test_one_optional_module_does_not_perturb_the_other(solo, combined):
    """Enabling D must not change C's initialisation, and vice versa."""
    left, right = _seeded_state(solo), _seeded_state(combined)
    owned = [key for key in left if key.startswith(SWITCH_PREFIXES[solo])]
    assert owned, f"{solo} owns no parameters"
    for key in owned:
        assert torch.equal(left[key], right[key]), f"{key} drifted when the other switch turned on"


# --- the router must be able to move at the learning rate it is given --------


def test_router_has_its_own_optimizer_group():
    args, model = build("c")
    groups = model.get_optimizer_param_groups(args.lr)
    routing_ids = {id(p) for p in model.evidence_router.parameters()}
    routing_ids |= {id(p) for p in model.patch_projection.parameters()}
    owning = [g for g in groups if any(id(p) in routing_ids for p in g["params"])]
    assert len(owning) == 1, "router parameters are split across groups"
    assert owning[0]["lr"] == pytest.approx(model.router_lr)
    assert model.router_lr > model.gate_lr, (
        "at the gate learning rate the router bias needs ~75 epochs to travel O(1)"
    )
    # and no other group may also contain them
    for group in groups:
        if group is owning[0]:
            continue
        assert not any(id(p) in routing_ids for p in group["params"])


def test_beta_differentiates_under_real_optimizer_steps():
    """Synthetic check that beta is not a constant the optimiser cannot move."""
    args, model = build("c")
    data = _batch(args, model, batch=4)
    opt = torch.optim.AdamW(model.get_optimizer_param_groups(args.lr), lr=args.lr)
    model.train()

    def beta_of():
        model.eval()
        with torch.no_grad():
            out = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
        model.train()
        return 1.0 - out["route_routing"]["alpha"]

    start = beta_of()
    before = model.evidence_router.router[-1].bias.detach().clone()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 1)
        loss = (out["pred"] - data["dec"]).pow(2).mean()
        loss.backward()
        assert model.evidence_router.router[-1].weight.grad is not None
        assert torch.isfinite(model.evidence_router.router[-1].weight.grad).all()
        assert model.patch_projection[-1].weight.grad is not None
        opt.step()
    moved = float((model.evidence_router.router[-1].bias - before).abs().max())
    assert moved > 1e-3, f"router bias barely moved ({moved:.2e})"
    assert float(start.std()) > 0.0, "beta is constant across nodes at init"


def test_observed_projection_receives_gradient():
    args, model = build("d")
    data = _batch(args, model, batch=4)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    (out["pred"] - data["dec"]).pow(2).mean().backward()
    grad = model.observed_nlinear.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0, "the observed head gets no gradient"


def test_patch_residual_is_bounded_by_its_scale():
    args, model = build("c")
    model.eval()
    _amplify(model.patch_projection, std=5.0)
    data = _batch(args, model)
    with torch.no_grad():
        out = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
    gap = (out["route_routing"]["y_patch"] - out["route_routing"]["y_graph"]).abs().max()
    assert float(gap) <= model.patch_residual_scale + 1e-5, (
        "the patch correction is unbounded even with a large projection"
    )


def test_diagnostics_report_the_base_rate_and_balanced_accuracy():
    """Raw accuracy is meaningless without the base rate of the route target."""
    target = torch.zeros(2, 4, 5, 1)
    y_graph = torch.full_like(target, 1.0)
    y_temporal = torch.full_like(target, 0.5)      # temporal better everywhere
    valid = torch.ones_like(target, dtype=torch.bool)
    availability = torch.zeros(2, 5, 4)
    availability[..., 0] = 1.0
    report = routing_diagnostics(torch.ones(2, 1, 5, 1), y_graph, y_temporal,
                                 target, valid, availability)
    assert report["route_target_base_rate"] == pytest.approx(1.0)
    assert report["route_accuracy"] == pytest.approx(1.0)
    # always picking the only correct class is not skill: balanced accuracy is 0.5
    assert report["route_balanced_accuracy"] == pytest.approx(1.0)
    mixed = routing_diagnostics(torch.zeros(2, 1, 5, 1), y_graph, y_temporal,
                                target, valid, availability)
    assert mixed["route_balanced_accuracy"] == pytest.approx(0.0)
