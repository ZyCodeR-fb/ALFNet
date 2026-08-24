"""What counts as leakage between ALFNet's branches, and what does not.

The earlier version of this file asserted that ALFTemporal's output must not
depend on the values sitting in held-out node slots. That is the wrong contract
and it is recorded here so it is not reintroduced. Once a shared imputer runs
over the full node axis, every node's representation is *expected* to depend on
its neighbours' imputed values -- that is what propagation is for. Perturbing a
legitimate intermediate tensor therefore proves nothing.

The contract that matters is on the raw payload signal: measurements the
protocol forbids for this phase must not reach the visible input, the imputed
representation, either branch, the loss, the gradients or an optimiser step.
`tests/test_runtime_context.py` asserts it where the exposure is decided; this
file asserts it end to end, through the real v4 loader, the real adapter and a
real ALFNet forward, because that is the path a run will take.
"""

import copy
import dataclasses

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from args import load_args_from_config
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from lib.runtime_context import ContextRefused, build_runtime_context
from lib.v4_payload import load_v4_payload, save_v4_payload

CONFIGS = {"PEMS08": "configs/paper/v4/PEMS08_ALFNet.yaml",
           "PEMSBAY": "configs/paper/v4/PEMSBAY_ALFNet.yaml"}
PAYLOADS = {"PEMS08": "data_v4/PEMS08/raw_data.pkl",
            "PEMSBAY": "data_v4/PEMS-BAY/raw_data.pkl"}


# --- fixtures -----------------------------------------------------------------


def _build_model(dataset, context, seed=1234):
    import model.ALFNet.ALFNet as alfnet

    args = load_args_from_config(CONFIGS[dataset], [])
    args.device = "cpu"
    torch.manual_seed(seed)
    return alfnet.Model.from_runtime_context(args, context)


def _batch(dataset, context, payload_path=None, batch_size=2, scaler=None):
    data = V4LifecycleDataset(context, payload_path or PAYLOADS[dataset],
                              scaler=scaler)
    return next(iter(DataLoader(data, batch_size=batch_size, shuffle=False))), data


@pytest.fixture(scope="module")
def p8_context():
    return build_runtime_context("PEMS08", "train")


@pytest.fixture(scope="module")
def p8_batch(p8_context):
    return _batch("PEMS08", p8_context)[0]


@pytest.fixture(scope="module")
def p8_model(p8_context):
    return _build_model("PEMS08", p8_context)


def _forbidden_nodes(context):
    mask = np.ones(context.num_nodes, dtype=bool)
    mask[context.signal_scope] = False
    return np.nonzero(mask)[0]


def _poison(tmp_path, nodes, value, channel=None, name="raw_data.pkl",
            payload_path=PAYLOADS["PEMS08"]):
    """A payload identical to the canonical one except on `nodes`.

    Every timestep, so the forbidden nodes' *history and future* both change.
    A large positive value rather than NaN: a leak then shows up as a large
    finite difference in whatever it reached, instead of turning one comparison
    into NaN != NaN and leaving the rest silent.
    """
    payload = load_v4_payload(payload_path)
    data = payload["data"].copy()
    if channel is None:
        data[:, nodes, :] = value
    else:
        data[:, nodes, channel] = value
    return save_v4_payload(tmp_path / name, data=data, adj=payload["adj"],
                           time_stamp=payload["time_stamp"],
                           pos_stamp=payload["pos_stamp"])


# --- the wrong contract, kept as a guard --------------------------------------


def test_depending_on_an_imputed_representation_is_allowed():
    """A guard against reintroducing the wrong requirement.

    Two nodes, one in scope and one not. The imputer legitimately gives the
    out-of-scope node a representation derived from topology and from in-scope
    measurements, and a downstream branch may read it. What it may not do is
    depend on the out-of-scope node's own measurement, which the context has
    already removed by this point.
    """
    context = build_runtime_context("PEMS08", "train")
    raw = torch.zeros(1, 4, context.num_nodes, 1)
    observed = torch.ones(1, 4, context.num_nodes, dtype=torch.bool)
    visible, available = context.visible_signal(raw, observed)

    # a toy imputer: every node takes the mean of its in-scope neighbours
    adjacency = (context.adjacency > 0).float()
    weights = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    filled = torch.einsum("nm,btmc->btnc", weights, visible)

    forbidden = ~context.signal_scope_mask()
    assert forbidden.any()
    # The out-of-scope nodes carry a representation, and it came from the graph,
    # not from their own measurements.
    assert filled.shape[2] == context.num_nodes
    assert torch.equal(visible[:, :, forbidden, :],
                       torch.zeros_like(visible[:, :, forbidden, :]))


# --- one imputer, one node axis, two branches ---------------------------------


def _instrument(model):
    """Capture what the imputer and each branch actually received.

    The graph expert is reached through `__call__`, so a forward hook sees it.
    The patch expert is reached through `forward_v4`, which no hook observes --
    wrapping the bound method is the only way to record it, and it also proves
    the v4 entry point is the one being used.
    """
    seen = {"grin_in": [], "grin_out": [], "graph_in": [], "patch_in": [],
            "grin_adj": [], "graph_adj": [], "adj_ptr": []}

    def imputer_hook(module, inputs, output):
        seen["grin_adj"].append(inputs[0].detach().clone())
        seen["adj_ptr"].append(inputs[0].data_ptr())
        seen["grin_in"].append(inputs[1].detach().clone())
        seen["grin_out"].append(output.detach().clone())

    def graph_hook(module, inputs, output):
        seen["graph_adj"].append(inputs[0].detach().clone())
        seen["adj_ptr"].append(inputs[0].data_ptr())
        seen["graph_in"].append(inputs[1].detach().clone())

    handles = [model.shared_input_imputer.register_forward_hook(imputer_hook),
               model.graph_expert.register_forward_hook(graph_hook)]
    original = model.patch_expert.forward_v4

    def patch_v4(x_enc, x_t_mark):
        seen["patch_in"].append(x_enc.detach().clone())
        return original(x_enc, x_t_mark)

    model.patch_expert.forward_v4 = patch_v4
    return seen, handles, original


def test_both_branches_receive_the_same_full_n_imputed_tensor(p8_context, p8_batch):
    """The property this file exists for, through the real v4 forward."""
    model = _build_model("PEMS08", p8_context).eval()
    seen, handles, original = _instrument(model)
    try:
        with torch.no_grad():
            output = model.forward_v4(p8_batch, p8_context)
    finally:
        for handle in handles:
            handle.remove()
        model.patch_expert.forward_v4 = original

    nodes = p8_context.num_nodes
    # One imputer, one call. A second call would mean two different fills.
    assert len(seen["grin_out"]) == 1, seen["grin_out"]
    assert len(seen["graph_in"]) == 1 and len(seen["patch_in"]) == 1

    grin_in, filled = seen["grin_in"][0], seen["grin_out"][0]
    graph_in, patch_in = seen["graph_in"][0], seen["patch_in"][0]
    for name, tensor in (("grin input", grin_in), ("grin output", filled),
                         ("graph input", graph_in), ("patch input", patch_in),
                         ("prediction", output["pred"])):
        assert tensor.shape[2] == nodes, (name, tensor.shape)

    # Both branches slice the same tensor, so the common raw/imputed channels
    # are the same numbers rather than two paths that happen to agree.
    raw = model.raw_enc_in
    assert graph_in.shape[-1] >= raw and patch_in.shape[-1] >= raw
    assert torch.equal(graph_in[..., :raw], patch_in[..., :raw])
    common = min(graph_in.shape[-1], patch_in.shape[-1])
    assert torch.equal(graph_in[..., :common], patch_in[..., :common])
    assert torch.equal(graph_in, filled[..., :graph_in.shape[-1]])
    assert torch.equal(patch_in, filled[..., :patch_in.shape[-1]])
    assert torch.isfinite(output["pred"]).all()

    # The adjacency is the context's own full-N matrix, not one re-derived from
    # a path, and both consumers were handed the same object.
    for stage in ("grin_adj", "graph_adj"):
        adjacency = seen[stage][0]
        assert adjacency.shape == (nodes, nodes), stage
        assert torch.equal(adjacency, p8_context.adjacency.to(adjacency.dtype)), stage
    # The same storage, so the adapter resolved it once and neither consumer
    # rebuilt or reloaded it.
    assert len(set(seen["adj_ptr"])) == 1, seen["adj_ptr"]
    assert seen["adj_ptr"][0] == model._v4_adjacency(
        p8_context, p8_batch["base_visible_input"].device,
        p8_batch["base_visible_input"].dtype).data_ptr()


def test_the_active_path_has_no_k_to_n_scatter(p8_context, p8_batch):
    import model.ALFTemporal.ALFTemporal as alftemporal

    assert not hasattr(alftemporal.Model, "_scatter_to_full_nodes")
    for name in ("load_project_adjacency", "load_project_positions"):
        assert not hasattr(alftemporal, name)

    model = _build_model("PEMS08", p8_context).eval()
    with torch.no_grad():
        output = model.forward_v4(p8_batch, p8_context)
    assert output["pred"].shape[2] == p8_context.num_nodes

# --- fail closed on anything that is not this run's full-N batch ---------------


def test_a_k_node_batch_is_refused_rather_than_padded(p8_context, p8_batch,
                                                      p8_model):
    known = len(p8_context.known_nodes)
    assert known < p8_context.num_nodes
    batch = dict(p8_batch)
    for field in ("base_visible_input", "lifecycle_context"):
        batch[field] = batch[field][:, :, :known]
    with pytest.raises(ValueError, match="never padded"):
        p8_model.forward_v4(batch, p8_context)


def test_a_batch_from_another_phase_is_refused(p8_context, p8_model):
    """Right shapes, right assets, wrong exposure -- and it must still lose."""
    validation = build_runtime_context("PEMS08", "validation")
    batch, _ = _batch("PEMS08", validation, batch_size=1)
    assert (validation.static_asset_hash() == p8_context.static_asset_hash())
    with pytest.raises(ContextRefused, match="exposure permissions"):
        p8_model.forward_v4(batch, p8_context)


def test_a_batch_mixing_two_exposures_is_refused(p8_context, p8_batch, p8_model):
    batch = copy.deepcopy(p8_batch)
    batch["context_identity"]["phase"] = ["train", "validation"]
    with pytest.raises(ValueError, match="mixes phase"):
        p8_model.forward_v4(batch, p8_context)


def test_a_context_on_another_node_order_is_refused(p8_context, p8_batch,
                                                    p8_model):
    """A permuted graph has a different node order and a different hash."""
    permutation = np.roll(np.arange(p8_context.num_nodes), 1)
    permuted = dataclasses.replace(
        p8_context,
        adjacency=p8_context.adjacency[permutation][:, permutation].contiguous(),
        adjacency_sha256="0" * 64, node_ids_sha256="1" * 64, _cache={})
    assert permuted.static_asset_hash() != p8_context.static_asset_hash()
    with pytest.raises(ValueError, match="different static graph assets"):
        p8_model.forward_v4(p8_batch, permuted)


def test_a_missing_context_and_legacy_tuple_are_refused(p8_batch, p8_model):
    with pytest.raises(ValueError, match="requires a RuntimeDataContext"):
        p8_model.forward_v4(p8_batch, None)
    with pytest.raises(ValueError, match="ten-tuple"):
        p8_model.forward_v4((torch.zeros(1),) * 10,
                            build_runtime_context("PEMS08", "train"))


@pytest.mark.parametrize("field", ["base_visible_input", "lifecycle_context",
                                   "encoder_time_marks", "decoder_time_marks",
                                   "context_identity"])
def test_a_batch_missing_a_required_field_is_refused(p8_context, p8_batch,
                                                     p8_model, field):
    batch = {key: value for key, value in p8_batch.items() if key != field}
    with pytest.raises(ValueError, match="missing"):
        p8_model.forward_v4(batch, p8_context)


# --- both datasets, a real context and a real forward -------------------------


@pytest.mark.parametrize("dataset, phase", [("PEMS08", "test"),
                                            ("PEMSBAY", "validation")])
def test_a_real_v4_eval_forward_on_each_dataset(dataset, phase):
    context = build_runtime_context(dataset, phase)
    batch, data = _batch(dataset, context, batch_size=1)
    model = _build_model(dataset, context).eval()
    with torch.no_grad():
        output = model.forward_v4(batch, context)
    assert output["pred"].shape == (1, data.pred_len, context.num_nodes,
                                    model.output_dim)
    assert torch.isfinite(output["pred"]).all()


def test_the_parameter_set_is_exactly_what_it_was(p8_context):
    """forward_v4 is wiring. It may not add, rename or resize a parameter."""
    model = _build_model("PEMSBAY", build_runtime_context("PEMSBAY", "train"))
    assert sum(p.numel() for p in model.parameters()) == 1_037_529
    p8 = _build_model("PEMS08", p8_context)
    assert not [name for name, _ in p8.named_parameters() if "v4" in name]
    # The adjacency cache is a plain attribute, so nothing about it can be
    # mistaken for a learned tensor or travel in a checkpoint.
    assert not [key for key in p8.state_dict() if "v4" in key]


# --- the counterfactual: a forbidden measurement changes nothing ---------------


def _run(dataset, context, payload_path, scaler=None, train=False, seed=1234):
    """One full pass, from the payload through the loader to a prediction."""
    batch, data = _batch(dataset, context, payload_path, scaler=scaler)
    model = _build_model(dataset, context, seed=seed)
    model.train() if train else model.eval()
    seen, handles, original = _instrument(model)
    try:
        torch.manual_seed(seed)
        if train:
            output = model.forward_v4(batch, context)
        else:
            with torch.no_grad():
                output = model.forward_v4(batch, context)
    finally:
        for handle in handles:
            handle.remove()
        model.patch_expert.forward_v4 = original
    return {"batch": batch, "data": data, "model": model, "seen": seen,
            "pred": output["pred"]}


def _masked_mse(prediction, batch):
    mask = batch["loss_mask"].unsqueeze(-1).to(prediction.dtype)
    return ((prediction - batch["target"]) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


@pytest.fixture(scope="module")
def forbidden_counterfactual(tmp_path_factory):
    """Two payloads that differ only where the phase may not look."""
    context = build_runtime_context("PEMS08", "train")
    forbidden = _forbidden_nodes(context)
    assert forbidden.size > 0
    poisoned = _poison(tmp_path_factory.mktemp("forbidden"), forbidden, 1.0e6)

    canonical = load_v4_payload(PAYLOADS["PEMS08"])["data"]
    altered = load_v4_payload(poisoned)["data"]
    # The two payloads really do differ, and only on the forbidden nodes.
    assert not np.array_equal(canonical[:, forbidden, :], altered[:, forbidden, :])
    allowed = np.setdiff1d(np.arange(context.num_nodes), forbidden)
    assert np.array_equal(canonical[:, allowed, :], altered[:, allowed, :])
    return context, poisoned


def test_a_forbidden_measurement_reaches_neither_branch_nor_the_prediction(
        forbidden_counterfactual):
    context, poisoned = forbidden_counterfactual
    base = _run("PEMS08", context, PAYLOADS["PEMS08"])
    other = _run("PEMS08", context, poisoned)

    # The scaler is fitted on training time and training-known nodes, so a
    # forbidden node's values may not move it either.
    assert base["data"].scaler == other["data"].scaler
    for field in ("base_visible_input", "lifecycle_context", "loss_mask",
                  "target", "input_available_mask", "decoder_available_mask",
                  "encoder_time_marks"):
        assert torch.equal(base["batch"][field], other["batch"][field]), field
    for stage in ("grin_in", "grin_out", "graph_in", "patch_in"):
        assert torch.equal(base["seen"][stage][0], other["seen"][stage][0]), stage
    assert torch.equal(base["pred"], other["pred"])
    assert torch.isfinite(base["pred"]).all()


def test_a_forbidden_measurement_moves_no_loss_gradient_or_optimiser_step(
        forbidden_counterfactual):
    """The training-time counterfactual: identical seed, identical everything.

    Exact equality, not a tolerance. The two runs execute the same operations on
    the same numbers in the same order, so any difference at all would mean a
    forbidden value entered the computation rather than that floating point
    drifted.
    """
    context, poisoned = forbidden_counterfactual
    results, losses, steps = [], [], []
    for payload in (PAYLOADS["PEMS08"], poisoned):
        run = _run("PEMS08", context, payload, train=True)
        loss = _masked_mse(run["pred"], run["batch"])
        run["model"].zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: (None if parameter.grad is None
                            else parameter.grad.detach().clone())
                     for name, parameter in run["model"].named_parameters()}
        optimiser = torch.optim.AdamW(run["model"].parameters(), lr=1e-3)
        optimiser.step()
        steps.append({key: value.detach().clone()
                      for key, value in run["model"].state_dict().items()})
        losses.append(loss.detach().clone())
        results.append(gradients)

    assert torch.equal(losses[0], losses[1])
    assert set(results[0]) == set(results[1])
    assert any(value is not None for value in results[0].values())
    for name in results[0]:
        first, second = results[0][name], results[1][name]
        assert (first is None) == (second is None), name
        if first is not None:
            assert torch.equal(first, second), name
    assert set(steps[0]) == set(steps[1])
    for name in steps[0]:
        assert torch.equal(steps[0][name], steps[1][name]), name


def test_an_in_scope_measurement_does_move_the_model(tmp_path):
    """The positive control: the counterfactual above is not a constant model.

    The same machinery, the same shared scaler, one *permitted* node changed.
    Poisoned with a large positive value so natural availability and the origin
    list are untouched and the only thing that moves is the measurement itself.
    """
    context = build_runtime_context("PEMS08", "train")
    visible = int(context.known_nodes[0])
    poisoned = _poison(tmp_path, np.array([visible]), 9.99e2,
                       channel=context.target_channel)
    base = _run("PEMS08", context, PAYLOADS["PEMS08"])
    other = _run("PEMS08", context, poisoned, scaler=base["data"].scaler)

    assert base["batch"]["loss_mask"].equal(other["batch"]["loss_mask"])
    inputs = (base["batch"]["base_visible_input"],
              other["batch"]["base_visible_input"])
    assert not torch.equal(*inputs)
    assert (inputs[0] - inputs[1]).abs().max().item() > 0
    for stage in ("grin_out", "graph_in", "patch_in"):
        assert not torch.equal(base["seen"][stage][0], other["seen"][stage][0]), stage
    assert not torch.equal(base["pred"], other["pred"])
    assert (base["pred"] - other["pred"]).abs().max().item() > 1e-6
