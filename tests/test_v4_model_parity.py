"""Two models, one v4 batch, and one definition of teacher forcing.

The comparison this repo exists to make is only worth anything if ALFNet and
MoGERNN receive the same common signal and are allowed to read the same futures.
Both halves of that are asserted here behaviourally, not by reading the source:

* MoGERNN gets `base_visible_input` and nothing else. Poisoning the lifecycle
  channels must not move a single number it produces -- a source read is not
  enough, because five baselines were once consuming lifecycle channels through
  a config-derived `input_dim` with no visible slice anywhere.
* neither decoder may read a future outside `teacher_forcing_mask`. The test
  writes NaN, Inf and 1e30 into the masked-off target positions and requires the
  prediction to be bit-identical, then flips one allowed position to prove the
  first result is not a model that ignores its decoder input.
"""

import dataclasses

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from args import load_args_from_config
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from lib.runtime_context import ContextRefused, build_runtime_context
from lib.teacher_forcing import (
    TeacherForcingRefused, masked_teacher_forcing, step_teacher_input,
)
from lib.v4_payload import load_v4_payload, save_v4_payload

ALFNET_CONFIGS = {"PEMS08": "configs/paper/v4/PEMS08_ALFNet.yaml",
                  "PEMSBAY": "configs/paper/v4/PEMSBAY_ALFNet.yaml"}
MOGERNN_CONFIGS = {"PEMS08": "configs/paper/v4/PEMS08_MoGERNN.yaml",
                   "PEMSBAY": "configs/paper/v4/PEMSBAY_MoGERNN.yaml"}
# The v3 archive entry point takes only v3 configs, and is exercised as such.
V3_CONFIGS = {"ALFNet": "configs/ALFNet/gs_p8.yaml",
              "MoGERNN": "configs/MoGERNN/PEMS08_lcbt_v3.yaml"}
PAYLOADS = {"PEMS08": "data_v4/PEMS08/raw_data.pkl",
            "PEMSBAY": "data_v4/PEMS-BAY/raw_data.pkl"}


# --- fixtures -----------------------------------------------------------------


def _alfnet(dataset, context, seed=1234):
    import model.ALFNet.ALFNet as alfnet

    args = load_args_from_config(ALFNET_CONFIGS[dataset], [])
    args.device = "cpu"
    torch.manual_seed(seed)
    return alfnet.Model.from_runtime_context(args, context)


def _mogernn(dataset, context, seed=1234):
    import model.comparison_model.MoGERNN.MoGERNN as mogernn

    args = load_args_from_config(MOGERNN_CONFIGS[dataset], [])
    args.device = "cpu"
    torch.manual_seed(seed)
    return mogernn.Model.from_runtime_context(args, context)


BUILDERS = {"ALFNet": _alfnet, "MoGERNN": _mogernn}


def _batch(dataset, context, payload_path=None, batch_size=2):
    data = V4LifecycleDataset(context, payload_path or PAYLOADS[dataset])
    return next(iter(DataLoader(data, batch_size=batch_size, shuffle=False)))


@pytest.fixture(scope="module")
def context():
    return build_runtime_context("PEMS08", "train")


@pytest.fixture(scope="module")
def batch(context):
    return _batch("PEMS08", context)


def _forbidden_nodes(runtime_context):
    mask = np.ones(runtime_context.num_nodes, dtype=bool)
    mask[runtime_context.signal_scope] = False
    return np.nonzero(mask)[0]


def _predict(model, batch, runtime_context, teacher_forcing_mask=None, train=False,
             seed=7):
    model.train() if train else model.eval()
    torch.manual_seed(seed)
    if train:
        return model.forward_v4(batch, runtime_context,
                                teacher_forcing_mask=teacher_forcing_mask)["pred"]
    with torch.no_grad():
        return model.forward_v4(batch, runtime_context,
                                teacher_forcing_mask=teacher_forcing_mask)["pred"]


# --- MoGERNN reads the common input and nothing else ---------------------------


def test_mogernn_v4_reads_only_common_base_input(context, batch):
    """The lifecycle channels are ALFNet's method inputs, not a shared feature."""
    model = _mogernn("PEMS08", context)
    reference = _predict(model, batch, context)

    poisoned = dict(batch)
    lifecycle = batch["lifecycle_context"].clone()
    # Every lifecycle channel, to a value nothing could survive being fed.
    lifecycle[:] = 1.0e6
    poisoned["lifecycle_context"] = lifecycle
    poisoned["lifecycle_role_ids"] = torch.zeros_like(batch["lifecycle_role_ids"])
    poisoned["lifecycle_state_age"] = torch.full_like(batch["lifecycle_state_age"],
                                                      1.0e6)
    assert not torch.equal(poisoned["lifecycle_context"], batch["lifecycle_context"])
    assert torch.equal(poisoned["base_visible_input"], batch["base_visible_input"])

    assert torch.equal(_predict(model, poisoned, context), reference)


def test_an_in_scope_measurement_does_move_mogernn(context, batch):
    """The positive control for the test above."""
    model = _mogernn("PEMS08", context)
    reference = _predict(model, batch, context)

    moved = dict(batch)
    visible = batch["base_visible_input"].clone()
    node = int(context.known_nodes[0])
    visible[:, :, node, :] = visible[:, :, node, :] + 1.0
    moved["base_visible_input"] = visible

    prediction = _predict(model, moved, context)
    assert not torch.equal(prediction, reference)
    assert (prediction - reference).abs().max().item() > 1e-6


def test_both_adapters_receive_the_identical_common_tensor(context, batch):
    """One `base_visible_input`, handed unchanged to both models."""
    seen = {}

    def record(name, model):
        original = model.data_pre if name == "MoGERNN" else model.shared_input_imputer

        def hook(module, inputs, output):
            seen[name] = inputs[1].detach().clone()

        return original.register_forward_hook(hook)

    alf, mog = _alfnet("PEMS08", context).eval(), _mogernn("PEMS08", context).eval()
    handles = [record("ALFNet", alf), record("MoGERNN", mog)]
    try:
        with torch.no_grad():
            alf.forward_v4(batch, context)
            mog.forward_v4(batch, context)
    finally:
        for handle in handles:
            handle.remove()

    raw = alf.raw_enc_in
    assert mog.raw_enc_in == raw
    # ALFNet appends its lifecycle context after the common channels; MoGERNN
    # never sees them. The common part must be the same numbers.
    assert torch.equal(seen["ALFNet"][..., :raw], seen["MoGERNN"][..., :raw])
    assert (seen["ALFNet"][..., :raw] - seen["MoGERNN"][..., :raw]).abs().max() == 0
    assert torch.equal(seen["MoGERNN"], batch["base_visible_input"])
    assert seen["MoGERNN"].shape[2] == context.num_nodes


def test_mogernn_v4_runs_on_the_contexts_full_n_adjacency(context, batch):
    seen = {}

    model = _mogernn("PEMS08", context).eval()

    def hook(module, inputs, output):
        # No return value: a forward hook that returns one replaces the output.
        seen["adj"] = inputs[0].detach().clone()

    handle = model.data_pre.register_forward_hook(hook)
    try:
        with torch.no_grad():
            prediction = model.forward_v4(batch, context)["pred"]
    finally:
        handle.remove()

    nodes = context.num_nodes
    assert seen["adj"].shape == (nodes, nodes)
    assert torch.equal(seen["adj"], context.adjacency.to(seen["adj"].dtype))
    assert prediction.shape == (batch["base_visible_input"].shape[0], 12, nodes, 1)
    assert torch.isfinite(prediction).all()


# --- fail closed, both models the same way ------------------------------------


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_k_node_batch_is_refused_by_both(context, batch, who):
    model = BUILDERS[who]("PEMS08", context).eval()
    known = len(context.known_nodes)
    assert known < context.num_nodes
    short = dict(batch)
    for field in ("base_visible_input", "lifecycle_context"):
        short[field] = batch[field][:, :, :known]
    with pytest.raises(ValueError, match="never padded"):
        model.forward_v4(short, context)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_permuted_node_order_is_refused_by_both(context, batch, who):
    model = BUILDERS[who]("PEMS08", context).eval()
    permutation = np.roll(np.arange(context.num_nodes), 1)
    permuted = dataclasses.replace(
        context,
        adjacency=context.adjacency[permutation][:, permutation].contiguous(),
        adjacency_sha256="0" * 64, node_ids_sha256="1" * 64, _cache={})
    assert permuted.static_asset_hash() != context.static_asset_hash()
    with pytest.raises(ValueError, match="different static graph assets"):
        model.forward_v4(batch, permuted)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_batch_from_another_exposure_is_refused_by_both(context, who):
    model = BUILDERS[who]("PEMS08", context).eval()
    validation = build_runtime_context("PEMS08", "validation")
    assert validation.static_asset_hash() == context.static_asset_hash()
    with pytest.raises(ContextRefused, match="exposure permissions"):
        model.forward_v4(_batch("PEMS08", validation, batch_size=1), context)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_wrong_target_channel_is_refused_by_both(context, who):
    """A config that contradicts the payload loses at construction."""
    import model.ALFNet.ALFNet as alfnet
    import model.comparison_model.MoGERNN.MoGERNN as mogernn

    module = alfnet if who == "ALFNet" else mogernn
    config = ALFNET_CONFIGS if who == "ALFNet" else MOGERNN_CONFIGS
    args = load_args_from_config(config["PEMS08"], [])
    args.device = "cpu"
    args.target_channel = context.target_channel + 3
    with pytest.raises(ValueError, match="refusing rather than silently winning"):
        module.Model.from_runtime_context(args, context)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_the_v3_path_and_a_missing_context_are_refused_by_both(context, batch, who):
    import model.ALFNet.ALFNet as alfnet
    import model.comparison_model.MoGERNN.MoGERNN as mogernn

    module = alfnet if who == "ALFNet" else mogernn
    config = ALFNET_CONFIGS if who == "ALFNet" else MOGERNN_CONFIGS
    with pytest.raises(ValueError, match="requires a RuntimeDataContext"):
        module.Model.from_runtime_context(
            load_args_from_config(config["PEMS08"], []), None)

    # The archive constructor takes a v3 config; a v4 one is refused there.
    args = load_args_from_config(V3_CONFIGS[who], [])
    args.device = "cpu"
    archive = module.Model(args).eval()
    with pytest.raises(ValueError, match="built through from_runtime_context"):
        archive.forward_v4(batch, context)


# --- the masked teacher forcing contract --------------------------------------


def test_the_teacher_tensor_is_zero_outside_the_mask():
    """Defence one, pinned on its own.

    The end-to-end tests below assert the property that matters, but they hold
    even with this broken because the decoder's per-step `where` catches it too.
    Defence in depth is the intent; an untested layer of it is not, so each layer
    gets a test that fails when only that layer is removed.
    """
    target = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4, 1)
    target[0, 0, 0] = float("nan")
    available = torch.ones(2, 3, 4, dtype=torch.bool)
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[0, 1, 2] = True

    teacher_input, returned = masked_teacher_forcing(
        target, available, mask, training=True, who="unit")
    assert torch.equal(returned, mask)
    assert teacher_input[0, 1, 2].item() == target[0, 1, 2].item()
    # Everywhere else exactly zero -- including the NaN, which `target * mask`
    # would have left as NaN.
    assert torch.equal(teacher_input[~mask],
                       torch.zeros_like(teacher_input[~mask]))
    assert torch.isfinite(teacher_input).all()

    empty, all_false = masked_teacher_forcing(
        target, available, None, training=False, who="unit")
    assert not bool(all_false.any())
    assert torch.equal(empty, torch.zeros_like(empty))


def test_a_decoder_step_selects_per_node_not_per_batch():
    """Defence two, pinned on its own: the selection is elementwise."""
    teacher_input = torch.full((2, 3, 4, 1), 9.0)
    output = torch.zeros(2, 4, 1)
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[0, 1, 2] = True

    selected = step_teacher_input(teacher_input, mask, 1, output)
    assert selected[0, 2, 0].item() == 9.0
    assert selected[0, 0, 0].item() == 0.0 and selected[1, 2, 0].item() == 0.0
    assert int((selected == 9.0).sum()) == 1


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_teacher_forcing_is_per_node_within_a_forced_step(context, batch, who):
    """A batch-level boolean would let every node read its future at that step.

    One horizon step is forced; inside it exactly one node is forced and another
    is not. Perturbing the unforced node's future in that same step must change
    nothing, while perturbing the forced node's must.
    """
    model = BUILDERS[who]("PEMS08", context)
    allowed = batch["decoder_available_mask"].to(torch.bool)
    step = 3
    nodes = allowed[0, step].nonzero().flatten().tolist()
    assert len(nodes) >= 2
    forced_node, free_node = int(nodes[0]), int(nodes[1])

    forced = torch.zeros_like(allowed)
    forced[0, step, forced_node] = True
    reference = _predict(model, batch, context, forced, train=True)

    quiet = batch["target"].clone()
    quiet[0, step, free_node] = 1.0e6
    assert torch.equal(_predict(model, dict(batch, target=quiet), context, forced,
                                train=True), reference)

    loud = batch["target"].clone()
    loud[0, step, forced_node] = loud[0, step, forced_node] + 5.0
    assert not torch.equal(_predict(model, dict(batch, target=loud), context, forced,
                                    train=True), reference)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_an_unmasked_future_never_reaches_a_decoder(context, batch, who):
    """NaN, Inf and 1e30 where the mask is false; the prediction must not move.

    `where(mask, target, 0)` rather than `target * mask`: multiplying would turn
    every one of these into NaN and the leak would be loud instead of silent,
    which is not the property being asserted.
    """
    model = BUILDERS[who]("PEMS08", context)
    allowed = batch["decoder_available_mask"].to(torch.bool)
    # Force on roughly half the allowed positions, deterministically.
    forced = allowed.clone()
    forced[:, ::2] = False
    assert bool(forced.any()) and bool((allowed & ~forced).any())

    reference = _predict(model, batch, context, forced, train=True)
    for poison in (float("nan"), float("inf"), -1.0e30, 1.0e30):
        target = batch["target"].clone()
        target[~forced] = poison
        moved = dict(batch, target=target)
        assert torch.equal(_predict(model, moved, context, forced, train=True),
                           reference), poison
    assert torch.isfinite(reference).all()


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_forced_future_does_reach_the_decoder(context, batch, who):
    """The positive control: teacher forcing is wired, not a no-op."""
    model = BUILDERS[who]("PEMS08", context)
    allowed = batch["decoder_available_mask"].to(torch.bool)
    forced = allowed.clone()
    forced[:, ::2] = False

    reference = _predict(model, batch, context, forced, train=True)
    target = batch["target"].clone()
    target[forced] = target[forced] + 5.0
    prediction = _predict(model, dict(batch, target=target), context, forced,
                          train=True)
    assert not torch.equal(prediction, reference)
    assert (prediction - reference).abs().max().item() > 1e-6

    # ... and with no mask at all the decoder feeds on its own output only.
    free = _predict(model, batch, context, None, train=True)
    assert not torch.equal(free, reference)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_forcing_outside_the_decoder_available_mask_is_refused(context, batch, who):
    model = BUILDERS[who]("PEMS08", context).train()
    allowed = batch["decoder_available_mask"].to(torch.bool)
    outside = (~allowed).nonzero()
    assert outside.shape[0] > 0
    illegal = torch.zeros_like(allowed)
    illegal[tuple(outside[0].tolist())] = True
    with pytest.raises(TeacherForcingRefused, match="decoder_available_mask is false"):
        model.forward_v4(batch, context, teacher_forcing_mask=illegal)


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_teacher_forcing_in_eval_is_refused(context, batch, who):
    model = BUILDERS[who]("PEMS08", context).eval()
    allowed = batch["decoder_available_mask"].to(torch.bool)
    one = torch.zeros_like(allowed)
    one[tuple(allowed.nonzero()[0].tolist())] = True
    with pytest.raises(TeacherForcingRefused, match="eval mode"):
        model.forward_v4(batch, context, teacher_forcing_mask=one)
    # An all-false mask and None are both fine in eval.
    with torch.no_grad():
        assert torch.equal(
            model.forward_v4(batch, context,
                             teacher_forcing_mask=torch.zeros_like(allowed))["pred"],
            model.forward_v4(batch, context)["pred"])


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_a_float_mask_is_refused(context, batch, who):
    model = BUILDERS[who]("PEMS08", context).train()
    allowed = batch["decoder_available_mask"].to(torch.bool)
    with pytest.raises(TeacherForcingRefused, match="must be bool"):
        model.forward_v4(batch, context,
                         teacher_forcing_mask=allowed.to(torch.float32))


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_no_v4_decoder_draws_a_random_number(monkeypatch, context, batch, who):
    """The schedule is the runner's to generate, not the model's to invent.

    `torch.rand` is banned globally rather than in one decoder's namespace, so
    this also rules out a random decision taken anywhere else on the v4 path.
    """
    model = BUILDERS[who]("PEMS08", context).train()
    allowed = batch["decoder_available_mask"].to(torch.bool)

    def boom(*args, **kwargs):
        raise AssertionError("the v4 path drew a random number")

    monkeypatch.setattr(torch, "rand", boom)
    prediction = model.forward_v4(batch, context, teacher_forcing_mask=allowed)["pred"]
    assert prediction.shape[2] == context.num_nodes
    # ... and with no mask either, which is the eval and the default path.
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model.forward_v4(batch, context)["pred"]).all()


@pytest.mark.parametrize("who", sorted(BUILDERS))
def test_the_same_batch_and_mask_reproduce(context, batch, who):
    model = BUILDERS[who]("PEMS08", context)
    allowed = batch["decoder_available_mask"].to(torch.bool)
    runs = []
    for _ in range(2):
        fresh = BUILDERS[who]("PEMS08", context)
        prediction = _predict(fresh, batch, context, allowed, train=True)
        loss = prediction.pow(2).mean()
        fresh.zero_grad(set_to_none=True)
        loss.backward()
        runs.append((prediction.detach().clone(),
                     {name: (None if parameter.grad is None
                             else parameter.grad.detach().clone())
                      for name, parameter in fresh.named_parameters()}))
    assert torch.equal(runs[0][0], runs[1][0])
    assert any(value is not None for value in runs[0][1].values())
    for name in runs[0][1]:
        first, second = runs[0][1][name], runs[1][1][name]
        assert (first is None) == (second is None), name
        if first is not None:
            assert torch.equal(first, second), name
    del model


# --- the forbidden-signal counterfactual, now for MoGERNN too ------------------


@pytest.fixture(scope="module")
def poisoned_payload(tmp_path_factory, context):
    """A payload identical to the canonical one outside the phase's scope."""
    forbidden = _forbidden_nodes(context)
    assert forbidden.size > 0
    payload = load_v4_payload(PAYLOADS["PEMS08"])
    data = payload["data"].copy()
    data[:, forbidden, :] = 1.0e6
    path = save_v4_payload(tmp_path_factory.mktemp("forbidden") / "raw_data.pkl",
                           data=data, adj=payload["adj"],
                           time_stamp=payload["time_stamp"],
                           pos_stamp=payload["pos_stamp"])
    allowed = np.setdiff1d(np.arange(context.num_nodes), forbidden)
    assert np.array_equal(payload["data"][:, allowed, :], data[:, allowed, :])
    return path


def test_a_forbidden_measurement_moves_nothing_in_mogernn(context, poisoned_payload):
    """End to end: v4 loader -> MoGERNN.forward_v4 -> prediction, loss, backward."""
    results = []
    for payload in (PAYLOADS["PEMS08"], poisoned_payload):
        batch = _batch("PEMS08", context, payload)
        model = _mogernn("PEMS08", context).train()
        torch.manual_seed(7)
        prediction = model.forward_v4(
            batch, context,
            teacher_forcing_mask=batch["decoder_available_mask"].to(torch.bool))["pred"]
        mask = batch["loss_mask"].unsqueeze(-1).to(prediction.dtype)
        loss = ((prediction - batch["target"]) ** 2 * mask).sum() / mask.sum().clamp_min(1)
        model.zero_grad(set_to_none=True)
        loss.backward()
        results.append((batch["base_visible_input"], prediction.detach().clone(),
                        loss.detach().clone(),
                        {name: (None if parameter.grad is None
                                else parameter.grad.detach().clone())
                         for name, parameter in model.named_parameters()}))

    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])
    assert any(value is not None for value in results[0][3].values())
    for name in results[0][3]:
        first, second = results[0][3][name], results[1][3][name]
        assert (first is None) == (second is None), name
        if first is not None:
            assert torch.equal(first, second), name


# --- nothing about either model moved -----------------------------------------


def test_neither_parameter_set_changed(context):
    alfnet = _alfnet("PEMS08", context)
    assert sum(p.numel() for p in alfnet.parameters()) == 1_037_529
    assert sum(p.numel() for p in _alfnet(
        "PEMSBAY", build_runtime_context("PEMSBAY", "train")).parameters()) == 1_037_529

    import model.comparison_model.MoGERNN.MoGERNN as mogernn

    for dataset in sorted(MOGERNN_CONFIGS):
        args = load_args_from_config(MOGERNN_CONFIGS[dataset], [])
        args.device = "cpu"
        torch.manual_seed(1234)
        archive = mogernn.Model(args)
        bound = _mogernn(dataset, build_runtime_context(dataset, "train"))
        # The v4 entry point binds metadata and builds the identical network.
        assert sum(p.numel() for p in bound.parameters()) == 226_270
        assert {name: tuple(p.shape) for name, p in bound.named_parameters()} == \
               {name: tuple(p.shape) for name, p in archive.named_parameters()}
        assert not [key for key in bound.state_dict() if "v4" in key or "context" in key]


@pytest.mark.parametrize("dataset", sorted(PAYLOADS))
def test_a_real_v4_eval_forward_for_mogernn_on_each_dataset(dataset):
    runtime_context = build_runtime_context(
        dataset, "test" if dataset == "PEMS08" else "validation")
    batch = _batch(dataset, runtime_context, batch_size=1)
    model = _mogernn(dataset, runtime_context).eval()
    with torch.no_grad():
        prediction = model.forward_v4(batch, runtime_context)["pred"]
    assert prediction.shape == (1, 12, runtime_context.num_nodes, 1)
    assert torch.isfinite(prediction).all()
