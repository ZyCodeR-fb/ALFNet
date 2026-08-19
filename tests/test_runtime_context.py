"""What the runtime context exposes, and what it refuses.

The counterfactual here is on the *raw payload signal*, not on an intermediate
tensor. Perturbing a legitimate intermediate representation proves nothing:
once a shared imputer runs over the full node axis, every node's representation
is expected to depend on the imputed values of its neighbours. The question that
matters is whether measurements the protocol forbids can reach anything at all.
"""

import numpy as np
import pytest
import torch

from lib.runtime_context import (
    ContextRefused,
    RuntimeDataContext,
    build_runtime_context,
    split_unseen_pool,
)

DATASETS = ("PEMS03", "PEMS04", "PEMS08", "PEMSBAY")


@pytest.fixture(scope="module")
def contexts():
    return {phase: build_runtime_context("PEMS08", phase)
            for phase in ("train", "validation", "test")}


def _batch(context, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (2, 6, context.num_nodes)
    raw = torch.randn(*shape, 1, generator=generator)
    observed = torch.rand(*shape, generator=generator) > 0.1
    return raw, observed


# --- node scopes --------------------------------------------------------------


def test_validation_and_test_unseen_are_disjoint():
    for dataset in DATASETS:
        context = build_runtime_context(dataset, "train")
        overlap = set(context.validation_unseen.tolist()) & set(context.test_unseen.tolist())
        assert overlap == set(), (
            f"{dataset}: a node in both splits makes the validation metric a "
            f"partial read of the test split: {sorted(overlap)}")


def test_the_split_is_deterministic_and_seed_dependent():
    pool = np.arange(40, dtype=np.int64)
    assert np.array_equal(split_unseen_pool(pool, 32)[0], split_unseen_pool(pool, 32)[0])
    assert not np.array_equal(split_unseen_pool(pool, 32)[0], split_unseen_pool(pool, 99)[0])


def test_every_phase_runs_on_the_full_node_axis(contexts):
    for phase, context in contexts.items():
        assert context.adjacency.shape == (context.num_nodes, context.num_nodes)
        assert context.node_ids.size == context.num_nodes, phase


def test_signal_scope_follows_the_phase(contexts):
    train, validation, test = (contexts[p] for p in ("train", "validation", "test"))
    known = set(train.known_nodes.tolist())
    assert set(train.signal_scope.tolist()) == known
    assert set(validation.signal_scope.tolist()) == known | set(
        validation.validation_unseen.tolist())
    assert set(test.signal_scope.tolist()) == known | set(test.test_unseen.tolist())
    # the crossed exclusions: neither split may read the other's nodes
    assert not set(validation.signal_scope.tolist()) & set(validation.test_unseen.tolist())
    assert not set(test.signal_scope.tolist()) & set(test.validation_unseen.tolist())


def test_training_scores_only_known_nodes(contexts):
    train = contexts["train"]
    scored = torch.nonzero(train.loss_scope_mask()).flatten().numpy()
    assert set(scored.tolist()) == set(train.known_nodes.tolist())


# --- the raw-signal counterfactual -------------------------------------------


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -1e30, 1e30])
def test_forbidden_raw_values_cannot_reach_the_visible_signal(contexts, poison):
    """The load-bearing test. Out-of-scope nodes are replaced, not multiplied.

    `raw * mask` would be arithmetically tempting and wrong: NaN * 0 is NaN, so a
    single missing measurement on a node the phase may not read would propagate
    through every downstream reduction.
    """
    for phase, context in contexts.items():
        raw, observed = _batch(context)
        clean_visible, clean_available = context.visible_signal(raw, observed)

        forbidden = torch.ones(context.num_nodes, dtype=torch.bool)
        forbidden[torch.from_numpy(context.signal_scope)] = False
        assert forbidden.any(), f"{phase} has no forbidden node; test is vacuous"

        poisoned = raw.clone()
        poisoned[:, :, forbidden, :] = poison
        poisoned_observed = observed.clone()
        poisoned_observed[:, :, forbidden] = True        # also flip missingness
        visible, available = context.visible_signal(poisoned, poisoned_observed)

        assert torch.isfinite(visible).all(), f"{phase}: {poison} leaked through"
        assert torch.equal(visible, clean_visible), f"{phase}: visible signal moved"
        assert torch.equal(available, clean_available)


def test_in_scope_values_do_reach_the_visible_signal(contexts):
    """Control: without this the invariance above could hold by erasing everything."""
    context = contexts["train"]
    raw, observed = _batch(context)
    base, _ = context.visible_signal(raw, observed)
    moved = raw.clone()
    moved[:, :, context.known_nodes, :] += 5.0
    assert not torch.equal(context.visible_signal(moved, observed)[0], base)


def test_an_unmasked_nan_on_an_in_scope_node_is_refused(contexts):
    """Fail closed: a NaN that survives masking is a defect, not a fill value."""
    context = contexts["train"]
    raw, observed = _batch(context)
    raw[0, 0, context.known_nodes[0], 0] = float("nan")
    observed[0, 0, context.known_nodes[0]] = True
    with pytest.raises(ContextRefused, match="not finite"):
        context.visible_signal(raw, observed)


# --- teacher forcing ----------------------------------------------------------


def test_forbidden_future_targets_never_reach_the_decoder(contexts):
    for phase, context in contexts.items():
        future = torch.randn(2, 6, context.num_nodes, 1)
        observed = torch.ones(2, 6, context.num_nodes, dtype=torch.bool)
        forbidden = torch.ones(context.num_nodes, dtype=torch.bool)
        forbidden[torch.from_numpy(context.signal_scope)] = False

        base = context.masked_decoder_input(future, observed)
        assert torch.equal(base[:, :, forbidden, :], torch.zeros_like(base[:, :, forbidden, :]))

        poisoned = future.clone()
        poisoned[:, :, forbidden, :] = 1e9
        assert torch.equal(context.masked_decoder_input(poisoned, observed), base), (
            f"{phase}: a held-out node's future value moved the decoder input")


def test_target_availability_is_not_the_loss_mask(contexts):
    """Distinct fields on purpose: a target can exist and still not be scorable."""
    context = contexts["train"]
    observed = torch.ones(2, 6, context.num_nodes, dtype=torch.bool)
    assert context.target_available_mask(observed).all()
    assert not context.loss_mask(observed).all()
    assert int(context.loss_mask(observed)[0, 0].sum()) == context.known_nodes.size


# --- identity -----------------------------------------------------------------


def test_a_module_that_resolved_a_different_graph_is_refused(contexts):
    context = contexts["train"]
    context.assert_matches(context.identity(), "self")
    wrong = dict(context.identity(), adjacency_sha256="0" * 64)
    with pytest.raises(ContextRefused, match="adjacency_sha256"):
        context.assert_matches(wrong, "ALFTemporal")


def test_a_tampered_payload_is_refused(tmp_path, monkeypatch):
    import json
    from lib import runtime_context as module

    source = module.MANIFEST_DIR / "PEMS08_run_manifest.json"
    manifest = json.loads(source.read_text("utf-8"))
    manifest["payload"]["sha256"] = "0" * 64
    (tmp_path / "PEMS08_run_manifest.json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ContextRefused, match="payload hash"):
        build_runtime_context("PEMS08", "train", manifest_dir=tmp_path)


def test_a_missing_manifest_fails_closed(tmp_path):
    with pytest.raises(ContextRefused, match="fails closed"):
        build_runtime_context("PEMS08", "train", manifest_dir=tmp_path)


def test_the_context_is_not_training_ready(contexts):
    for context in contexts.values():
        assert context.readiness["training_ready"] is False
        assert context.provisional_split is True, (
            "the V/T partition is provisional until Stage 2-B freezes it")
