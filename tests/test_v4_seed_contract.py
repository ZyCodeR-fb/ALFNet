import random

import numpy as np
import pytest
import torch

from experiments.v4_runner import RunnerRefused, build_v4_session
from lib.runtime_context import build_runtime_context
from lib.v4_seeds import (
    EpochWorkerSeeder,
    NAMESPACES,
    SEED_POLICY,
    derive_seed,
    seed_bundle,
)


CONFIG = "configs/MoGERNN/PEMS08.yaml"


def test_one_run_seed_has_distinct_stable_named_substreams():
    first = seed_bundle(32)
    second = seed_bundle(32)
    assert first == second
    assert first["policy"] == SEED_POLICY
    assert set(first["derived"]) == set(NAMESPACES)
    assert len(set(first["derived"].values())) == len(NAMESPACES)
    assert first["derived"]["node_split"] == 32
    assert seed_bundle(42) != first


def test_node_split_is_repeatable_and_changes_with_run_seed():
    a = build_runtime_context("PEMS08", "validation", seed=32)
    b = build_runtime_context("PEMS08", "validation", seed=32)
    c = build_runtime_context("PEMS08", "validation", seed=42)
    assert np.array_equal(a.validation_unseen, b.validation_unseen)
    assert not np.array_equal(a.validation_unseen, c.validation_unseen)
    assert not set(a.validation_unseen) & set(a.test_unseen)


def test_worker_seed_is_a_function_of_run_epoch_and_worker():
    seeder = EpochWorkerSeeder(32)
    seeder.set_epoch(4)
    seeder(3)
    first = (random.random(), np.random.rand(), torch.rand(()).item())
    seeder(3)
    second = (random.random(), np.random.rand(), torch.rand(()).item())
    assert first == second
    seeder(4)
    assert (random.random(), np.random.rand(), torch.rand(()).item()) != first


def test_session_records_complete_seed_contract_and_refuses_free_init_seed():
    session = build_v4_session(CONFIG, device="cpu", run_seed=32,
                               batch_size=1, num_workers=0)
    assert session.seed_contract == seed_bundle(32)
    assert session.args.seed == 32
    assert session.args.model_init_seed == derive_seed(32, "model_init")
    assert session.validation_dataset.validation_seed == 12026
    with pytest.raises(RunnerRefused, match="derived from run_seed"):
        build_v4_session(CONFIG, device="cpu", run_seed=32, init_seed=1234,
                         batch_size=1, num_workers=0)


def test_active_v4_refuses_any_stride_other_than_one(tmp_path):
    text = open(CONFIG, encoding="utf-8").read()
    assert "slide_step: 1" in text
    changed = tmp_path / "stride12.yaml"
    changed.write_text(text.replace("slide_step: 1", "slide_step: 12", 1),
                       encoding="utf-8")
    with pytest.raises(RunnerRefused, match="slide_step=1"):
        build_v4_session(changed, device="cpu", run_seed=32,
                         batch_size=1, num_workers=0)


def test_model_initialisation_now_changes_with_run_seed_but_repeats_exactly():
    def state(seed):
        session = build_v4_session(CONFIG, device="cpu", run_seed=seed,
                                   batch_size=1, num_workers=0)
        return {name: value.detach().clone()
                for name, value in session.model.state_dict().items()}

    a, b, c = state(32), state(32), state(42)
    assert a.keys() == b.keys() == c.keys()
    assert all(torch.equal(a[name], b[name]) for name in a)
    assert any(not torch.equal(a[name], c[name]) for name in a
               if torch.is_floating_point(a[name]))


def test_derived_seed_rejects_unknown_namespace():
    with pytest.raises(ValueError, match="unknown"):
        derive_seed(32, "mystery")
