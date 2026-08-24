"""The shared v4 runner: one schedule, one teacher-forcing policy, no test split.

The property these tests exist for is that ALFNet and MoGERNN see the *same*
run. Not similar -- the same origins, the same episode design, the same loss
mask and the same teacher-forcing decisions, so that any difference between the
two models is the model. Anything that made the schedule depend on batch size,
shuffle order, worker count or how much RNG a model happens to consume would
break that silently, which is why the policy is hashed from the sample's own
identity rather than drawn.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments import v4_runner
from experiments.v4_runner import (
    RunnerRefused,
    build_v4_session,
    compute_losses,
    early_stopping_metric,
    train_one_epoch,
    valid_positions,
    validate,
)
from lib.lcbt_v4 import (
    TEACHER_FORCING_POLICY,
    build_v4_teacher_forcing_mask,
    teacher_forcing_metadata,
    teacher_forcing_ratio,
)
from lib.runtime_context import build_runtime_context

PAYLOAD = "data_v4/PEMS08/raw_data.pkl"
CONFIGS = {"ALFNet": "configs/paper/v4/PEMS08_ALFNet.yaml",
           "MoGERNN": "configs/paper/v4/PEMS08_MoGERNN.yaml"}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def train_context():
    return build_runtime_context("PEMS08", "train")


@pytest.fixture(scope="module")
def train_dataset(train_context):
    return V4LifecycleDataset(train_context, PAYLOAD)


def _session(model_name, batch_size=2, **kwargs):
    return build_v4_session(CONFIGS[model_name], batch_size=batch_size, **kwargs)


def _design(batch):
    """The fields that describe *which measurement* a batch asks for."""
    return {
        "origin": batch["origin"].tolist(),
        "scenario": list(batch["scenario"]),
        "evidence_steps": batch["evidence_steps"].tolist(),
        "calendar_regime": list(batch["calendar_regime"]),
        "phase_id": batch["phase_id"].tolist(),
        "loss_mask": batch["loss_mask"].clone(),
        "target_cohort_mask": batch["target_cohort_mask"].clone(),
    }


# --- the teacher-forcing policy ------------------------------------------------


@pytest.mark.parametrize("epoch, expected", [(0, 1.0), (20, 0.5), (39, 0.025),
                                             (40, 0.0), (41, 0.0), (100, 0.0)])
def test_the_ratio_reproduces_the_v3_schedule(epoch, expected):
    assert teacher_forcing_ratio(epoch) == pytest.approx(expected)
    if 0 < epoch < 40:
        assert 0.0 < teacher_forcing_ratio(epoch) < 1.0


def test_the_policy_is_stable_across_processes():
    """SHA256, not Python's per-process salted `hash`.

    Only PYTHONHASHSEED varies between the runs; everything else is this
    process's environment, so a difference could only come from the salt.
    """
    from lib.lcbt_v4 import _uniform01

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lib.lcbt_v4 import _uniform01, TEACHER_FORCING_POLICY\n"
        "print(repr(_uniform01(TEACHER_FORCING_POLICY, 32, 3, 4096, 7)))\n" % str(ROOT)
    )
    here = _uniform01(TEACHER_FORCING_POLICY, 32, 3, 4096, 7)
    for salt in ("0", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": salt})
        assert result.returncode == 0, result.stderr
        assert float(result.stdout.strip()) == here, salt


def test_the_mask_follows_the_sample_not_the_batch(train_dataset):
    """Reorder the samples and change the batch size; the decisions travel."""
    indices = [0, 1, 2, 3, 4, 5]
    samples = [train_dataset[index] for index in indices]
    collate = torch.utils.data._utils.collate.default_collate

    straight = build_v4_teacher_forcing_mask(collate(samples), 32, 3, True)
    reversed_batch = collate(list(reversed(samples)))
    flipped = build_v4_teacher_forcing_mask(reversed_batch, 32, 3, True)
    assert torch.equal(straight, flipped.flip(0))

    # ... and split into three batches of two, then reassembled.
    pieces = [build_v4_teacher_forcing_mask(collate(samples[i:i + 2]), 32, 3, True)
              for i in (0, 2, 4)]
    assert torch.equal(straight, torch.cat(pieces, dim=0))

    # A different seed or epoch is a different schedule; otherwise this would
    # pass with a policy that ignored both.
    assert not torch.equal(straight,
                           build_v4_teacher_forcing_mask(collate(samples), 33, 3, True))
    assert not torch.equal(straight,
                           build_v4_teacher_forcing_mask(collate(samples), 32, 4, True))


def test_the_mask_is_a_subset_of_decoder_availability(train_dataset):
    collate = torch.utils.data._utils.collate.default_collate
    batch = collate([train_dataset[index] for index in range(4)])
    available = batch["decoder_available_mask"].to(torch.bool)
    for epoch in (0, 5, 20):
        mask = build_v4_teacher_forcing_mask(batch, 32, epoch, True)
        assert not bool((mask & ~available).any())
    assert bool(build_v4_teacher_forcing_mask(batch, 32, 0, True).any())


def test_validation_and_test_masks_are_empty(train_dataset):
    collate = torch.utils.data._utils.collate.default_collate
    batch = collate([train_dataset[index] for index in range(4)])
    for epoch in (0, 1, 39):
        mask = build_v4_teacher_forcing_mask(batch, 32, epoch, False)
        assert mask.dtype == torch.bool and not bool(mask.any())
    assert teacher_forcing_metadata(32, 0, False)["ratio"] == 0.0


def test_both_models_receive_the_identical_teacher_mask():
    """The mask is a function of the batch, so the two models cannot diverge."""
    masks, designs = {}, {}
    for name in sorted(CONFIGS):
        session = _session(name)
        session.train_dataset.set_epoch(2)
        loader = DataLoader(session.train_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        masks[name] = build_v4_teacher_forcing_mask(batch, session.run_seed, 2, True)
        designs[name] = _design(batch)
    assert torch.equal(masks["ALFNet"], masks["MoGERNN"])
    for key in designs["ALFNet"]:
        left, right = designs["ALFNet"][key], designs["MoGERNN"][key]
        if torch.is_tensor(left):
            assert torch.equal(left, right), key
        else:
            assert left == right, key


# --- the schedule --------------------------------------------------------------


def test_set_epoch_is_deterministic_and_moves_only_the_schedule(train_context):
    dataset = V4LifecycleDataset(train_context, PAYLOAD)

    def signature():
        return [(entry["scenario"], int(entry["evidence_steps"]),
                 tuple(int(node) for node in entry["changed_nodes"]))
                for entry in dataset.schedule[:8]]

    origins, scaler = dataset.origins.copy(), dict(dataset.scaler)
    first = signature()
    dataset.set_epoch(1)
    moved = signature()
    dataset.set_epoch(0)
    assert signature() == first
    assert moved != first
    dataset.set_epoch(1)
    assert signature() == moved
    # Nothing else moved.
    assert (dataset.origins == origins).all()
    assert dict(dataset.scaler) == scaler
    assert dataset.boundaries == V4LifecycleDataset(train_context, PAYLOAD).boundaries


def test_the_validation_schedule_ignores_the_training_epoch():
    context = build_runtime_context("PEMS08", "validation")
    train = V4LifecycleDataset(build_runtime_context("PEMS08", "train"), PAYLOAD)
    dataset = V4LifecycleDataset(context, PAYLOAD, scaler=train.scaler)
    before = dataset.schedule
    for epoch in (1, 7, 40):
        dataset.set_epoch(epoch)
        assert dataset.schedule is before
    assert dataset.validation_seed != dataset.seed


def test_the_two_datasets_share_one_training_fitted_scaler():
    session = _session("ALFNet")
    assert session.validation_dataset.scaler is session.train_dataset.scaler
    assert session.scaler is session.train_dataset.scaler
    assert session.scaler["fitted_on"] == "train window, training-known nodes only"
    # Independently: a scaler fitted on the validation dataset alone would be a
    # different object even if the numbers happened to agree today.
    alone = V4LifecycleDataset(session.validation_context, PAYLOAD)
    assert alone.scaler is not session.scaler


def test_both_models_see_the_same_design_at_the_same_epoch():
    sessions = {name: _session(name) for name in sorted(CONFIGS)}
    for epoch in (0, 3):
        designs = {}
        for name, session in sessions.items():
            session.train_dataset.set_epoch(epoch)
            loader = DataLoader(session.train_dataset, batch_size=4, shuffle=False)
            designs[name] = _design(next(iter(loader)))
        for key in designs["ALFNet"]:
            left, right = designs["ALFNet"][key], designs["MoGERNN"][key]
            if torch.is_tensor(left):
                assert torch.equal(left, right), (epoch, key)
            else:
                assert left == right, (epoch, key)


# --- the runner ----------------------------------------------------------------


def test_the_runner_refuses_to_touch_a_test_split(monkeypatch):
    with pytest.raises(RunnerRefused, match="will not build a 'test' context"):
        v4_runner.resolve_context("PEMS08", "test", 32)

    original = v4_runner.build_runtime_context

    def guarded(dataset, phase, **kwargs):
        assert phase != "test", "the runner opened a test context"
        return original(dataset, phase, **kwargs)

    monkeypatch.setattr(v4_runner, "build_runtime_context", guarded)
    session = _session("MoGERNN")
    assert {session.train_context.phase, session.validation_context.phase} == \
           {"train", "validation"}
    assert not hasattr(session, "test_loader")


def test_the_session_is_built_data_first():
    session = _session("ALFNet")
    assert session.dataset == "PEMS08" and session.model_name == "ALFNet"
    assert session.train_context.static_asset_hash() == \
           session.validation_context.static_asset_hash()
    assert session.train_context.exposure_hash() != \
           session.validation_context.exposure_hash()
    assert session.model.static_asset_hash == session.train_context.static_asset_hash()
    assert session.args.lifecycle_training_strategy == "lcbt_v4"


@pytest.mark.parametrize("who", sorted(CONFIGS))
def test_the_loss_uses_exactly_the_loaders_mask(who):
    """Outside `loss_mask` the target may be anything at all."""
    session = _session(who, batch_size=1)
    batch = next(iter(DataLoader(session.train_dataset, batch_size=2, shuffle=False)))
    mask = batch["loss_mask"].unsqueeze(-1)
    assert bool(mask.any()) and bool((~mask).any())

    session.model.train()

    def run(target):
        torch.manual_seed(11)
        session.model.zero_grad(set_to_none=True)
        losses = compute_losses(
            session.model.forward_v4(dict(batch, target=target),
                                     session.train_context, teacher_forcing_mask=None),
            batch if target is batch["target"] else dict(batch, target=target))
        losses["optimize"].backward()
        return (float(losses["base"]),
                {name: (None if p.grad is None else p.grad.detach().clone())
                 for name, p in session.model.named_parameters()})

    reference = run(batch["target"])
    for poison in (float("nan"), float("inf"), 1.0e30):
        target = batch["target"].clone()
        target[~mask.expand_as(target)] = poison
        loss, gradients = run(target)
        assert loss == reference[0], poison
        for name in gradients:
            if gradients[name] is not None:
                assert torch.equal(gradients[name], reference[1][name]), (poison, name)

    # The non-empty control: inside the mask it does matter.
    target = batch["target"].clone()
    target[mask.expand_as(target)] = target[mask.expand_as(target)] + 3.0
    assert run(target)[0] != reference[0]


def test_valid_positions_is_the_mask_and_finiteness():
    session = _session("MoGERNN")
    batch = next(iter(DataLoader(session.train_dataset, batch_size=2, shuffle=False)))
    prediction = torch.zeros_like(batch["target"])
    assert torch.equal(valid_positions(batch, prediction),
                       batch["loss_mask"].unsqueeze(-1))
    prediction[0, 0, 0] = float("nan")
    assert not bool(valid_positions(batch, prediction)[0, 0, 0].any())


@pytest.mark.parametrize("who", sorted(CONFIGS))
def test_validation_updates_nothing(who):
    session = _session(who, batch_size=1)
    optimizer = torch.optim.AdamW(session.model.parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in session.model.named_parameters()}
    report = validate(session, epoch=0, max_batches=2)
    after = {name: p.detach().clone() for name, p in session.model.named_parameters()}
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert optimizer.state_dict()["state"] == {}
    assert report["teacher_forcing_ratio"] == 0.0
    assert report["count"] > 0 and report["scenarios_present"] >= 1
    assert early_stopping_metric(report) == report["life_avg_mae"]
    for key in ("mae", "mse", "life_avg_mae"):
        assert report[key] == report[key] and report[key] < float("inf"), key


@pytest.mark.parametrize("who", sorted(CONFIGS))
def test_one_training_step_reproduces(who):
    results = []
    for _ in range(2):
        session = _session(who, batch_size=1)
        optimizer = torch.optim.AdamW(session.model.parameters(), lr=1e-3)
        report = train_one_epoch(session, optimizer, 0, max_batches=1)
        results.append((report["base"], report["optimize"],
                        {name: p.detach().clone()
                         for name, p in session.model.state_dict().items()}))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]
    for name in results[0][2]:
        assert torch.equal(results[0][2][name], results[1][2][name]), name


def test_only_alfnet_reports_a_role_mix_term():
    """MoGERNN declares no `forecast_loss_mode`, and none is invented for it."""
    reports = {}
    for name in sorted(CONFIGS):
        session = _session(name)
        optimizer = torch.optim.AdamW(session.model.parameters(), lr=1e-3)
        reports[name] = train_one_epoch(session, optimizer, 0, max_batches=2)
    assert reports["MoGERNN"]["role_mix"] is None
    assert reports["MoGERNN"]["aux"] is None
    assert reports["MoGERNN"]["optimize"] == reports["MoGERNN"]["base"]
    assert reports["ALFNet"]["role_mix"] is not None
    for report in reports.values():
        assert report["base"] == report["base"] and report["base"] < float("inf")
        assert report["teacher_forcing_policy"] == TEACHER_FORCING_POLICY
        assert report["ratio"] == 1.0 and report["epoch"] == 0


def test_an_epoch_with_no_valid_position_is_an_error_not_a_zero(monkeypatch):
    session = _session("MoGERNN")
    optimizer = torch.optim.AdamW(session.model.parameters(), lr=1e-3)
    monkeypatch.setattr(v4_runner, "valid_positions",
                        lambda batch, prediction: torch.zeros_like(
                            batch["loss_mask"].unsqueeze(-1)))
    with pytest.raises(RunnerRefused, match="refusing to report a zero"):
        train_one_epoch(session, optimizer, 0, max_batches=2)
    with pytest.raises(RunnerRefused, match="refusing to report a zero"):
        validate(session, 0, max_batches=2)


def test_an_unknown_model_is_refused(monkeypatch):
    monkeypatch.setitem(v4_runner.MODEL_ENTRY_POINTS, "ALFNet", None)
    monkeypatch.delitem(v4_runner.MODEL_ENTRY_POINTS, "ALFNet")
    with pytest.raises(RunnerRefused, match="no v4 entry point"):
        build_v4_session(CONFIGS["ALFNet"])


def test_the_runner_uses_the_declared_v4_strategy():
    assert _session("ALFNet").args.lifecycle_training_strategy == "lcbt_v4"


# --- the smoke, as a test ------------------------------------------------------


@pytest.mark.parametrize("who", sorted(CONFIGS))
def test_the_pems08_cpu_smoke_runs_and_moves_parameters(who):
    """Two train batches, two validation batches, one epoch, CPU.

    A plumbing check and nothing else: two batches of an untrained model say
    nothing about either architecture, and no number from here is comparable
    with anything.
    """
    session = _session(who, batch_size=2, num_workers=0)
    optimizer = torch.optim.AdamW(session.model.parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in session.model.named_parameters()}

    report = train_one_epoch(session, optimizer, 0, max_batches=2)
    assert report["batches"] == 2 and report["skipped"] == 0
    for key in ("base", "optimize"):
        assert report[key] == report[key] and abs(report[key]) < float("inf"), key

    moved = [name for name, p in session.model.named_parameters()
             if not torch.equal(before[name], p.detach())]
    assert moved, "the optimiser step changed nothing"

    validation = validate(session, epoch=0, max_batches=2)
    assert validation["batches"] == 2
    assert validation["life_avg_mae"] == validation["life_avg_mae"]
    assert validation["teacher_forcing_ratio"] == 0.0
