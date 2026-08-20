"""The 36 development cells: arithmetic, coverage, and what they refuse to be.

A table with six main rows, six phase rows and three metrics is the easiest
object in this repo to mistake for a result. So the checks here are as much
about labelling and refusal as about arithmetic: the collector must reject a
test context, must never quietly report 35 cells, and must not let the words
`final`, `confirmation` or `test` into its schema.

The arithmetic is checked twice over -- once by hand on numbers small enough to
verify on paper, and once against the frozen `ples_protocol` aggregation applied
to rows built independently of the collector.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from data_provider.ples_protocol import (
    PHASE_ORDER,
    _calendar_balanced_metrics_v2,
    _tau_balanced_metrics_v2,
)
from experiments.v4_validation_collector import (
    METRICS,
    PHASE_ID_TO_NAME,
    CollectorRefused,
    ValidationCellCollector,
    aggregate_seeds,
    cell_count,
)
from lib.runtime_context import build_runtime_context

ROOT = Path(__file__).resolve().parents[1]
SCALER = {"mean": 100.0, "std": 10.0,
          "fitted_on": "train window, training-known nodes only", "n_values": 1}
REGIME = "Weekday/EarlyAM"


@pytest.fixture(scope="module")
def validation_context():
    return build_runtime_context("PEMS08", "validation")


def _batch(entries, nodes=2, horizon=2):
    """One synthetic validation batch.

    `entries` is a list of (scenario, phase_id, evidence_steps, origin, regime,
    target_value, prediction_value) in *normalised* units, with every node and
    horizon step masked in.
    """
    size = len(entries)
    target = torch.zeros(size, horizon, nodes, 1)
    prediction = torch.zeros(size, horizon, nodes, 1)
    for index, entry in enumerate(entries):
        target[index] = entry[5]
        prediction[index] = entry[6]
    batch = {
        "target": target,
        "loss_mask": torch.ones(size, horizon, nodes, dtype=torch.bool),
        "scenario": [entry[0] for entry in entries],
        "phase_id": torch.tensor([entry[1] for entry in entries]),
        "evidence_steps": torch.tensor([entry[2] for entry in entries]),
        "origin": torch.tensor([entry[3] for entry in entries]),
        "calendar_regime": [entry[4] for entry in entries],
    }
    return batch, prediction


def _full_batch():
    """Every main group and every phase represented at least once."""
    return _batch([
        ("zero_history", -1, 0, 10, REGIME, 1.0, 1.5),
        ("steady", -1, 0, 11, REGIME, 2.0, 1.0),
        ("join", 0, 1, 12, REGIME, 1.0, 2.0),
        ("join", 1, 6, 13, REGIME, 1.0, 1.25),
        ("join", 2, 12, 14, REGIME, 2.0, 2.5),
        ("fault", 3, 1, 15, REGIME, 1.0, 0.5),
        ("fault", 4, 6, 16, REGIME, 3.0, 2.0),
        ("fault", 5, 12, 17, REGIME, 1.0, 1.75),
    ])


def _collect(context, entries=None):
    batch, prediction = _full_batch() if entries is None else _batch(entries)
    collector = ValidationCellCollector(context, SCALER)
    collector.update(batch, prediction)
    return collector


# --- arithmetic ----------------------------------------------------------------


def test_the_metrics_match_a_hand_computation(validation_context):
    """One group, one k, one regime: nothing to average over but the cells."""
    # normalised target 1.0 -> 110.0, prediction 1.5 -> 115.0, error 5.0
    collector = _collect(validation_context,
                         [("zero_history", -1, 0, 10, REGIME, 1.0, 1.5)] )
    cell = collector.rows["main"]["Unseen"][0]
    assert cell["count"] == 4                      # 2 horizon steps x 2 nodes
    assert cell["treated_ae_sum"] == pytest.approx(4 * 5.0)
    assert cell["treated_se_sum"] == pytest.approx(4 * 25.0)
    assert cell["treated_ape_sum"] == pytest.approx(4 * 5.0 / 110.0)

    metrics = _tau_balanced_metrics_v2(collector.rows["main"]["Unseen"], "treated")
    assert metrics["MAE"] == pytest.approx(5.0)
    assert metrics["RMSE"] == pytest.approx(5.0)
    assert metrics["MAPE"] == pytest.approx(5.0 / 110.0 * 100.0)


def test_the_table_agrees_with_the_frozen_aggregation(validation_context):
    """Rows rebuilt independently, then run through ples_protocol directly."""
    collector = _collect(validation_context)
    table = collector.table()
    for name in ("Unseen", "Steady", "Join", "Fault"):
        expected = _tau_balanced_metrics_v2(collector.rows["main"][name], "treated")
        for metric in METRICS:
            assert table["main"][name][metric] == pytest.approx(expected[metric])
    for name in PHASE_ORDER:
        expected = _tau_balanced_metrics_v2(collector.rows["phase"][name], "treated")
        for metric in METRICS:
            assert table["phase"][name][metric] == pytest.approx(expected[metric])
    # LifeAvg is the mean of the four, WorstGroup their max -- frozen definitions.
    for metric in METRICS:
        four = [table["main"][name][metric] for name in
                ("Unseen", "Steady", "Join", "Fault")]
        assert table["main"]["LifeAvg"][metric] == pytest.approx(float(np.mean(four)))
        assert table["main"]["WorstGroup"][metric] == pytest.approx(max(four))


def test_the_calendar_weighting_is_the_frozen_one(validation_context):
    """Two regimes with different errors: the 5/7-2/7 shares must show.

    The other rows are present only so the table can be built at all -- every
    row must exist or the collector refuses, which is the point of the coverage
    test below.
    """
    entries = [
        ("steady", -1, 0, 10, "Weekday/EarlyAM", 1.0, 2.0),   # error 10
        ("steady", -1, 0, 11, "Weekend/EarlyAM", 1.0, 3.0),   # error 20
        ("zero_history", -1, 0, 12, REGIME, 1.0, 1.5),
        ("join", 0, 1, 13, REGIME, 1.0, 2.0),
        ("join", 1, 6, 14, REGIME, 1.0, 1.25),
        ("join", 2, 12, 15, REGIME, 2.0, 2.5),
        ("fault", 3, 1, 16, REGIME, 1.0, 0.5),
        ("fault", 4, 6, 17, REGIME, 3.0, 2.0),
        ("fault", 5, 12, 18, REGIME, 1.0, 1.75),
    ]
    collector = _collect(validation_context, entries)
    expected = _calendar_balanced_metrics_v2(collector.rows["main"]["Steady"],
                                             "treated")
    table = collector.table()
    assert table["main"]["Steady"]["MAE"] == pytest.approx(expected["MAE"])
    # (5/7 * 10 + 2/7 * 20) / (5/7 + 2/7), not the unweighted 15.
    assert table["main"]["Steady"]["MAE"] == pytest.approx(90.0 / 7.0)
    assert table["main"]["Steady"]["MAE"] != pytest.approx(15.0)


# --- coverage ------------------------------------------------------------------


def test_the_phase_mapping_is_complete_and_unique():
    assert sorted(PHASE_ID_TO_NAME) == [0, 1, 2, 3, 4, 5]
    assert len(set(PHASE_ID_TO_NAME.values())) == 6
    assert set(PHASE_ID_TO_NAME.values()) == set(PHASE_ORDER)


def test_the_table_is_exactly_thirty_six_cells(validation_context):
    table = _collect(validation_context).table()
    assert len(table["main"]) == 6 and len(table["phase"]) == 6
    assert cell_count(table) == 36
    assert table["development_validation_only"] is True
    assert table["split"] == "validation"
    for section in ("main", "phase"):
        for row, cell in table[section].items():
            assert set(METRICS) <= set(cell), (section, row)
            assert cell["finite"] is True
            assert cell["valid_target_count"] > 0
            assert cell["origin_count"] > 0 and cell["sample_count"] > 0
            for metric in METRICS:
                assert math.isfinite(cell[metric])


def test_the_table_is_json_safe_with_allow_nan_false(validation_context):
    table = _collect(validation_context).table()
    text = json.dumps(table, allow_nan=False, sort_keys=True)
    assert "NaN" not in text and "Infinity" not in text


def test_the_schema_never_calls_itself_final_or_a_test(validation_context):
    table = _collect(validation_context).table()
    lowered = json.dumps({k: v for k, v in table.items() if k != "note"}).lower()
    for word in ("\"final", "confirmation", "\"test"):
        assert word not in lowered, word


# --- fail closed ---------------------------------------------------------------


@pytest.mark.parametrize("phase", ["train", "test"])
def test_a_non_validation_context_is_refused(phase):
    with pytest.raises(CollectorRefused, match="validation"):
        ValidationCellCollector(build_runtime_context("PEMS08", phase), SCALER)


def test_a_missing_row_fails_closed_rather_than_shrinking(validation_context):
    """Seven of the eight entries: the table must raise, not report 33 cells."""
    collector = _collect(validation_context, [
        ("zero_history", -1, 0, 10, REGIME, 1.0, 1.5),
        ("steady", -1, 0, 11, REGIME, 2.0, 1.0),
        ("join", 0, 1, 12, REGIME, 1.0, 2.0),
        ("fault", 3, 1, 15, REGIME, 1.0, 0.5),
    ])
    with pytest.raises(CollectorRefused, match="no valid cell|no valid cells"):
        collector.table()


def test_an_empty_collector_fails_closed(validation_context):
    collector = ValidationCellCollector(validation_context, SCALER)
    with pytest.raises(CollectorRefused):
        collector.table()


def test_a_shape_mismatch_is_refused(validation_context):
    batch, prediction = _full_batch()
    collector = ValidationCellCollector(validation_context, SCALER)
    with pytest.raises(CollectorRefused, match="does not match target"):
        collector.update(batch, prediction[:, :1])


def test_a_non_finite_cell_is_refused(validation_context, monkeypatch):
    collector = _collect(validation_context)
    collector.rows["main"]["Steady"][0]["treated_ae_sum"] = float("inf")
    with pytest.raises(CollectorRefused):
        collector.table()


# --- the mask decides which targets count --------------------------------------


def test_targets_outside_the_mask_move_no_cell(validation_context):
    batch, prediction = _full_batch()
    mask = batch["loss_mask"].clone()
    mask[:, 0] = False                          # half the horizon excluded
    batch = dict(batch, loss_mask=mask)

    def table_for(target):
        collector = ValidationCellCollector(validation_context, SCALER)
        collector.update(dict(batch, target=target), prediction)
        return collector.table()

    reference = table_for(batch["target"])
    for poison in (float("nan"), float("inf"), 1.0e30):
        target = batch["target"].clone()
        target[~mask.unsqueeze(-1).expand_as(target)] = poison
        moved = table_for(target)
        for section in ("main", "phase"):
            for row in reference[section]:
                for metric in METRICS:
                    assert moved[section][row][metric] == \
                           pytest.approx(reference[section][row][metric]), \
                           (poison, section, row, metric)


def test_a_target_inside_the_mask_does_move_its_cell(validation_context):
    """The non-empty control for the test above."""
    batch, prediction = _full_batch()

    def table_for(target):
        collector = ValidationCellCollector(validation_context, SCALER)
        collector.update(dict(batch, target=target), prediction)
        return collector.table()

    reference = table_for(batch["target"])
    target = batch["target"].clone()
    target[1] = target[1] + 4.0                 # the Steady sample only
    moved = table_for(target)
    assert moved["main"]["Steady"]["MAE"] != pytest.approx(
        reference["main"]["Steady"]["MAE"])
    assert moved["main"]["Unseen"]["MAE"] == pytest.approx(
        reference["main"]["Unseen"]["MAE"])
    assert moved["main"]["LifeAvg"]["MAE"] != pytest.approx(
        reference["main"]["LifeAvg"]["MAE"])


# --- three-seed aggregation ----------------------------------------------------


def test_the_seed_aggregate_uses_sample_std(validation_context):
    tables = {}
    for offset, seed in enumerate((32, 42, 52)):
        batch, prediction = _full_batch()
        collector = ValidationCellCollector(validation_context, SCALER)
        collector.update(batch, prediction + 0.1 * offset)
        tables[seed] = collector.table()
    summary = aggregate_seeds(tables)

    assert summary["seeds"] == [32, 42, 52] and summary["n"] == 3
    assert summary["std_ddof"] == 1
    assert summary["development_validation_only"] is True
    values = [tables[seed]["main"]["Unseen"]["MAE"] for seed in (32, 42, 52)]
    cell = summary["main"]["Unseen"]["MAE"]
    assert cell["mean"] == pytest.approx(float(np.mean(values)))
    assert cell["std"] == pytest.approx(float(np.std(values, ddof=1)))
    # ddof=0 would be smaller; at n=3 by a factor sqrt(2/3).
    assert cell["std"] != pytest.approx(float(np.std(values, ddof=0)))
    assert cell["per_seed"] == {"32": values[0], "42": values[1], "52": values[2]}
    assert summary["main"]["Unseen"]["valid_target_count_agrees"] is True
    json.dumps(summary, allow_nan=False)


def test_the_seed_aggregate_reports_disagreeing_counts(validation_context):
    tables = {}
    for seed in (32, 42):
        collector = _collect(validation_context)
        tables[seed] = collector.table()
    tables[42]["main"]["Unseen"]["valid_target_count"] += 1
    summary = aggregate_seeds(tables)
    assert summary["main"]["Unseen"]["valid_target_count_agrees"] is False


# --- the plan ------------------------------------------------------------------


def test_the_plan_is_registered_and_still_holds():
    from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan

    plan = assert_plan_still_holds(load_plan())
    assert plan["status"] == "preregistered_development"
    assert plan["datasets"] == ["PEMS08", "PEMSBAY"]
    assert plan["seeds"] == [32, 42, 52]
    assert plan["trial_budget"] == 12 and len(plan["configs"]) == 4
    assert plan["test_access"] == "forbidden"
    assert plan["phases_allowed"] == ["train", "validation"]
    assert plan["training"]["checkpoint_metric"] == "validation life_avg_mae"
    assert plan["training"]["tie_rule"] == "keep_earlier_epoch"
    assert plan["next_round"]["max_candidate_families"] == 2
    assert plan["next_round"]["max_trials_per_model"] == 4
    # The held-out pair appears only in the prohibition that names them; never
    # as a dataset, a config or a data asset this plan may read.
    reachable = json.dumps({key: plan[key] for key in
                            ("datasets", "configs", "data", "models")})
    assert "PEMS03" not in reachable and "PEMS04" not in reachable
    assert any("PEMS03" in line and "PEMS04" in line for line in plan["prohibited"])


def test_an_edited_plan_fails_closed(tmp_path, monkeypatch):
    from scripts import build_v4_development_plan as builder

    plan = builder.load_plan()
    tampered = dict(plan, seeds=[32])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(tampered, indent=1, sort_keys=True), "utf-8")
    monkeypatch.setattr(builder, "PLAN_PATH", path)
    with pytest.raises(ValueError, match="does not describe its contents"):
        builder.load_plan()


def test_a_config_the_plan_did_not_register_is_refused():
    from scripts import train_v4

    with pytest.raises(Exception, match="development plan registered"):
        train_v4.resolve_plan(ROOT / "configs/paper/v4/PEMS03_ALFNet.yaml")


def test_a_drifted_config_hash_stops_the_run(monkeypatch):
    from scripts import build_v4_development_plan as builder
    from scripts import train_v4

    plan = builder.load_plan()
    drifted = json.loads(json.dumps(plan))
    drifted["configs"]["PEMS08_ALFNet"]["sha256"] = "0" * 64
    monkeypatch.setattr(train_v4, "load_plan", lambda: drifted)
    monkeypatch.setattr(train_v4, "assert_plan_still_holds", lambda p: p)
    with pytest.raises(Exception, match="the plan registered"):
        train_v4.train(ROOT / "configs/paper/v4/PEMS08_ALFNet.yaml",
                       output_root=None, device="cpu", max_epochs=1,
                       max_train_batches=1, max_validation_batches=1,
                       batch_size=1, num_workers=0, save=False,
                       collect_development_table=True)
