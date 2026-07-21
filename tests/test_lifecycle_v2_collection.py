import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_lifecycle_v2_results",
        ROOT / "scripts" / "collect_lifecycle_v2_results.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    root,
    model,
    variant,
    seed,
    offset=0.0,
    protocol_hash="protocol-v2",
    formal=True,
):
    run_dir = root / f"{model}-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": 2,
        "protocol_hash": protocol_hash,
        "training_protocol_hash": "training-v2",
        "required_training_protocol_hash": "training-v2",
        "formal_lcbt_v2": formal,
        "lce_protocol_hash": "lce-v1",
        "ples_protocol_hash": "ples-v2",
        "model": model,
        "model_variant": variant,
        "seed": seed,
        "dataset": {"name": "METRLA"},
    }
    main = {
        name: {"MAPE": base + offset, "MAE": base / 2 + offset, "RMSE": base + 1 + offset}
        for name, base in {
            "Unseen": 16.0,
            "Steady": 9.0,
            "Join": 10.0,
            "Fault": 11.0,
            "LifeAvg": 11.5,
        }.items()
    }
    phases = {
        name: {"MAPE": 10.0 + index + offset, "MAE": 4.0 + offset, "RMSE": 7.0 + offset}
        for index, name in enumerate(("Cold", "Warm", "Ready", "Onset", "Decay", "Outage"))
    }
    for filename, table, metrics in (
        ("lce_main_summary.json", "LCE v1 main", main),
        ("lce_phase_summary.json", "LCE v1 phases", phases),
    ):
        with (run_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump({**common, "table": table, "metrics": metrics}, handle)
    return run_dir


def test_lce_collector_reports_three_seed_mean_std_and_checks_hashes():
    collector = _load_collector()
    root = ROOT / ".pytest_cache" / "lce_v2_collection"
    run_dirs = []
    for model, variant in (("ALFNet", "rf_l2g"), ("MoGERNN", "mg_l2")):
        for seed, offset in ((32, -1.0), (42, 0.0), (52, 1.0)):
            run_dirs.append(_write_run(root / variant, model, variant, seed, offset))

    summary = collector.aggregate_results(run_dirs, expected_seeds=(32, 42, 52))

    assert summary["seeds"] == [32, 42, 52]
    assert len(summary["models"]) == 2
    alf_life = next(
        row
        for row in summary["main_rows"]
        if row["model"] == "ALFNet" and row["group"] == "LifeAvg"
    )
    assert alf_life["MAPE_mean"] == pytest.approx(11.5)
    assert alf_life["MAPE_std"] == pytest.approx(1.0)

    bad = _write_run(root / "bad", "ALFNet", "rf_l2g", 52, protocol_hash="changed")
    with pytest.raises(ValueError, match="protocol hash"):
        collector.aggregate_results([run_dirs[0], run_dirs[1], bad], expected_seeds=(32, 42, 52))


def test_lce_collector_rejects_missing_or_duplicate_seeds():
    collector = _load_collector()
    root = ROOT / ".pytest_cache" / "lce_v2_collection_seed_check"
    seed32 = _write_run(root, "ALFNet", "rf_l2g", 32)
    seed42 = _write_run(root, "ALFNet", "rf_l2g", 42)

    with pytest.raises(ValueError, match="expected seeds"):
        collector.aggregate_results([seed32, seed42], expected_seeds=(32, 42, 52))
    with pytest.raises(ValueError, match="duplicate seed"):
        collector.aggregate_results([seed32, seed32, seed42], expected_seeds=(32, 42))


def test_lce_collector_rejects_diagnostic_legacy_checkpoints():
    collector = _load_collector()
    root = ROOT / ".pytest_cache" / "lce_v2_collection_provenance"
    diagnostic = _write_run(
        root,
        "ALFNet",
        "legacy-diagnostic",
        32,
        formal=False,
    )
    with pytest.raises(ValueError, match="diagnostic or legacy"):
        collector.aggregate_results([diagnostic], expected_seeds=(32,))
