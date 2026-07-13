import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "lifecycle_stress_map.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("lifecycle_stress_map", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_results(scale=1.0, include_routing=False):
    routing = None
    if include_routing:
        routing = {
            "VS": {"alpha_mean": 1.0},
            "AAS": {"alpha_mean": 0.8},
            "NAS": {"alpha_mean": 1.0},
            "FS": {"alpha_mean": 0.7},
        }
    return {
        "config": {"input_len": 12, "slide_step": 12, "async_commissioning_k": 12},
        "event_plan": [
            {
                "event_id": 1,
                "event_time_index": 120,
                "evaluation_window_range": [11, 14],
                "nas_nodes": [1, 2],
                "fs_nodes": [3],
                "aas_nodes": [4, 5],
                "vs_nodes": [6],
            }
        ],
        "event_metrics": {
            1: {
                "NAS": {
                    "metrics": None,
                    "window_metrics": [
                        {
                            "window_index": 11,
                            "window_offset": 0,
                            "alpha_mean": 0.6,
                            "MAPE": 10.0 * scale,
                            "MAE": 1.0,
                            "RMSE": 2.0,
                        },
                        {"window_index": 14, "window_offset": 3, "MAPE": 8.0 * scale, "MAE": 1.0, "RMSE": 2.0},
                    ],
                },
                "FS": {
                    "metrics": None,
                    "window_metrics": [
                        {"window_index": 11, "window_offset": 0, "MAPE": 20.0 * scale, "MAE": 1.0, "RMSE": 2.0}
                    ],
                },
                "AAS": {"metrics": None, "window_metrics": []},
                "VS": {
                    "metrics": None,
                    "window_metrics": [
                        {"window_index": 11, "window_offset": 0, "MAPE": 30.0 * scale, "MAE": 1.0, "RMSE": 2.0}
                    ],
                },
            }
        },
        "routing_diagnostics": routing,
    }


def test_build_stress_rows_derives_sensor_state_and_visibility():
    stress = _load_script_module()

    rows = stress.build_stress_rows(_sample_results(include_routing=True), model_name="ours")

    by_role = {(row["role"], row["window_offset"]): row for row in rows}
    assert by_role[("NAS", 0)]["sensor_state"] == "Warm-up"
    assert by_role[("NAS", 3)]["sensor_state"] == "Adapted"
    assert by_role[("FS", 0)]["sensor_state"] == "Degrading"
    assert by_role[("VS", 0)]["sensor_state"] == "Cold-start"
    assert by_role[("NAS", 0)]["visibility"] == 1.0
    assert by_role[("NAS", 0)]["lifecycle_age"] == 0.0
    assert by_role[("NAS", 0)]["history_steps"] == 12.0
    assert by_role[("NAS", 0)]["patch_residual_weight"] == 0.4
    assert by_role[("FS", 0)]["visibility"] == 0.0
    assert by_role[("AAS", 0)]["graph_weight"] is None if ("AAS", 0) in by_role else True
    assert by_role[("FS", 0)]["patch_residual_weight"] == 0.30000000000000004


def test_align_and_aggregate_grid_computes_relative_improvement():
    stress = _load_script_module()
    ours = stress.build_stress_rows(_sample_results(scale=0.5, include_routing=True), model_name="ALFNet")
    baseline = stress.build_stress_rows(_sample_results(scale=1.0), model_name="MoGERNN")

    aligned = stress.align_with_baseline(ours, baseline)
    grid = stress.aggregate_grid(
        aligned,
        age_bins=[0, 2, 10],
        visibility_bins=[0.0, 0.5, 1.0],
        x_axis="lifecycle_age",
    )

    assert aligned[0]["baseline_mape"] == aligned[0]["mape"] * 2
    assert aligned[0]["improvement_pct"] == 50.0
    assert any(cell["improvement_pct"] == 50.0 for cell in grid)
    assert any(cell["patch_residual_weight"] is not None for cell in grid)


def test_aggregate_grid_uses_ratio_of_cell_means_for_improvement():
    stress = _load_script_module()
    rows = [
        {
            "lifecycle_age": 0.0,
            "visibility": 0.2,
            "mape": 5.0,
            "mae": 1.0,
            "rmse": 2.0,
            "baseline_mape": 10.0,
            "improvement_pct": 50.0,
            "graph_weight": 0.8,
            "patch_residual_weight": 0.2,
        },
        {
            "lifecycle_age": 0.0,
            "visibility": 0.2,
            "mape": 9.0,
            "mae": 1.0,
            "rmse": 2.0,
            "baseline_mape": 10.0,
            "improvement_pct": 10.0,
            "graph_weight": 0.6,
            "patch_residual_weight": 0.4,
        },
    ]
    grid = stress.aggregate_grid(
        rows,
        age_bins=[0, 1],
        visibility_bins=[0.0, 0.5],
        x_axis="lifecycle_age",
    )

    assert grid[0]["age_bin"] == "0"
    assert grid[0]["visibility_bin"] == "0-50%"
    assert grid[0]["improvement_pct"] == 30.0


def test_idw_fill_preserves_measured_values_and_fills_sparse_cells():
    stress = _load_script_module()
    matrix = [
        [20.0, None, 10.0],
        [None, None, None],
        [-10.0, None, 5.0],
    ]

    filled = stress.idw_fill_matrix(matrix)

    assert filled[0][0] == 20.0
    assert filled[2][0] == -10.0
    assert math.isfinite(filled[1][1])


def test_build_role_trajectory_aggregates_beta_by_age_band():
    stress = _load_script_module()
    rows = [
        {
            "role": "NAS",
            "lifecycle_age": 0.0,
            "visibility": 0.10,
            "improvement_pct": 4.0,
            "patch_residual_weight": 0.10,
            "graph_weight": 0.90,
        },
        {
            "role": "NAS",
            "lifecycle_age": 2.0,
            "visibility": 0.30,
            "improvement_pct": 8.0,
            "patch_residual_weight": 0.30,
            "graph_weight": 0.70,
        },
        {
            "role": "NAS",
            "lifecycle_age": 6.0,
            "visibility": 0.80,
            "improvement_pct": 12.0,
            "patch_residual_weight": 0.50,
            "graph_weight": 0.50,
        },
    ]

    points = stress.build_role_trajectory(
        rows,
        role="NAS",
        age_bins=[0, 4, 10],
        x_axis="lifecycle_age",
        overlay_key="patch_residual_weight",
    )

    assert len(points) == 2
    assert points[0]["x"] == 0.5
    assert points[0]["visibility"] == pytest.approx(0.20)
    assert points[0]["overlay_weight"] == pytest.approx(0.20)
    assert points[1]["x"] == 1.5
    assert points[1]["visibility"] == pytest.approx(0.80)


def test_plot_paper_stress_map_writes_a_nonempty_figure():
    age_bins = [0, 4, 10, math.inf]
    visibility_bins = [0.0, 0.25, 0.75, 1.0]
    cells = [
        {
            "age_bin": "0-3",
            "visibility_bin": "0-25%",
            "count": 8,
            "improvement_pct": 20.0,
            "mape": 12.0,
            "graph_weight": 0.9,
            "patch_residual_weight": 0.1,
        },
        {
            "age_bin": "10+",
            "visibility_bin": "75-100%",
            "count": 24,
            "improvement_pct": 8.0,
            "mape": 9.0,
            "graph_weight": 0.8,
            "patch_residual_weight": 0.2,
        },
    ]
    rows = [
        {
            "role": "NAS",
            "lifecycle_age": 0.0,
            "visibility": 0.10,
            "improvement_pct": 20.0,
            "patch_residual_weight": 0.1,
            "graph_weight": 0.9,
        },
        {
            "role": "NAS",
            "lifecycle_age": 12.0,
            "visibility": 0.90,
            "improvement_pct": 8.0,
            "patch_residual_weight": 0.2,
            "graph_weight": 0.8,
        },
    ]
    output_path = ROOT / ".pytest_cache" / "paper_stress_map.png"
    plot_script = f"""
import importlib.util
from pathlib import Path

inf = float("inf")
spec = importlib.util.spec_from_file_location("lifecycle_stress_map_plot_test", Path({str(SCRIPT_PATH)!r}))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
plotted = stress.plot_paper_stress_map(
    {cells!r},
    {rows!r},
    {age_bins!r},
    {visibility_bins!r},
    value_key="improvement_pct",
    output_path=Path({str(output_path)!r}),
    title="Lifecycle Stress Map",
    x_axis="lifecycle_age",
    overlay_key="patch_residual_weight",
)
raise SystemExit(0 if plotted else 1)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", plot_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert output_path.stat().st_size > 1000
    finally:
        output_path.unlink(missing_ok=True)


def test_aggregate_transition_profile_uses_all_events_in_each_age_block():
    stress = _load_script_module()
    rows_by_model = {
        "ALFNet": [
            {"event_id": 1, "role": "NAS", "lifecycle_age": 0, "mape": 10.0},
            {"event_id": 1, "role": "NAS", "lifecycle_age": 1, "mape": 12.0},
            {"event_id": 2, "role": "NAS", "lifecycle_age": 0, "mape": 14.0},
            {"event_id": 2, "role": "NAS", "lifecycle_age": 1, "mape": 16.0},
        ],
        "MoGERNN": [
            {"event_id": 1, "role": "NAS", "lifecycle_age": 0, "mape": 12.0},
            {"event_id": 1, "role": "NAS", "lifecycle_age": 1, "mape": 14.0},
            {"event_id": 2, "role": "NAS", "lifecycle_age": 0, "mape": 16.0},
            {"event_id": 2, "role": "NAS", "lifecycle_age": 1, "mape": 18.0},
        ],
    }

    profile = stress.aggregate_transition_profile(rows_by_model, roles=["NAS"], block_size=2)

    alf = next(row for row in profile if row["model"] == "ALFNet")
    assert alf["age_start"] == 0
    assert alf["age_end"] == 1
    assert alf["event_count"] == 2
    assert alf["window_count"] == 4
    assert alf["mape"] == pytest.approx(13.0)
    assert alf["mape_q25"] < alf["mape"] < alf["mape_q75"]


def test_summarize_transition_profile_reports_integrated_gain_and_win_rate():
    stress = _load_script_module()
    profile = []
    values = {
        "ALFNet": [10.0, 8.0],
        "MoGERNN": [12.0, 10.0],
        "PatchSTG": [11.0, 9.0],
    }
    for model, mapes in values.items():
        for block, mape in enumerate(mapes):
            profile.append(
                {
                    "model": model,
                    "role": "NAS",
                    "age_block": block,
                    "age_start": block * 6,
                    "age_end": block * 6 + 5,
                    "age_center": block * 6 + 2.5,
                    "age_width": 6,
                    "mape": mape,
                }
            )

    summary = stress.summarize_transition_profile(profile, "ALFNet", ["MoGERNN", "PatchSTG"])

    patch = next(row for row in summary if row["baseline"] == "PatchSTG")
    assert patch["integrated_gain_pct"] == pytest.approx(10.0)
    assert patch["transition_win_rate"] == pytest.approx(1.0)
    assert patch["win_blocks"] == 2
    assert patch["total_blocks"] == 2


def test_plot_transition_response_writes_a_nonempty_figure():
    profile = []
    for role, role_shift in [("NAS", 0.0), ("FS", 2.0)]:
        for model, model_shift in [("ALFNet", 0.0), ("MoGERNN", 2.0), ("PatchSTG", 1.0)]:
            for block in range(3):
                mape = 12.0 + role_shift + model_shift - block
                profile.append(
                    {
                        "model": model,
                        "role": role,
                        "age_block": block,
                        "age_start": block * 6,
                        "age_end": block * 6 + 5,
                        "age_center": block * 6 + 2.5,
                        "age_width": 6,
                        "event_count": 4,
                        "window_count": 24,
                        "mape": mape,
                        "mape_q25": mape - 0.5,
                        "mape_q75": mape + 0.5,
                    }
                )
    output_path = ROOT / ".pytest_cache" / "lifecycle_transition_response.png"
    plot_script = f"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("lifecycle_transition_plot_test", Path({str(SCRIPT_PATH)!r}))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
profile = {profile!r}
summary = stress.summarize_transition_profile(profile, "ALFNet", ["MoGERNN", "PatchSTG"])
plotted = stress.plot_transition_response(
    profile,
    summary,
    Path({str(output_path)!r}),
    ours_name="ALFNet",
    baseline_names=["MoGERNN", "PatchSTG"],
    early_horizon=6,
)
raise SystemExit(0 if plotted else 1)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", plot_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert output_path.stat().st_size > 1000
    finally:
        output_path.unlink(missing_ok=True)
