import importlib.util
from pathlib import Path


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
                        {"window_index": 11, "window_offset": 0, "MAPE": 10.0 * scale, "MAE": 1.0, "RMSE": 2.0},
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
        x_axis="event_age",
    )

    assert aligned[0]["baseline_mape"] == aligned[0]["mape"] * 2
    assert aligned[0]["improvement_pct"] == 50.0
    assert any(cell["improvement_pct"] == 50.0 for cell in grid)
    assert any(cell["patch_residual_weight"] is not None for cell in grid)
