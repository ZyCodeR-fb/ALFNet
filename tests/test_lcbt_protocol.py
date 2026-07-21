import numpy as np
import pytest
import yaml
import gc
from pathlib import Path

from data_provider.lcbt_protocol import (
    LCBT_PHASES,
    accumulate_phase_mae,
    build_lcbt_schedule,
    compose_effective_lifecycle_features,
    finalize_phase_mae,
    lcbt_manifest_hash,
    resolve_lcbt_boundaries,
)
from data_provider.graph_data_loader import _lcbt_schedule_summary


ROOT = Path(__file__).resolve().parents[1]
LCBT_CONFIGS = [
    ROOT / "configs" / family / "METRLA_lcbt.yaml"
    for family in (
        "MoGERNN",
        "PatchSTG",
        "FCLSTM",
        "GWNET",
        "HA",
        "MCPST_FSL",
        "SLPF",
        "STBP",
        "STFiT",
        "STGCN",
        "STGODE",
        "STID",
    )
] + [
    ROOT / "configs" / "ALFNet" / "rf_lcb_g.yaml",
    ROOT / "configs" / "ALFNet" / "rf_lcb_s.yaml",
]
LCBT_V2_CONFIGS = [
    ROOT / "configs" / family / "METRLA_lcbt_v2.yaml"
    for family in (
        "MoGERNN",
        "PatchSTG",
        "FCLSTM",
        "GWNET",
        "HA",
        "MCPST_FSL",
        "SLPF",
        "STBP",
        "STFiT",
        "STGCN",
        "STGODE",
        "STID",
    )
] + [
    ROOT / "configs" / "ALFNet" / "rf_lcb2_g.yaml",
    ROOT / "configs" / "ALFNet" / "rf_lcb2_s.yaml",
]


def test_lcbt_boundaries_are_chronological_and_scaler_is_train_only():
    boundaries = resolve_lcbt_boundaries(
        total_steps=1000,
        train_end_ratio=0.50,
        val_end_ratio=0.60,
        test_start_ratio=0.60,
        scaler_end_ratio=0.50,
    )

    assert boundaries == {
        "train_start": 0,
        "train_end": 500,
        "val_start": 500,
        "val_end": 600,
        "test_start": 600,
        "test_end": 1000,
        "scaler_start": 0,
        "scaler_end": 500,
    }


def test_manifest_hash_is_independent_of_machine_specific_data_path():
    common = {
        "protocol_name": "lcbt_v1",
        "data_hash": "same-bytes",
        "boundaries": {"train_end": 50, "val_end": 60},
    }
    windows = dict(common, data_path="A:/projects/MoGERNN/data/METRLA.pkl")
    ubuntu = dict(common, data_path="/srv/MoGERNN/data/METRLA.pkl")
    assert lcbt_manifest_hash(windows) == lcbt_manifest_hash(ubuntu)


def test_lcbt_schedule_balances_scenarios_and_all_transition_positions():
    num_nodes = 60
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node in range(num_nodes):
        adjacency[node, (node + 1) % num_nodes] = 1.0
        adjacency[(node + 1) % num_nodes, node] = 1.0

    schedule = build_lcbt_schedule(
        num_samples=260,
        input_len=12,
        adjacency=adjacency,
        calendar_regimes=["OffPeak"] * 260,
        seed=2026,
        epoch=3,
    )

    scenarios = [item["scenario"] for item in schedule]
    assert scenarios.count("join") == 104
    assert scenarios.count("fault") == 104
    assert scenarios.count("steady") == 52

    join_steps = np.bincount(
        [item["transition_steps"] for item in schedule if item["scenario"] == "join"],
        minlength=13,
    )
    fault_steps = np.bincount(
        [item["transition_steps"] for item in schedule if item["scenario"] == "fault"],
        minlength=13,
    )
    assert join_steps.tolist() == [8] * 13
    assert fault_steps.tolist() == [8] * 13

    phase_ids = {
        item["phase_id"]
        for item in schedule
        if item["scenario"] != "steady"
    }
    assert phase_ids == set(range(len(LCBT_PHASES)))
    assert all(item["phase_id"] == -1 for item in schedule if item["scenario"] == "steady")
    summary = _lcbt_schedule_summary(schedule, num_nodes, input_len=12)
    assert summary["scenario_histogram"] == {"join": 104, "fault": 104, "steady": 52}
    assert summary["transition_histogram"]["join"] == {
        str(index): 8 for index in range(13)
    }


def test_effective_mask_combines_natural_and_lifecycle_missingness():
    raw = np.arange(12, dtype=np.float32).reshape(4, 3, 1) + 1.0
    raw[2, 1, 0] = np.nan
    lifecycle_mask = np.ones((4, 3), dtype=bool)
    lifecycle_mask[:2, 0] = False
    role_ids = np.zeros((4, 3), dtype=np.int64)

    features, effective = compose_effective_lifecycle_features(
        raw,
        lifecycle_mask,
        role_ids,
        append_role=True,
        append_age=True,
        append_observation_mask=True,
    )

    assert features.shape == (4, 3, 7)
    assert not effective[0, 0]
    assert not effective[1, 0]
    assert not effective[2, 1]
    assert effective[3, 1]
    assert np.isnan(features[0, 0, 0])
    assert np.isnan(features[2, 1, 0])
    np.testing.assert_array_equal(features[..., -1], effective.astype(np.float32))
    np.testing.assert_allclose(features[:, 0, 5], [0.0, 0.0, 0.25, 0.50])


def test_lcbt_schedule_is_deterministic_per_seed_epoch_and_changes_across_epochs():
    adjacency = np.eye(24, dtype=np.float32)
    regimes = ["AM", "PM", "OffPeak", "Weekend"] * 20
    first = build_lcbt_schedule(80, 12, adjacency, regimes, seed=11, epoch=4)
    replay = build_lcbt_schedule(80, 12, adjacency, regimes, seed=11, epoch=4)
    next_epoch = build_lcbt_schedule(80, 12, adjacency, regimes, seed=11, epoch=5)

    def compact(schedule):
        return [
            (
                item["scenario"],
                item["transition_steps"],
                item["phase_id"],
                tuple(item["changed_nodes"]),
                item["topology_mode"],
            )
            for item in schedule
        ]

    assert compact(first) == compact(replay)
    assert compact(first) != compact(next_epoch)


def test_val_phase_mae_is_original_scale_changed_cohort_macro():
    true = np.zeros((6, 2, 3, 1), dtype=np.float32)
    pred = np.zeros_like(true)
    phase_ids = np.arange(6)
    changed = np.zeros((6, 3), dtype=np.float32)
    changed[:, 1] = 1.0
    for phase_index in range(6):
        pred[phase_index, :, 1, 0] = float(phase_index + 1)
        pred[phase_index, :, 0, 0] = 100.0
    sums = [0.0] * 6
    counts = [0] * 6

    accumulate_phase_mae(
        sums,
        counts,
        true,
        pred,
        phase_ids,
        changed,
    )
    metrics = finalize_phase_mae(sums, counts)

    assert metrics["Cold_MAE"] == pytest.approx(1.0)
    assert metrics["Outage_MAE"] == pytest.approx(6.0)
    assert metrics["mae"] == pytest.approx(3.5)


@pytest.mark.parametrize("config_path", LCBT_CONFIGS, ids=lambda path: path.parent.name)
def test_all_lcbt_yamls_share_the_protocol_contract(config_path):
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    lifecycle = config["lifecycle"]
    training = config["training"]

    assert config_path.exists()
    assert training["max_epoch"] == 100
    assert training["patience"] == 10
    assert training["stop_based"] == "val_phase"
    assert lifecycle["train_for_async_lifecycle"] is True
    assert lifecycle["lifecycle_training_strategy"] == "lcbt_v1"
    assert lifecycle["lifecycle_role_features"] is True
    assert lifecycle["lifecycle_history_age_feature"] is True
    assert lifecycle["lifecycle_observation_mask_feature"] is True
    assert lifecycle["lifecycle_keep_random_mask"] is False
    assert lifecycle["lcbt_cohort_fractions"] == [0.025, 0.05, 0.10]
    assert lifecycle["lcbt_cohort_weights"] == [0.25, 0.50, 0.25]
    assert config["async_test"]["ples_seed"] == 2026
    assert len(config["runtime"]["model_variant"]) <= 12


def _load_integration_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    from args import load_args_from_config
    from lib.utils import load_project_adjacency
    from model.regist_model import model_dict
    from utils.model_output import unpack_model_output

    return (
        torch,
        load_args_from_config,
        load_project_adjacency,
        model_dict,
        unpack_model_output,
    )


def _lcbt_smoke_input(torch, args, num_nodes, batch_size=1):
    generator = torch.Generator().manual_seed(2026)
    x_enc = torch.randn(
        batch_size,
        args.input_len,
        num_nodes,
        args.enc_in,
        generator=generator,
    )
    x_enc[..., args.raw_enc_in:] = 0.0
    role_start = args.raw_enc_in
    x_enc[..., role_start] = 1.0
    age_index = role_start + 4
    mask_index = age_index + 1
    observation_mask = torch.ones(
        batch_size,
        args.input_len,
        num_nodes,
        1,
    )
    observation_mask[:, : args.input_len // 2, 0] = 0.0
    x_enc[:, : args.input_len // 2, 0, : args.raw_enc_in] = 0.0
    x_enc[..., age_index:age_index + 1] = observation_mask.cumsum(dim=1) / float(
        args.input_len
    )
    x_enc[..., mask_index:mask_index + 1] = observation_mask
    x_dec = torch.randn(
        batch_size,
        args.pred_len,
        num_nodes,
        args.c_out,
        generator=generator,
    )
    enc_t_mark = torch.zeros(batch_size, args.input_len, args.enc_mark_in)
    dec_t_mark = torch.zeros(batch_size, args.pred_len, args.enc_mark_in)
    return x_enc, x_dec, enc_t_mark, dec_t_mark


@pytest.mark.parametrize(
    "config_path",
    LCBT_CONFIGS + LCBT_V2_CONFIGS,
    ids=lambda path: f"{path.parent.name}-{path.stem}",
)
def test_all_lcbt_models_build_and_accept_a_finite_tail_batch(config_path):
    (
        torch,
        load_args_from_config,
        load_project_adjacency,
        model_dict,
        unpack_model_output,
    ) = _load_integration_modules()
    args = load_args_from_config(
        str(config_path),
        overrides={"device": "cpu", "num_workers": 0},
    )
    model = model_dict[args.model].Model(args).float()
    model.train()
    adjacency = torch.as_tensor(
        load_project_adjacency({"data_path": args.data_path})
    ).float()
    x_enc, x_dec, enc_t_mark, dec_t_mark = _lcbt_smoke_input(
        torch,
        args,
        adjacency.size(0),
        batch_size=1,
    )

    output = model(
        adjacency,
        x_enc,
        enc_t_mark if args.t_mark else None,
        None,
        x_dec,
        dec_t_mark,
        1,
    )
    prediction, _ = unpack_model_output(output)

    assert prediction.shape == (
        1,
        args.pred_len,
        adjacency.size(0),
        args.c_out,
    )
    assert torch.isfinite(prediction).all()
    del prediction, output, model
    gc.collect()


def test_lcbt_grin_scope_and_final_model_regression_contract():
    (
        _,
        load_args_from_config,
        _,
        model_dict,
        _,
    ) = _load_integration_modules()
    from layers.source_grin import GRINInputImputer

    cases = {
        "configs/ALFNet/rf_alf_sd.yaml": ("graph", True, False),
        "configs/ALFNet/rf_lcb_g.yaml": ("graph", True, False),
        "configs/ALFNet/rf_lcb_s.yaml": ("shared", False, True),
        "configs/ALFNet/rf_lcb2_g.yaml": ("graph", True, False),
        "configs/ALFNet/rf_lcb2_s.yaml": ("shared", False, True),
    }
    for config_path, expected in cases.items():
        args = load_args_from_config(
            str(ROOT / config_path),
            overrides={"device": "cpu", "num_workers": 0},
        )
        model = model_dict[args.model].Model(args).float()
        scope, graph_grin, shared_grin = expected
        assert model.input_imputation_scope == scope
        assert (model.graph_expert.data_pre is not None) is graph_grin
        assert (model.shared_input_imputer is not None) is shared_grin
        assert sum(
            isinstance(module, GRINInputImputer) for module in model.modules()
        ) == 1
        if config_path.endswith("rf_alf_sd.yaml"):
            assert sum(parameter.numel() for parameter in model.parameters()) == 4_841_931
            assert not any(
                key.startswith("shared_input_imputer.")
                for key in model.state_dict()
            )
        del model
        gc.collect()
