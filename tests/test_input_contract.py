"""Pins the 2026-07 audit of what each model's forward actually consumes.

Every model receives the same seven-channel tensor, but half of them slice the
raw channel straight back off. These tests read the source of each adapter so
the recorded contract cannot drift away from the code without failing here.
"""

import re
from pathlib import Path

import pytest

from experiments.ples_evaluation import CHANNEL_NAMES, MODEL_INPUT_CONTRACT, _input_contract


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "model" / "comparison_model"

# model directory -> registered model name
DIRECTORY_TO_MODEL = {
    "FCLSTM": "FCLSTM", "GWNet": "GWNET", "HA": "HA", "MCPST_FSL": "MCPST_FSL",
    "MoGERNN": "MoGERNN", "PatchSTG": "PatchSTG", "SLPF": "SLPF", "STBP": "STBP",
    "STFiT": "STFiT", "STGCN": "STGCN", "STID": "STID", "stgode": "STGODE",
}
# adapters that reduce x_enc to the raw channel before the backbone sees it
SLICES_TO_RAW = re.compile(r"x_enc\[\.\.\., ?:\s*self\.(raw_enc_in|output_dim)\]")


def _adapter_source(directory):
    for name in ("__init__.py", f"{directory}.py"):
        path = COMPARISON / directory / name
        if path.exists() and "def forward(self, adj, x_enc" in path.read_text(encoding="utf-8"):
            return path.read_text(encoding="utf-8")
    raise AssertionError(f"no lifecycle adapter found for {directory}")


@pytest.mark.parametrize("directory,model", sorted(DIRECTORY_TO_MODEL.items()))
def test_recorded_contract_matches_the_adapter_source(directory, model):
    source = _adapter_source(directory)
    recorded = MODEL_INPUT_CONTRACT[model]
    slices_to_raw = bool(SLICES_TO_RAW.search(source))
    assert slices_to_raw == recorded.startswith("raw"), (
        f"{model}: source slices to raw={slices_to_raw} but contract says {recorded!r}"
    )


def test_every_registered_model_declares_a_contract():
    for model in DIRECTORY_TO_MODEL.values():
        assert model in MODEL_INPUT_CONTRACT, f"{model} has no declared input contract"
    for name in ("ALFNet", "ALFGraph", "ALFTemporal"):
        assert MODEL_INPUT_CONTRACT[name] == "all"


def test_contract_payload_flags_lifecycle_consumption():
    from types import SimpleNamespace

    assert _input_contract(SimpleNamespace(model="ALFNet"))["consumes_lifecycle_channels"] is True
    assert _input_contract(SimpleNamespace(model="MoGERNN"))["consumes_lifecycle_channels"] is False
    assert _input_contract(SimpleNamespace(model="Nope"))["consumed"] == "undeclared"
    assert list(CHANNEL_NAMES) == [
        "raw", "role_AAS", "role_NAS", "role_VS", "role_FS", "age", "obs_mask",
    ]


def test_stid_reads_the_calendar_channels_it_was_given():
    """The backbone used to hardcode index 1/2, which are role one-hots here."""
    from types import SimpleNamespace

    from model.comparison_model.STID.stid import STID

    backbone = STID(
        node_dim=4, input_len=12, input_dim=9, embed_dim=4, output_len=12, num_layer=1,
        temp_dim_tid=4, temp_dim_diw=4, time_of_day_size=1440, day_of_week_size=7,
        if_T_i_D=True, if_D_i_W=True, if_node=False,
        tid_index=7, diw_index=8,
        node_num=5, output_dim=1, seq_len=12, horizon=12,
    )
    assert (backbone.tid_index, backbone.diw_index) == (7, 8)
    # The reference layout still works when nothing is prepended.
    reference = STID(
        node_dim=4, input_len=12, input_dim=3, embed_dim=4, output_len=12, num_layer=1,
        temp_dim_tid=4, temp_dim_diw=4, time_of_day_size=1440, day_of_week_size=7,
        if_T_i_D=True, if_D_i_W=True, if_node=False,
        node_num=5, output_dim=1, seq_len=12, horizon=12,
    )
    assert (reference.tid_index, reference.diw_index) == (1, 2)


def test_objective_contract_reports_the_run_level_knobs():
    from types import SimpleNamespace

    from experiments.ples_evaluation import _objective_contract

    off = _objective_contract(SimpleNamespace())
    assert off["training_matches_evaluation_weighting"] is False
    assert off["lcbt_stratum_balanced_loss"] == 0.0
    on = _objective_contract(SimpleNamespace(lcbt_stratum_balanced_loss=1.0))
    assert on["training_matches_evaluation_weighting"] is True
