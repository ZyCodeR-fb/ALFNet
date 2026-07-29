"""Pins what each model's forward actually consumes.

Every model receives the same seven-channel tensor. Published baselines must
reduce it to their source-native inputs -- the traffic series plus whatever
calendar features their own paper defines -- because the lifecycle role, age and
observation-mask channels are this method's contribution, and feeding them to a
baseline turns it into a different model.

Two layers of check: a source read, so the recorded contract cannot drift from
the code silently, and a behavioural one that poisons the non-raw channels and
requires bit-identical output. The behavioural check is the load-bearing one --
a baseline can consume the channels through a config-derived `input_dim` with no
visible slice anywhere in its adapter, which is exactly how five of them were
reading lifecycle inputs before this was fixed.
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
# Adapters that reduce x_enc to the raw channels before the backbone sees it,
# either by an explicit slice or through the shared helper.
SLICES_TO_RAW = re.compile(
    r"x_enc\[\.\.\., ?:\s*self\.(raw_enc_in|output_dim)\]"
    r"|source_native_input\(\s*x_enc"
)


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


BASELINES = ["FCLSTM", "GWNET", "STGCN", "STGODE", "STID", "MoGERNN",
             "PatchSTG", "SLPF", "STBP", "STFiT", "HA", "MCPST_FSL"]


@pytest.mark.parametrize("model", BASELINES)
def test_baseline_output_is_unchanged_by_the_lifecycle_channels(model):
    """Behavioural: poison every non-raw channel and require identical output.

    Reading the source is not enough -- a baseline can consume the channels
    through a config-derived `input_dim` without any visible slice. This drives
    the real forward instead. The seed is reset per call so a stochastic module
    (STFiT carries a VAE) cannot be mistaken for one that reads the extra
    channels: that false positive is exactly what this control caught.
    """
    import torch

    from args import load_args_from_config
    from model.regist_model import load_model_class

    args = load_args_from_config(f"configs/{model}/METRLA_lcbt_v3.yaml", [])
    args.device = "cpu"
    net = load_model_class(args.model)(args).eval()
    nodes = len(getattr(net, "known_nodes", range(207)))
    generator = torch.Generator().manual_seed(1)
    adjacency = torch.rand(nodes, nodes, generator=generator)
    clean = torch.randn(2, args.input_len, nodes, args.enc_in, generator=generator)
    poisoned = clean.clone()
    poisoned[..., args.raw_enc_in:] = 99.0
    marks = torch.rand(2, args.input_len, 3, generator=generator) * 5
    horizon = torch.rand(2, args.pred_len, 3, generator=generator) * 5

    def run(inputs):
        torch.manual_seed(0)
        with torch.no_grad():
            out = net(adjacency, inputs, marks, None, None, horizon, 0)
        return out["pred"] if isinstance(out, dict) else out

    baseline, repeat = run(clean), run(clean)
    assert torch.equal(baseline, repeat), "not reproducible; the check below is void"
    assert torch.equal(baseline, run(poisoned)), (
        f"{model} reacts to the lifecycle channels it must not consume"
    )


def test_alfnet_does_consume_the_lifecycle_channels():
    """The mirror image: the method must actually use its own inputs."""
    import torch

    from args import load_args_from_config
    from model.regist_model import load_model_class

    args = load_args_from_config("configs/ALFNet/fg1.yaml", [])
    args.device = "cpu"
    net = load_model_class(args.model)(args).eval()
    nodes = len(net.known_nodes)
    generator = torch.Generator().manual_seed(1)
    adjacency = torch.rand(nodes, nodes, generator=generator)
    clean = torch.randn(2, args.input_len, nodes, args.enc_in, generator=generator)
    poisoned = clean.clone()
    poisoned[..., args.raw_enc_in:] = 99.0
    marks = torch.rand(2, args.input_len, 3, generator=generator) * 5
    horizon = torch.rand(2, args.pred_len, 3, generator=generator) * 5
    with torch.no_grad():
        base = net(adjacency, clean, marks, None, None, horizon, 0)["pred"]
        other = net(adjacency, poisoned, marks, None, None, horizon, 0)["pred"]
    assert not torch.allclose(base, other)


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
