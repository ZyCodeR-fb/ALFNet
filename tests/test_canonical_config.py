from pathlib import Path

from args import load_args_from_config

ROOT = Path(__file__).resolve().parents[1]


def test_same_yaml_has_stable_identity():
    path = ROOT / "configs/ALFNet/PEMS08.yaml"
    first = load_args_from_config(path)
    second = load_args_from_config(path)
    assert first.canonical_config_hash == second.canonical_config_hash
    assert first.canonical_config == second.canonical_config


def test_runtime_device_is_not_configuration_identity():
    path = ROOT / "configs/MoGERNN/PEMS08.yaml"
    first = load_args_from_config(path)
    second = load_args_from_config(path, {"device": "cpu"})
    assert first.canonical_config_hash == second.canonical_config_hash


def test_stride_is_part_of_configuration_identity(tmp_path):
    source = ROOT / "configs/MoGERNN/PEMS08.yaml"
    text = source.read_text("utf-8")
    assert "slide_step: 1" in text
    changed = tmp_path / "changed.yaml"
    changed.write_text(text.replace("slide_step: 1", "slide_step: 2"), "utf-8")
    assert (load_args_from_config(source).canonical_config_hash
            != load_args_from_config(changed).canonical_config_hash)
