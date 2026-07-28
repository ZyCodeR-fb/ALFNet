"""Generate combination YAMLs as the strict union of the single-switch files.

Hand-writing combinations is how protocol fields drift between arms. Every
combination here is computed from a.yaml / b.yaml / c.yaml / d.yaml against the
fg1 baseline: the diff of each single switch is extracted and re-applied, so a
combination cannot contain a value that no single switch introduced.
"""

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALFNET = ROOT / "configs" / "ALFNet"
TEMPORAL = ROOT / "configs" / "ALFTemporal"
# switches that live in the patch-branch config rather than the ALFNet config
PATCH_SWITCHES = {"a", "b"}


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, path + "."))
        else:
            out[path] = value
    return out


def assign(tree, dotted, value):
    node = tree
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def diff_against(base, other, ignore=()):
    base_flat, other_flat = flatten(base), flatten(other)
    return {
        key: other_flat[key]
        for key in other_flat
        if key not in ignore and base_flat.get(key) != other_flat[key]
    }


def build_patch_config(switches):
    """cond1 + the A and/or B deltas."""
    base = load(TEMPORAL / "METRLA_cond1.yaml")
    merged = deepcopy(base)
    for switch in sorted(switches):
        delta = diff_against(base, load(TEMPORAL / f"METRLA_cond1{switch}.yaml"),
                             ignore={"runtime.model_variant"})
        for key, value in delta.items():
            assign(merged, key, value)
    name = "cond1" + "".join(sorted(switches))
    assign(merged, "runtime.model_variant", name)
    return name, merged


def build_alfnet_config(switches):
    base = load(ALFNET / "fg1.yaml")
    merged = deepcopy(base)
    patch_switches = switches & PATCH_SWITCHES
    for switch in sorted(switches - PATCH_SWITCHES):
        delta = diff_against(base, load(ALFNET / f"{switch}.yaml"),
                             ignore={"runtime.model_variant"})
        for key, value in delta.items():
            assign(merged, key, value)
    if patch_switches:
        name, patch = build_patch_config(patch_switches)
        with (TEMPORAL / f"METRLA_{name}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(patch, handle, sort_keys=False, allow_unicode=True)
        assign(merged, "family.experts.patch.config_path",
               f"./configs/ALFTemporal/METRLA_{name}.yaml")
    stem = "".join(sorted(switches))
    assign(merged, "runtime.model_variant", stem)
    return stem, merged


COMBINATIONS = ["ac", "ad", "bc", "bd", "cd", "acd", "bcd", "abc", "abcd"]


def main():
    written = []
    for stem in COMBINATIONS:
        name, config = build_alfnet_config(set(stem))
        path = ALFNET / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        written.append(name)
    # A+B on its own is only a patch-branch config; write it so `ab` exists too
    name, patch = build_patch_config({"a", "b"})
    with (TEMPORAL / f"METRLA_{name}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(patch, handle, sort_keys=False, allow_unicode=True)
    print("wrote:", ", ".join(written), "| patch:", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
