"""Identity anchored to the frozen manifest, and content-addressed output.

Equality between two bundles is not identity. The previous guard compared the
two sides to each other, which accepted:

  * ALFNet and MoGERNN swapped between the ``--ours`` and ``--baseline`` slots,
    while the default labels still said ALFNet/MoGERNN -- so the reported sign
    was inverted and the report said the opposite of what it measured;
  * any two distinct model strings, including an ALFNet *ablation* standing in
    for the mainline;
  * two bundles that both omitted a field, since ``None == None``.

So identity is established against ``configs/paper/final_runs.json`` -- the same
authority the training, audit, evaluation and aggregation already share -- and
each side must match its own entry. Cross-side equality is then only required
for what the shared experimental design determines.

Output is content-addressed. An analysis hash over the inputs and the analysis
configuration names the directory, an existing directory is never overwritten,
and a ``COMPLETE`` marker distinguishes a finished report from a partial one.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The mainline pair the v3 experiment is defined over. A variant outside this
# map is not the published model, whatever its directory is called.
MAINLINE_ROLE = {"ours": "ALFNet", "baseline": "MoGERNN"}
MAINLINE_VARIANTS = {"ALFNet": {"gs", "gs_bay", "gs_p8"}, "MoGERNN": {"mg_l3"}}

FROZEN_OUTPUT_ROOTS = ("predictions", "checkpoints", "manifests",
                       "reports/final", "reports/final_18")


class IdentityRefused(SystemExit):
    """The bundles are not the runs the manifest says they should be."""


class OutputRefused(SystemExit):
    """The destination already exists, or is not a permitted location."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest():
    return json.loads((ROOT / "configs/paper/final_runs.json").read_text("utf-8"))


def find_run_entry(manifest, dataset, model, seed):
    matches = [r for r in manifest["runs"]
               if r["dataset"] == dataset and r["model"] == model and r["seed"] == seed]
    if len(matches) != 1:
        raise IdentityRefused(
            f"the frozen manifest has {len(matches)} entries for "
            f"{model}/{dataset}/seed {seed}; identity cannot be established")
    return matches[0]


def _require(value, field, where):
    if value is None or (isinstance(value, str) and not value.strip()):
        raise IdentityRefused(f"{where}: required identity field {field!r} is missing or empty")
    return value


def verify_role(summary, role, manifest):
    """Match one bundle against its own authoritative run entry.

    ``role`` is the slot it was passed in as. Checking the slot is what stops a
    swap: a MoGERNN bundle handed to ``--ours`` is refused here rather than
    silently reported under the ALFNet label.
    """
    if role not in MAINLINE_ROLE:
        raise IdentityRefused(f"unknown role {role!r}")
    expected_model = MAINLINE_ROLE[role]

    where = f"role={role}"
    model = _require(summary.get("model"), "model", where)
    seed = _require(summary.get("seed"), "seed", where)
    variant = _require(summary.get("model_variant"), "model_variant", where)
    dataset = _require((summary.get("dataset") or {}).get("name"), "dataset.name", where)

    if model != expected_model:
        raise IdentityRefused(
            f"{where}: this slot must hold {expected_model}, got {model!r}. "
            "Passing the two models in the wrong slots inverts every reported sign.")
    if variant not in MAINLINE_VARIANTS[expected_model]:
        raise IdentityRefused(
            f"{where}: model_variant {variant!r} is not a mainline {expected_model} "
            f"variant {sorted(MAINLINE_VARIANTS[expected_model])}; an ablation may not "
            "stand in for the published model")

    entry = find_run_entry(manifest, dataset, model, int(seed))
    if entry["model_variant"] != variant:
        raise IdentityRefused(
            f"{where}: manifest says variant {entry['model_variant']!r}, bundle says {variant!r}")
    if not summary.get("formal_protocol_match", False):
        raise IdentityRefused(f"{where}: formal_protocol_match is not true")
    return entry


def assert_shared_design(ours, baseline):
    """Only what the experimental design determines must match across sides."""
    problems = []
    for field in ("protocol_hash", "training_protocol_hash", "lce_protocol_hash",
                  "ples_protocol_hash"):
        left, right = _require(ours.get(field), field, "ours"), \
                      _require(baseline.get(field), field, "baseline")
        if left != right:
            problems.append(f"{field}: {left} vs {right}")
    ours_dataset, baseline_dataset = ours.get("dataset") or {}, baseline.get("dataset") or {}
    for field in ("name", "fingerprint", "adjacency_hash", "timestamps_hash",
                  "test_values_hash"):
        left = _require(ours_dataset.get(field), f"dataset.{field}", "ours")
        right = _require(baseline_dataset.get(field), f"dataset.{field}", "baseline")
        if left != right:
            problems.append(f"dataset.{field}: {left} vs {right}")
    if int(ours["seed"]) != int(baseline["seed"]):
        problems.append(
            f"seed: {ours['seed']} vs {baseline['seed']} -- a cross-seed comparison "
            "confounds the model difference with training randomness")
    if problems:
        raise IdentityRefused("the two bundles are not the same experiment:\n  - "
                              + "\n  - ".join(problems))


def analysis_hash(payload):
    """Stable digest over the analysis configuration and every input hash."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_output_dir(report_root, digest):
    """Create ``<report_root>/<digest>/`` exclusively, or refuse.

    Exclusive creation rather than ``mkdir(exist_ok=True)``: an existing
    directory may be a completed report, or the remains of a crashed one, and
    overwriting either loses information. A ``COMPLETE`` marker written last
    tells the two apart.
    """
    root = Path(report_root).resolve()
    for frozen in FROZEN_OUTPUT_ROOTS:
        if frozen in root.as_posix():
            raise OutputRefused(
                f"refusing to write diagnostics into a frozen location: {root} "
                f"(matches {frozen!r})")
    destination = root / digest
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        marker = destination / "COMPLETE"
        state = "complete" if marker.exists() else "incomplete (previous run failed)"
        raise OutputRefused(
            f"{destination} already exists and is {state}; refusing to overwrite. "
            "Same inputs and configuration give the same hash, so a rerun needs "
            "nothing written.") from None
    return destination


def mark_complete(destination, payload):
    (Path(destination) / "COMPLETE").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


# --- authoritative run resolution and input provenance (D2-A) ----------------
#
# Anchored to configs/paper/final_runs.json, which is the same authority the
# training, audit, evaluation and aggregation already share. Comparing the two
# bundles to each other was never identity: it accepted swapped roles, an
# arbitrary model string and an ablation standing in for the mainline.
#
# What each field's evidence actually is matters, and is recorded rather than
# implied. Some fields appear in both the bundle and the authority and are
# cross-checked; some appear only in one and are merely recorded. Calling the
# second kind "verified" would be false.

ANALYSIS_IDENTITY_SCHEMA = 1

# Cross-checked: present in the frozen bundle *and* in final_runs.json.
CROSS_CHECKED_FIELDS = ("dataset", "model", "model_variant", "seed")
# Recorded from the bundle only; no second source exists to check them against.
BUNDLE_ONLY_FIELDS = ("exp_id", "protocol_hash", "training_protocol_hash",
                      "lce_protocol_hash", "ples_protocol_hash")
# Declared by the authority only; the result bundle does not carry them, so they
# are reported as declared and never as verified.
AUTHORITY_ONLY_FIELDS = ("run_id", "canonical_config_hash",
                         "legacy_canonical_config_hash", "config")


def resolve_run(summary, role, bundle_dir, manifest, output_root="./outputs"):
    """Resolve one bundle to its unique final-run entry, or refuse.

    ``role`` is the slot the bundle was passed in as, so a MoGERNN bundle handed
    to ``--ours`` is refused here rather than reported under the ALFNet label.
    Display labels never decide a statistical role.
    """
    entry = verify_role(summary, role, manifest)

    seed = int(_require(summary.get("seed"), "seed", f"role={role}"))
    dataset = _require((summary.get("dataset") or {}).get("name"), "dataset.name",
                       f"role={role}")
    matches = [r for r in manifest["runs"]
               if r["dataset"] == dataset and r["model"] == summary["model"]
               and int(r["seed"]) == seed]
    if len(matches) != 1:
        raise IdentityRefused(
            f"role={role}: {len(matches)} final-run entries for "
            f"{summary['model']}/{dataset}/seed {seed}; identity is not unique")
    if matches[0]["run_id"] != entry["run_id"]:
        raise IdentityRefused(f"role={role}: run entry disagreement")

    for field in CROSS_CHECKED_FIELDS:
        found = summary.get(field) if field != "dataset" else dataset
        expected = entry[field] if field != "dataset" else entry["dataset"]
        if field == "seed":
            found, expected = seed, int(entry["seed"])
        if found != expected:
            raise IdentityRefused(
                f"role={role}: {field} is {found!r} in the bundle but {expected!r} "
                "in the frozen run list")

    # The bundle must live where this final run writes, which is what ties an
    # on-disk directory to a run_id -- the summary itself carries no run_id.
    expected_root = (Path(output_root).resolve() / entry["output_subdir"]
                     / entry["run_id"] / "predictions")
    actual = Path(bundle_dir).resolve()
    if actual.parent != expected_root:
        raise IdentityRefused(
            f"role={role}: bundle at {actual} is not under the directory this "
            f"final run writes ({expected_root}); it may belong to another run")

    declared = summary.get("required_training_protocol_hash")
    if declared and declared != summary.get("training_protocol_hash"):
        raise IdentityRefused(
            f"role={role}: training_protocol_hash {summary.get('training_protocol_hash')!r} "
            f"!= required {declared!r}")
    for field in BUNDLE_ONLY_FIELDS:
        _require(summary.get(field), field, f"role={role}")
    return entry


def _repo_relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        # Outside the repo: record the name only. Absolute machine paths must
        # never enter a hash, or the same inputs hash differently per machine.
        return f"<external>/{path.name}"


def input_provenance(files):
    """``{logical role: {path, sha256}}`` for every real input file."""
    out = {}
    for role, path in sorted(files.items()):
        resolved = Path(path)
        if not resolved.exists():
            raise IdentityRefused(f"provenance input missing: {role} -> {path}")
        out[role] = {"repo_relative_path": _repo_relative(resolved),
                     "sha256": file_sha256(resolved)}
    return out


def code_commit():
    import subprocess
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:                                    # noqa: BLE001
        return None


def build_analysis_identity(*, analysis_type, dataset, roles, seeds, entries,
                            summaries, endpoint_inventory, resampling,
                            bootstrap_samples, bootstrap_seed, margins,
                            inputs, display_labels=None):
    """Canonical, deterministic identity + provenance payload.

    ``display_labels`` is carried outside the hashed statistical configuration on
    purpose: renaming a plot legend must not change the identity of the numbers.
    """
    statistical = {
        "analysis_identity_schema": ANALYSIS_IDENTITY_SCHEMA,
        "analysis_type": analysis_type,
        "dataset": dataset,
        "canonical_model_roles": dict(roles),
        "seeds": sorted(int(s) for s in seeds),
        "resolved_runs": {
            role: {
                **{field: entry[field] for field in AUTHORITY_ONLY_FIELDS
                   if field in entry},
                **{field: entry[field] for field in CROSS_CHECKED_FIELDS},
            }
            for role, entry in sorted(entries.items())
        },
        "bundle_declared": {
            role: {field: summary.get(field) for field in BUNDLE_ONLY_FIELDS}
            for role, summary in sorted(summaries.items())
        },
        "endpoint_inventory": endpoint_inventory,
        "resampling_design": resampling,
        "bootstrap": {"samples": int(bootstrap_samples), "rng_seed": int(bootstrap_seed)},
        "margins": margins,
        "inferential_status": "descriptive_not_formal",
        "post_hoc_diagnostic_only": True,
        "input_provenance": input_provenance(inputs),
        "evidence_notes": {
            "cross_checked": list(CROSS_CHECKED_FIELDS),
            "bundle_declared_only": list(BUNDLE_ONLY_FIELDS),
            "authority_declared_only": list(AUTHORITY_ONLY_FIELDS),
            "canonical_config_hash": (
                "declared by configs/paper/final_runs.json only. The frozen "
                "result bundles do not carry it, so it is reported as declared "
                "and is NOT independently verified here."
            ),
        },
    }
    payload = dict(statistical)
    payload["analysis_hash"] = analysis_hash(statistical)
    payload["code_commit"] = code_commit()
    if display_labels:
        # Outside the hash: presentation, not statistics.
        payload["display_labels"] = dict(display_labels)
    return payload
