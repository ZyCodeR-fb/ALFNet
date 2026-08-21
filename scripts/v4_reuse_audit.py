#!/usr/bin/env python
"""Decide, per run, whether an existing checkpoint may stand in for a new one.

Twelve PEMS08 and PEMS-BAY runs already exist under the development plan. They
were trained on the same configs, the same payloads and the same code that the
five-dataset plan would use, so retraining them would burn days of GPU to
reproduce numbers we already have. But "would produce the same thing" is a claim,
and this is where it gets checked rather than assumed.

Two rules shape it.

**Per run, not per batch.** A run that fails any check is marked `train`; it does
not condemn the other eleven, and it is never silently reused.

**A code allowance names a content pair, not a file.** Whitelisting
`scripts/train_v4.py` outright would let any future edit to it ride along. Each
allowance below pins the exact before/after digests of one reviewed change, so
the next edit to that file invalidates reuse until somebody looks at it. That is
not hypothetical: adding `allow_phase` to the collector invalidated all twelve
until the change was reviewed and pinned, which is the mechanism working.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v4_development_plan import sha256_file  # noqa: E402
from scripts.train_v4 import COMPLETE_MARKER, RUN_STATE_NAME  # noqa: E402
from scripts.v4_development_eval_v2 import (  # noqa: E402
    EVAL_NAME,
    EVAL_SCHEMA,
    TOLERANCE,
)

# One reviewed transition, pinned to both digests. Not a licence for the file.
CODE_ALLOWANCES = [
    {
        "path": "scripts/train_v4.py",
        "from_sha256": "8d96cf4da260",
        "to_sha256": "b3c9a84b470a",
        "reason": ("the hardcoded 'eight v4 configs' message became the "
                   "generator's count. The change is inside a guard that raises "
                   "before any session, model or optimiser is built; no training "
                   "computation, objective, schedule or checkpoint rule moved."),
        "reviewed_diff": "3 lines, one `if` and its message",
    },
    {
        "path": "experiments/v4_validation_collector.py",
        "from_sha256": "90416de97cc8",
        "to_sha256": "00d738d70610",
        "reason": ("`_refuse_non_validation` gained an `allow_phase` argument "
                   "defaulting to 'validation', so the final test evaluator can "
                   "opt in by name instead of bypassing the constructor. Every "
                   "existing caller omits it, and `context.phase != allow_phase` "
                   "with that default is exactly the previous "
                   "`context.phase != 'validation'`. No metric, accumulation or "
                   "aggregation changed, and the 36 cells these runs recorded "
                   "are reproduced bit-identically by the recomputation."),
        "reviewed_diff": "one guard signature, its message, and one optional "
                         "constructor parameter",
    },
]

EXPECTED_PARAMETERS = {"ALFNet": 1_037_529, "MoGERNN": 226_270}


def _prefix_match(allowance, was, now):
    return (was.startswith(allowance["from_sha256"])
            and now.startswith(allowance["to_sha256"]))


def code_findings(recorded):
    """Anything moved since the run that is not a pinned, reviewed transition."""
    findings = []
    for name, was in sorted((recorded or {}).items()):
        path = ROOT / name
        if not path.exists():
            findings.append(f"{name}: recorded by the run but absent now")
            continue
        now = sha256_file(path)
        if now == was:
            continue
        allowed = [a for a in CODE_ALLOWANCES
                   if a["path"] == name and _prefix_match(a, was, now)]
        if not allowed:
            findings.append(
                f"{name}: changed since the run ({was[:12]} -> {now[:12]}) and "
                "the transition is not a reviewed allowance")
    return findings


def audit_run(dataset, model, seed, directory, config_path, data_entry,
              source_plan_sha256):
    """Every check for one candidate reuse. Returns a record; never raises."""
    findings = []
    record = {
        "run": f"{dataset}/{model}/seed{seed}",
        "dataset": dataset, "model": model, "seed": int(seed),
        "source_run_directory": str(Path(directory).relative_to(ROOT)),
        "source_plan_sha256": source_plan_sha256,
    }

    if not (directory / COMPLETE_MARKER).exists():
        findings.append("not COMPLETE")
        record.update(findings=findings, reuse_compatibility_status="INCOMPATIBLE",
                      execution_action="train")
        return record
    best_path = directory / "best.pt"
    try:
        best = torch.load(best_path, weights_only=False, map_location="cpu")
        last = torch.load(directory / RUN_STATE_NAME, weights_only=False,
                          map_location="cpu")
    except Exception as error:                       # noqa: BLE001 - reported
        findings.append(f"checkpoint unreadable: {error}")
        record.update(findings=findings, reuse_compatibility_status="INCOMPATIBLE",
                      execution_action="train")
        return record

    record["source_git_commit"] = best.get("git_commit")
    record["checkpoint_sha256"] = sha256_file(best_path)
    record["best_epoch"] = best.get("best_epoch")

    # 2-5. identity: who it is, what it read, what it was exposed to
    config_sha = sha256_file(config_path)
    for field, expected in (("dataset", dataset), ("model", model),
                            ("seed", int(seed)),
                            ("config_sha256", config_sha),
                            ("payload_sha256", data_entry["payload_sha256"]),
                            ("unseen_sha256", data_entry["unseen_sha256"]),
                            ("manifest_sha256", data_entry["manifest_sha256"])):
        actual = best.get(field, KeyError)
        if actual is KeyError:
            findings.append(f"{field}: absent from the checkpoint")
        elif actual != expected:
            findings.append(f"{field}: {actual!r} != {expected!r}")
    if not best.get("model_variant"):
        findings.append("model_variant: absent from the checkpoint")
    for field in ("static_asset_hash", "train_exposure_hash",
                  "validation_exposure_hash"):
        if not best.get(field):
            findings.append(f"{field}: absent from the checkpoint")
        elif best.get(field) != last.get(field):
            findings.append(f"{field}: best.pt and last.pt disagree")
    if best.get("train_exposure_hash") == best.get("validation_exposure_hash"):
        findings.append("train and validation share an exposure hash")

    # 6. the weights still fit the model this plan would build
    expected_parameters = EXPECTED_PARAMETERS.get(model)
    if best.get("parameter_count") != expected_parameters:
        findings.append(f"parameter_count {best.get('parameter_count')} != "
                        f"{expected_parameters}")
    state = best.get("model_state") or {}
    if not state:
        findings.append("model_state is empty")
    else:
        total = sum(int(t.numel()) for t in state.values() if torch.is_tensor(t))
        if expected_parameters and total < expected_parameters:
            findings.append(f"model_state holds {total} elements, fewer than the "
                            f"{expected_parameters} parameters expected")

    # 7. the corrected validation re-evaluation
    eval_path = directory / EVAL_NAME
    if not eval_path.exists():
        findings.append(f"{EVAL_NAME} is absent; the run has no corrected "
                        "validation re-evaluation")
    else:
        evaluation = json.loads(eval_path.read_text("utf-8"))
        if evaluation.get("schema_version") != EVAL_SCHEMA:
            findings.append(f"validation evaluation is schema "
                            f"{evaluation.get('schema_version')}, not {EVAL_SCHEMA}")
        if evaluation.get("partial_diagnostic_only"):
            findings.append("validation evaluation is a partial diagnostic")
        if int(evaluation.get("run_seed_used", -1)) != int(seed):
            findings.append(f"validation evaluated at seed "
                            f"{evaluation.get('run_seed_used')}, not {seed}")
        if not evaluation.get("agrees"):
            findings.append("validation evaluation does not agree with training")
        if float(evaluation.get("max_abs_cell_difference", 1.0)) > TOLERANCE:
            findings.append(f"validation max cell difference "
                            f"{evaluation.get('max_abs_cell_difference')} > {TOLERANCE}")
        if evaluation.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            findings.append("the validation evaluation scored a different "
                            "checkpoint from the one being reused")

    # 8. selection provenance
    for field, expected in (("early_stopping_metric", "validation life_avg_mae"),
                            ("tie_rule", "keep_earlier_epoch")):
        if best.get(field) != expected:
            findings.append(f"{field}: {best.get(field)!r} != {expected!r}")
    scaler = best.get("scaler") or {}
    if scaler.get("fitted_on") != "train window, training-known nodes only":
        findings.append(f"scaler fitted_on is {scaler.get('fitted_on')!r}")
    if best.get("best_epoch") is None:
        findings.append("best_epoch is absent")

    # 9-10. code, with narrowly pinned allowances
    findings.extend(code_findings(last.get("code_sha256")))

    record["findings"] = findings
    record["reuse_compatibility_status"] = "COMPATIBLE" if not findings else "INCOMPATIBLE"
    record["execution_action"] = "reuse" if not findings else "train"
    return record


def audit_reusable(development_plan, five_dataset_data, five_dataset_configs):
    """Audit every development run against what the five-dataset plan needs."""
    root = (ROOT / "outputs_v4/development_base"
            / development_plan["plan_sha256"])
    from scripts.v4_launcher import plan_runs

    records = []
    for run in plan_runs(root, development_plan):
        entry = five_dataset_configs.get(f"{run.dataset}_{run.model}")
        data = five_dataset_data.get(run.dataset)
        if entry is None or data is None:
            continue                    # this dataset is not reusable at all
        records.append(audit_run(
            run.dataset, run.model, run.seed, run.directory,
            ROOT / entry["path"], data, development_plan["plan_sha256"]))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)

    from scripts.build_v4_development_plan import (assert_plan_still_holds,
                                                   load_plan)
    from scripts import build_v4_five_dataset_plan as five

    development = assert_plan_still_holds(load_plan())
    plan = five.load_plan()
    records = audit_reusable(development, plan["data"], plan["configs"])
    if arguments.json:
        print(json.dumps(records, indent=1, sort_keys=True, allow_nan=False))
        return 0
    compatible = [r for r in records if r["execution_action"] == "reuse"]
    print(f"{len(compatible)}/{len(records)} development runs are reusable")
    for record in records:
        mark = "REUSE" if record["execution_action"] == "reuse" else "TRAIN"
        print(f"  {mark} {record['run']:26s} best_epoch="
              f"{record.get('best_epoch')} ckpt={record.get('checkpoint_sha256','-')[:12]}")
        for finding in record["findings"]:
            print(f"        - {finding}")
    return 0 if len(compatible) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
