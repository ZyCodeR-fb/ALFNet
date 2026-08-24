#!/usr/bin/env python
"""The five-dataset plan's running permit, and why it is a separate file.

The four original manifests say `training_ready: false`. That is the state they
were generated in and it is *history*: the twelve completed development runs
recorded those manifests' hashes, so editing a flag in place would change them
and orphan those runs. Equally, "the manifest says false but the entry point
trains anyway" is exactly the contradiction that makes a readiness flag
worthless.

So readiness moves out of the manifests and into a per-plan attestation. The
manifests keep describing the data; this file says whether *this plan* may run,
and the launcher and trainer check it before creating a run directory, spawning
a child or touching CUDA.

**The identity is one-way, deliberately.** A plan that hashed this file and a
file that hashed the plan could only ever be made to agree by rewriting one of
them until they did, which is not verification. Instead:

    plan  ->  freezes the experiment: datasets, configs, code, runs
    attestation -> references plan_sha256, and hashes the evidence
    launcher -> checks both, and that the attestation names *this* plan

The plan records this file's *path and schema*, never its digest, so nothing
points back. A test asserts that regenerating either one in any order converges
without a fixed-point loop.

    python scripts/build_v4_readiness.py
    python scripts/build_v4_readiness.py --check
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_five_dataset_plan as five  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

ATTESTATION_PATH = ROOT / "configs/paper/v4_five_dataset_readiness.json"
SCHEMA_VERSION = 1
GENERATOR = "scripts/build_v4_readiness.py"


class ReadinessRefused(RuntimeError):
    """This plan has no valid running permit."""


def attestation_hash(record):
    body = {key: value for key, value in record.items()
            if key != "attestation_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build(plan=None, preflight=None):
    plan = plan or five.load_plan()
    five.assert_plan_still_holds(plan)
    if preflight is not None:
        from scripts.run_v4_cpu_preflight import validate
        validate(preflight, plan)
    data = {}
    for dataset, entry in plan["data"].items():
        data[dataset] = {
            "manifest_sha256": entry["manifest_sha256"],
            "payload_sha256": entry["payload_sha256"],
            "unseen_sha256": entry["unseen_sha256"],
            "target": entry["target"], "target_channel": entry["target_channel"],
            "unit": entry["unit"], "num_nodes": entry["num_nodes"],
        }
    configs = {name: {"sha256": entry["sha256"],
                      "nested": {k: v["sha256"]
                                 for k, v in entry.get("nested", {}).items()}}
               for name, entry in plan["configs"].items()}

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v4_five_dataset_readiness_attestation",
        "generator": GENERATOR,
        "generator_sha256": sha256_file(ROOT / GENERATOR),
        # One-way: this references the plan. The plan never hashes this file.
        "plan_path": str(five.PLAN_PATH.relative_to(ROOT)),
        "plan_sha256": plan["plan_sha256"],
        "data": data,
        "configs": configs,
        "cpu_preflight": preflight or {},
        "reuse_policy": {"allowed": False, "runs_reused": 0},
        "readiness": {
            "data_ready": True,
            "graph_ready": True,
            # Both are earned, not declared, by the ten CPU preflights.
            "exposure_ready": bool(preflight and preflight.get("all_passed")),
            "training_ready": bool(preflight and preflight.get("all_passed")),
        },
        "manifest_readiness_note": (
            "the per-dataset manifests keep the readiness flags they were "
            "generated with; they describe the data, and the twelve completed "
            "development runs recorded their hashes. This attestation, not those "
            "flags, is what permits this plan to run."),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    record["attestation_sha256"] = attestation_hash(record)
    return record


def load(path=None):
    record = json.loads(Path(path or ATTESTATION_PATH).read_text("utf-8"))
    recorded, actual = record.get("attestation_sha256"), attestation_hash(record)
    if recorded != actual:
        raise ReadinessRefused(
            f"the readiness attestation's recorded hash {recorded} does not "
            f"describe its contents {actual}; it was edited after generation")
    return record


def assert_ready(plan, path=None):
    """The gate. Called before a run directory, a child process or CUDA."""
    target = Path(path or ATTESTATION_PATH)
    if not target.exists():
        raise ReadinessRefused(
            f"{target} is missing; this plan has no running permit. Generate it "
            f"with `python {GENERATOR}` after the preflights pass")
    record = load(target)
    if record.get("plan_sha256") != plan["plan_sha256"]:
        raise ReadinessRefused(
            f"the attestation permits plan {record.get('plan_sha256')}, not "
            f"{plan['plan_sha256']}")
    not_ready = [name for name, value in record["readiness"].items() if not value]
    if not_ready:
        raise ReadinessRefused(f"the attestation reports {not_ready} false")
    drifted = []
    for dataset, entry in record["data"].items():
        registered = plan["data"][dataset]
        for field in ("manifest_sha256", "payload_sha256", "unseen_sha256"):
            if registered[field] != entry[field]:
                drifted.append(f"{dataset}.{field}")
    for name, entry in record["configs"].items():
        if plan["configs"][name]["sha256"] != entry["sha256"]:
            drifted.append(f"{name} config")
    if drifted:
        raise ReadinessRefused(
            f"inputs have drifted since the attestation: {sorted(drifted)}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight", default=None,
                        help="JSON file holding the ten CPU preflight results")
    arguments = parser.parse_args(argv)

    plan = five.load_plan()
    if arguments.check:
        record = assert_ready(plan)
        print(f"attestation {record['attestation_sha256'][:16]} permits plan "
              f"{record['plan_sha256'][:16]}")
        print(f"  readiness: {record['readiness']}")
        print("  reuse:     forbidden (clean 30-run reproduction)")
        print(f"  preflight: {record['cpu_preflight'].get('combinations')} "
              f"combinations, all_passed="
              f"{record['cpu_preflight'].get('all_passed')}")
        return 0

    preflight = None
    if arguments.preflight:
        preflight = json.loads(Path(arguments.preflight).read_text("utf-8"))
    record = build(plan, preflight)
    ATTESTATION_PATH.write_text(
        json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n",
        "utf-8")
    print(f"wrote {ATTESTATION_PATH.relative_to(ROOT)}")
    print(f"attestation_sha256 = {record['attestation_sha256']}")
    print(f"readiness = {record['readiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
