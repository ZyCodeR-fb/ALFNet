"""Does a frozen run have exactly one complete, protocol-consistent result set?

Split out from the evaluation launcher so the bash side never has to reason
about which files a `--mode test_async` pass leaves behind, and so the launcher
and the aggregator agree on what "already evaluated" means. The rules live in
`scripts/final_runs.verify_eval`.

Run it from the active project environment (server: ReaCH, local Windows: COMG):

    python scripts/verify_eval_result.py
    python scripts/verify_eval_result.py --only alfnet-metrla-s32
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_runs import MANIFEST, load_manifest, verify_eval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--only")
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args()

    payload = load_manifest(options.manifest)
    runs = payload["runs"]
    if options.only:
        runs = [run for run in runs if run["run_id"] == options.only]
        if not runs:
            print(f"no run named {options.only!r}", file=sys.stderr)
            return 2

    failures = {}
    for run in runs:
        problems, directory = verify_eval(run, options.output_root)
        if problems:
            failures[run["run_id"]] = problems
        if not options.quiet:
            name = directory.name if directory else "-"
            print(f"{run['run_id']:<22}{'OK' if not problems else 'FAIL':<6}{name}")
            for problem in problems:
                print(f"    - {problem}")

    if options.quiet:
        return 1 if failures else 0
    print()
    if failures:
        print(f"{len(failures)}/{len(runs)} run(s) have no usable result set.")
        return 1
    print(f"all {len(runs)} run(s) have exactly one complete result set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
