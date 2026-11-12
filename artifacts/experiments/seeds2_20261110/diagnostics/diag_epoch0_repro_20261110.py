"""Diagnostic (2026-11-10): is the failed ALFNet epoch-0 reproduction (plan 643ed7a4, PEMS-BAY/PEMS08)
run-to-run GPU nondeterminism or a systematic path difference?

Re-runs scripts/run_seeds2_20261110.reproduce_epoch0 (same frozen plan, same context and guard; save=False,
nothing written to runs/) and records the per-batch training objective of every batch, so independent
reproductions can be compared with each other and with the original epoch-0 record. No test access.
Usage: python diag_epoch0_repro_20261110.py DATASET TAG
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from scripts import run_seeds2_20261110 as study  # noqa: E402
from experiments import v4_runner  # noqa: E402

ds, tag = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
p = study.check()
losses = []
original = v4_runner.compute_losses


def logged(output, batch, loss_name='mse', scaler=None):
    r = original(output, batch, loss_name, scaler=scaler)
    if torch.is_grad_enabled() and r['optimize'] is not None:
        losses.append(float(r['optimize'].detach()))
    return r


with patch.object(v4_runner, 'compute_losses', logged):
    result = study.reproduce_epoch0(p, ds)
out = Path(__file__).with_name(f'epoch0_{ds}_{tag}.json')
out.write_text(json.dumps(dict(dataset=ds, tag=tag, plan_sha256=p['plan_sha256'], passed=result['passed'],
                               worst_relative=result['worst_relative'], problems=result['problems'],
                               new=result['new'], batch_losses=losses), indent=1))
print(ds, tag, result['passed'], result['worst_relative'])
