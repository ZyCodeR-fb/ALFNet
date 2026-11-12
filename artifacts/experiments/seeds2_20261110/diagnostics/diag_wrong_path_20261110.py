"""Negative control (2026-11-10): PEMS08 epoch-0 reproduction on a deliberately wrong path
(flow_support_transfer adapter only, flow_unified missing), to show how far the epoch-0 training
losses move when the training path really differs. save=False; no test access."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from scripts import run_seeds2_20261110 as study  # noqa: E402

torch.set_num_threads(4)
p = study.check()
key = 'PEMS08_ALFNet_seed32'
p['configs'][key] = dict(p['configs'][key], study_context='support')  # in memory only: the wrong path
result = study.reproduce_epoch0(p, 'PEMS08')
out = Path(__file__).with_name('epoch0_PEMS08_wrong_path_support_only.json')
out.write_text(json.dumps(dict(plan_sha256=p['plan_sha256'], context='support (unified missing)', **result), indent=1))
print(result['passed'], result['problems'][:6])
