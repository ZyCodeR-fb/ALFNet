"""Expert weights of the OS router as a function of the observation state (no data, no forward pass).

The final ALFNet uses the dense soft router (`moge_router_mode: soft`): gates = softmax(state @ w_gate) with
state = [one-hot role (ON, NEW, UNO, OFF), state progress, observation mask] and no bias, so the gate of every
router is an exact function of the six state channels. For each final checkpoint (4 datasets x seeds 32/42/52)
this script evaluates that function on the states the loader produces
(data_provider/v4_lifecycle_loader.py::_lifecycle_state):
  ON  (progress 1, mask 1), UNO (progress 0, mask 0),
  NEW (progress p, mask 1) and OFF (progress p, mask 0) for p = 1/12 ... 12/12,
averages the gates over the routers of the graph core and writes one row per (dataset, seed, state, expert).
Output: paper/experiments_20261112/router/router_gates.csv (+ per-router file and audit json).
"""
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from layers.lifecycle_moge import LEGACY_SUPPORT_NAMES, LIFECYCLE_STATE_CHANNELS  # noqa: E402
from data_provider.lifecycle_roles import ROLE_ORDER  # noqa: E402

OUT = ROOT / 'paper/experiments_20261112/router'
PLAN = ROOT / 'configs/paper/seeds2_test_20261110/plan.json'
ROLE_LABEL = {'aas': 'ON', 'nas': 'NEW', 'vs': 'UNO', 'fs': 'OFF'}
STEPS = 12


def states():
    """[(label, role, progress, mask)] in the order used by the figure."""
    rows = [('ON', 'aas', 1.0, 1.0), ('UNO', 'vs', 0.0, 0.0)]
    rows += [(f'NEW p={k}/{STEPS}', 'nas', k / STEPS, 1.0) for k in range(1, STEPS + 1)]
    rows += [(f'OFF p={k}/{STEPS}', 'fs', k / STEPS, 0.0) for k in range(1, STEPS + 1)]
    return rows


def vector(role, progress, mask):
    x = torch.zeros(LIFECYCLE_STATE_CHANNELS, dtype=torch.float64)
    x[ROLE_ORDER.index(role)] = 1.0
    x[4], x[5] = progress, mask
    return x


def checkpoints():
    p = json.loads(PLAN.read_text())
    out = []
    for group in (p['entries'], p['reuse']):
        for e in group.values():
            if e['model'] == 'ALFNet':
                out.append((e['dataset'], int(e.get('seed', 32)), e['checkpoint']))
    return sorted(out)


def main():
    assert tuple(ROLE_ORDER) == ('aas', 'nas', 'vs', 'fs') and LIFECYCLE_STATE_CHANNELS == 6
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_router, audit = [], [], []
    for dataset, seed, path in checkpoints():
        cp = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        assert cp['dataset'] == dataset and int(cp['seed']) == seed
        assert tuple(cp['model_method_contract']['graph_support_names']) == tuple(LEGACY_SUPPORT_NAMES)
        gates = {k: v.double() for k, v in cp['model_state'].items() if k.endswith('router.w_gate')}
        assert gates and all(tuple(v.shape) == (6, len(LEGACY_SUPPORT_NAMES)) for v in gates.values())
        audit.append(dict(dataset=dataset, seed=seed, checkpoint=path, routers=len(gates), epoch=int(cp['epoch'])))
        for label, role, progress, mask in states():
            x = vector(role, progress, mask)
            g = torch.stack([torch.softmax(x @ w, dim=0) for w in gates.values()])
            for name, row in zip(gates, g):
                for expert, value in zip(LEGACY_SUPPORT_NAMES, row.tolist()):
                    per_router.append(dict(dataset=dataset, seed=seed, router=name, state=label, expert=expert, gate=value))
            for expert, value in zip(LEGACY_SUPPORT_NAMES, g.mean(0).tolist()):
                rows.append(dict(dataset=dataset, seed=seed, state=label, role=ROLE_LABEL[role], progress=progress,
                                 mask=mask, expert=expert, gate=value))
    for name, data in (('router_gates.csv', rows), ('router_gates_per_router.csv', per_router)):
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (OUT / 'audit.json').write_text(json.dumps(dict(
        router_mode='soft (configs: moge_router_mode: soft, moge_top_k: 1)', data_access='none', checkpoints=audit),
        indent=1))
    print(len(rows), 'rows;', len(audit), 'checkpoints')


if __name__ == '__main__':
    main()
