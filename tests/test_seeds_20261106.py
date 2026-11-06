"""Static checks of the frozen 2026-11-06 three-seed plan (no training, no test access)."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/paper/seeds_20261106/plan.json'


def test_plan_covers_ten_methods_four_datasets_three_seeds():
    p = json.loads(PLAN.read_text())
    pairs = {(c['dataset'], c['model']) for c in p['configs'].values()}
    assert len(pairs) == 40 and len(p['reuse']) == 40
    assert collections.Counter(c['seed'] for c in p['configs'].values()) == {32: 40, 42: 40, 52: 40}
    assert all(c['conditional'] == (c['seed'] == 32 and not c['lr_recovery']) for c in p['configs'].values())
    assert {(c['dataset'], c['model']) for c in p['configs'].values() if c['lr_recovery']} == {
        ('METRLA', 'GinAR'), ('PEMSBAY', 'GinAR')}
    assert 'ALFNet' not in {m for _, m in pairs}


def test_every_seed_uses_the_seed32_config_bytes():
    """Except GinAR on METR-LA/PEMS-BAY: lr .006 -> .001 only, all three seeds retrained."""
    p = json.loads(PLAN.read_text())
    for c in p['configs'].values():
        r = p['reuse'][f"{c['dataset']}_{c['model']}_seed32"]
        assert p['files'][r['checkpoint']] == r['checkpoint_sha256'] and c['sha256'] == p['files'][c['path']]
        if c['lr_recovery']:
            old, new = (ROOT / r['config']).read_text().splitlines(), (ROOT / c['path']).read_text().splitlines()
            changed = [(a.strip(), b.strip()) for a, b in zip(old, new) if a != b]
            assert len(old) == len(new) and changed[-1] == ('lr: 0.006', 'lr: 0.001') and len(changed) == 2
        else:
            assert c['path'] == r['config'] and c['sha256'] == r['config_sha256']


def test_no_historical_seed_run_is_reused():
    p = json.loads(PLAN.read_text())
    assert p['historical_seed_runs'] and not any(r['reusable'] for r in p['historical_seed_runs'])
