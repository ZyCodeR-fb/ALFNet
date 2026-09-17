"""CPU/std-lib only: read existing validation artifacts; never load a model/data split.

Run with the verified ReaCH Python. JSON goes to stdout; source files are read-only.
Bins are descriptive native-speed intervals, not changed evaluation masks.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / 'paper/experiments_evidence_snapshot.json'
PREDICTIONS = ROOT / 'outputs_v4/reports/observation_support_v1/paired_predictions.csv'
CHECKPOINT = Path(__file__).with_name('CHECKPOINT_20260912.json')


def main():
    checkpoint = json.loads(CHECKPOINT.read_text())
    mismatch = [f['path'] for f in checkpoint['files']
                if hashlib.sha256((ROOT / f['path']).read_bytes()).hexdigest() != f['sha256']]
    snapshot = json.loads(SNAPSHOT.read_text())
    contrasts = []
    for run in snapshot['runs']:
        if run['variant'] not in ('mask_only', 'static_graph', 'local_temporal'):
            continue
        full = snapshot['main'][run['dataset']]
        for group, rows in run['table'].items():
            for row, metrics in rows.items():
                contrasts.append(dict(dataset=run['dataset'], control=run['variant'],
                    group=group, row=row, full=full[group]['ALFNet'][row], control_metrics=metrics,
                    full_error_increase_pct={k: 100 * (full[group]['ALFNet'][row][k] / v - 1)
                                             for k, v in metrics.items()}))
    buckets = defaultdict(list)
    paired = {}
    max_mape_difference = 0.0
    for r in csv.DictReader(PREDICTIONS.open()):
        target, prediction = json.loads(r['target']), json.loads(r['prediction'])
        assert len(target) == len(prediction) == 12
        key = (r['condition'], r['pair_id'])
        identity = (target, r['mask'], r['node'], r['origin'], r['track'])
        assert key not in paired or paired[key] == identity
        paired[key] = identity
        apes = []
        for horizon, (y, p) in enumerate(zip(target, prediction), 1):
            assert math.isfinite(y) and math.isfinite(p) and y >= 0
            err = p - y
            ape = abs(err) / max(abs(y), 1e-6) * 100
            apes.append(ape)
            speed_bin = ('[0,20)' if y < 20 else '[20,40)' if y < 40
                         else '[40,60)' if y < 60 else '[60,inf)')
            for bin_name in ('all', speed_bin):
                buckets[(r['track'], r['condition'], r['model'], bin_name)].append(
                    (err, ape, r['pair_id'], r['node']))
        max_mape_difference = max(max_mape_difference, abs(sum(apes) / 12 - float(r['MAPE'])))
    assert max_mape_difference < 1e-4
    bins = []
    for (track, condition, model, bin_name), values in sorted(buckets.items()):
        n = len(values)
        total = len(buckets[(track, condition, model, 'all')])
        bins.append(dict(track=track, cohort='held-out' if track == 'join' else 'known',
            condition=condition, model=model, speed_bin=bin_name, cells=n,
            windows=len({v[2] for v in values}), nodes=len({v[3] for v in values}),
            MAE=sum(abs(v[0]) for v in values)/n,
            RMSE=math.sqrt(sum(v[0]**2 for v in values)/n),
            bias=sum(v[0] for v in values)/n, MAPE=sum(v[1] for v in values)/n,
            MAPE_contribution_pp=sum(v[1] for v in values)/total))
    output = dict(scope='existing seed32 validation artifacts only; no inference; descriptive bins',
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (SNAPSHOT, PREDICTIONS, CHECKPOINT, Path(__file__))},
        checkpoint_files_checked=len(checkpoint['files']), checkpoint_mismatches=mismatch,
        paired_targets_masks_metadata_equal=True, max_saved_window_MAPE_difference=max_mape_difference,
        contrasts=contrasts, speed_bins=bins)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
