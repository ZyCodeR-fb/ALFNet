"""Comparison tables with the two 2026 baselines MCPST and MetaDG, from frozen records only.

Adds the 2026-11-05 study (training plan a8d285aa with recoveries 034802bd and
b136d9b6; test plan configs/paper/extension2_test_20261105) to the 2026-11-03
tables (scripts/report_extension_20261103.py, read through its load()). Writes
paper/latex_neurips_20261020/results_extension2_20261105/{comparison_20261105.xlsx,
first_place_counts.json} and paper/extension_baselines_20261105/
{results_and_provenance.json, cells_33_long.csv, lifeavg_summary.csv}.
First place as before: ALFNet strictly lower than every other model in the set.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, write  # noqa: E402
from scripts import report_extension_20261103 as first  # noqa: E402
from scripts import test_extension2_20261105 as testing  # noqa: E402

DATASETS, SPLITS, CELLS, METRICS, SECTIONS = first.DATASETS, first.SPLITS, first.CELLS, first.METRICS, first.SECTIONS
EXISTING = first.EXISTING
BATCH1 = ('Merlin', 'HD-TTS', 'STONE', 'STOP')
BATCH2 = ('MCPST', 'MetaDG')
LATEX_DIR = ROOT / 'paper/latex_neurips_20261020/results_extension2_20261105'
BASE_DIR = ROOT / 'paper/extension_baselines_20261105'
THRESHOLD = first.THRESHOLD


def load():
    _, _, tables, _, provenance1 = first.load()
    xp = testing.check()
    xroot = testing.output(xp)
    assert read(xroot / 'COMPLETE.json')['passed']
    provenance = {}
    for key, e in xp['entries'].items():
        t, v = read(xroot / f'{key}.json'), read(xroot / 'validation' / f'{key}.json')
        assert t['passed'] and v['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256']
        assert t['split'] == 'test' and t['plan_sha256'] == xp['plan_sha256']
        tables['validation'][e['dataset']][e['model']] = e['table']
        tables['test'][e['dataset']][e['model']] = t['table']
        provenance[key] = dict(dataset=e['dataset'], model=e['model'], config=e['path'], config_sha256=e['sha256'],
                               training_plan=e['training_plan'], training_plan_sha256=e['training_plan_sha256'],
                               checkpoint=e['checkpoint'], checkpoint_sha256=e['checkpoint_sha256'],
                               best_epoch=e['best_epoch'], epochs_run=e['epochs_run'], training_report=e['report'],
                               train_skipped=e['train_skipped'],
                               validation_skipped_per_epoch=e['validation_skipped_per_epoch'],
                               validation_replay_max_abs_difference=v['max_abs_difference'],
                               test_record=str((xroot / f'{key}.json').relative_to(ROOT)),
                               test_samples=t['test_samples'])
    return xp, tables, provenance


def comparison_sets():
    return {
        'existing 8 (current paper)': EXISTING,
        'existing 8 + MCPST': EXISTING + ('MCPST',),
        'existing 8 + MetaDG': EXISTING + ('MetaDG',),
        'existing 8 + MCPST + MetaDG': EXISTING + BATCH2,
        'existing 8 + Merlin + STONE + MCPST + MetaDG': EXISTING + ('Merlin', 'STONE') + BATCH2,
        'existing 8 + HD-TTS + STONE + MCPST + MetaDG': EXISTING + ('HD-TTS', 'STONE') + BATCH2,
        'existing 8 + all 6 new': EXISTING + BATCH1 + BATCH2,
    }


def counts(tables):
    out = {}
    for split in SPLITS:
        out[split] = {}
        for label, members in comparison_sets().items():
            per = {}
            for d in DATASETS:
                models = {m: tables[split][d][m] for m in members} | {'ALFNet': tables[split][d]['ALFNet']}
                firsts = first.alfnet_first(models)
                lost = {}
                for cell in CELLS:
                    if cell not in firsts:
                        lost.setdefault(first.best_other(models, cell)[1], []).append('/'.join(cell[1:]))
                per[d] = dict(first=len(firsts), of=len(CELLS), meets_80=len(firsts) >= THRESHOLD * len(CELLS),
                              cells_won_by_others=lost)
            total = sum(v['first'] for v in per.values())
            out[split][label] = dict(per_dataset=per, total=total, of=len(CELLS) * len(DATASETS),
                                     share=total / (len(CELLS) * len(DATASETS)),
                                     all_datasets_meet_80=all(v['meets_80'] for v in per.values()),
                                     total_meets_80=total >= THRESHOLD * len(CELLS) * len(DATASETS))
    return out


def workbook(tables, count):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    readme = wb.active
    readme.title = '说明'
    for line in (
            '补充对比实验第二批（2026-11-05）：MCPST（MCPST-FSL，源域训练阶段）与 MetaDG，均为 2026 年方法；seed 32，单次运行。',
            '训练计划 a8d285aa（METR-LA MCPST 来自恢复 034802bd，PEMS08 MetaDG 来自恢复 b136d9b6），测试计划 extension2_test_20261105。',
            '第一批（Merlin、HD-TTS、STONE、STOP）与现有 8 个基线、ALFNet 数值来自已冻结记录，未重新推理。',
            '检查点按验证集 33 格 LifeAvg MAE 选择；测试每个检查点只评价一次，未参与任何选择。',
            '格式：每个数据集块内每列最低值红色加粗，次低值下划线。',
            '"第一"：ALFNet 在该格严格低于比较集中所有其他方法；80% 门槛：每数据集 ≥27/33，合计 ≥106/132。'):
        readme.append([line])
    readme.column_dimensions['A'].width = 120
    order = EXISTING + BATCH1 + BATCH2 + ('ALFNet',)
    for split in SPLITS:
        ws = first.styled_sheet(wb, f'{split}_LifeAvg', ['dataset', 'model', 'MAE', 'RMSE', 'MAPE'])
        for d in DATASETS:
            models = tables[split][d]
            names = [m for m in order if m in models]
            start = ws.max_row + 1
            for m in names:
                ws.append([first.DS_LABEL[d], m] + [round(models[m]['main']['LifeAvg'][x], 4) for x in METRICS])
            rows = list(range(start, start + len(names)))
            for j, x in enumerate(METRICS, start=3):
                first.mark_rank(ws, rows, j, [models[m]['main']['LifeAvg'][x] for m in names])
        for g in SECTIONS:
            ws = first.styled_sheet(wb, f'{split}_{g}', ['dataset', 'model'] +
                                    [f'{c}/{m}' for c in SECTIONS[g] for m in METRICS])
            first.write_wide(ws, tables[split], order, [g])
    for split in SPLITS:
        ws = first.styled_sheet(wb, f'{split}_first_counts', ['comparison set'] + [first.DS_LABEL[d] for d in DATASETS] +
                                ['total /132', 'share', 'every dataset ≥27/33', 'total ≥106/132'])
        for label, c in count[split].items():
            ws.append([label] + [c['per_dataset'][d]['first'] for d in DATASETS] +
                      [c['total'], round(c['share'], 4), 'yes' if c['all_datasets_meet_80'] else 'no',
                       'yes' if c['total_meets_80'] else 'no'])
        ws.column_dimensions['A'].width = 44
        ws = first.styled_sheet(wb, f'{split}_cells_lost', ['comparison set', 'dataset', 'winner', 'cells'])
        for label in ('existing 8 + MCPST + MetaDG', 'existing 8 + Merlin + STONE + MCPST + MetaDG'):
            for d in DATASETS:
                for winner, cells in count[split][label]['per_dataset'][d]['cells_won_by_others'].items():
                    ws.append([label, first.DS_LABEL[d], winner, ', '.join(cells)])
        ws.column_dimensions['D'].width = 120
    ws = first.styled_sheet(wb, 'test_LifeAvg_gain', ['comparison set', 'dataset', 'MAE gain %', 'RMSE gain %',
                                                     'MAPE gain %', 'strongest other (MAE)'])
    for label, members in comparison_sets().items():
        for d in DATASETS:
            models = {m: tables['test'][d][m] for m in members} | {'ALFNet': tables['test'][d]['ALFNet']}
            gains = []
            for x in METRICS:
                best = min(models[o]['main']['LifeAvg'][x] for o in models if o != 'ALFNet')
                gains.append(round(100 * (1 - models['ALFNet']['main']['LifeAvg'][x] / best), 3))
            strongest = min((models[o]['main']['LifeAvg']['MAE'], o) for o in models if o != 'ALFNet')[1]
            ws.append([label, first.DS_LABEL[d]] + gains + [strongest])
            for j, gval in enumerate(gains, start=3):
                if gval < 0:
                    ws.cell(ws.max_row, j).font = Font(color='C00000')
    ws.column_dimensions['A'].width = 44
    return wb


def main():
    xp, tables, provenance = load()
    count = counts(tables)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    workbook(tables, count).save(LATEX_DIR / 'comparison_20261105.xlsx')
    write(LATEX_DIR / 'first_place_counts.json', count)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for s in SPLITS for d in DATASETS for m in BATCH2 for r in first.long_rows(s, d, m, tables[s][d][m])]
    with (BASE_DIR / 'cells_33_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lifeavg = [dict(split=s, dataset=d, model=m, **{x: tables[s][d][m]['main']['LifeAvg'][x] for x in METRICS})
               for s in SPLITS for d in DATASETS for m in BATCH2]
    with (BASE_DIR / 'lifeavg_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lifeavg[0]))
        w.writeheader()
        w.writerows(lifeavg)
    write(BASE_DIR / 'results_and_provenance.json', dict(
        test_plan=str(testing.PLAN.relative_to(ROOT)), test_plan_sha256=xp['plan_sha256'], runs=provenance,
        tables={s: {d: {m: tables[s][d][m] for m in BATCH2} for d in DATASETS} for s in SPLITS},
        first_place_counts=count, test_accessed=True,
        note='seed 32 single run; test evaluated once per validation-selected checkpoint'))
    print('written', LATEX_DIR)
    for split in SPLITS:
        for label, c in count[split].items():
            print(split, f'{label:46s}', [c['per_dataset'][d]['first'] for d in DATASETS], c['total'], f"{c['share']:.3f}")


if __name__ == '__main__':
    main()
