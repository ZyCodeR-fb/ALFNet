"""Tables for the 2026-11-03 extension study, from frozen records only (no inference).

Reads the training plan's collected validation tables, the test plan's records,
the two reused w/o SAHC runs and the existing eight baselines + ALFNet
(paper/pems08_wd_test_20261012/selected_tables.json, read only). Writes:

* paper/extension_baselines_20261103/, paper/extension_ablations_20261103/
  (README, results_and_provenance.json, 33-cell long CSV, LifeAvg summary);
* paper/latex_neurips_20261020/results_extension_20261103/{comparison,ablation}_20261103.xlsx
  and first_place_counts.json.

First place follows paper/pems08_wd_test_20261012: ALFNet is first in a cell when
it is strictly lower than every other model in the compared set.
"""
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, sha, write  # noqa: E402
from scripts import run_extension_20261103 as training  # noqa: E402
from scripts import test_extension_20261103 as testing  # noqa: E402

SELECTED = ROOT / 'paper/pems08_wd_test_20261012/selected_tables.json'
EXISTING = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP')
NEW = ('Merlin', 'HDTTS', 'STONE', 'STOP')
DISPLAY = {'HDTTS': 'HD-TTS'}
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
DS_LABEL = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
SECTIONS = {'main': ('Unseen', 'Steady', 'Join', 'Fault', 'LifeAvg'),
            'phase': ('Cold', 'Warm', 'Ready', 'Onset', 'Decay', 'Outage')}
METRICS = ('MAE', 'RMSE', 'MAPE')
CELLS = [(g, c, m) for g, cs in SECTIONS.items() for c in cs for m in METRICS]
SPLITS = ('test', 'validation')
LATEX_DIR = ROOT / 'paper/latex_neurips_20261020/results_extension_20261103'
BASE_DIR = ROOT / 'paper/extension_baselines_20261103'
ABL_DIR = ROOT / 'paper/extension_ablations_20261103'
THRESHOLD = 0.8


def name(m):
    return DISPLAY.get(m, m)


def load():
    tp = training.check()
    xp = testing.check()
    xroot = testing.output(xp)
    assert read(xroot / 'COMPLETE.json')['passed']
    existing = read(SELECTED)
    tables = {s: {d: {} for d in DATASETS} for s in SPLITS}
    ablations = {s: {d: {} for d in DATASETS} for s in SPLITS}
    provenance = {}
    for key, e in xp['entries'].items():
        t, v = read(xroot / f'{key}.json'), read(xroot / 'validation' / f'{key}.json')
        assert t['passed'] and v['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256']
        assert t['split'] == 'test' and t['plan_sha256'] == xp['plan_sha256']
        target = tables if e['kind'] == 'baseline' else ablations
        label = name(e['model']) if e['kind'] == 'baseline' else e['label']
        target['validation'][e['dataset']][label] = e['table']
        target['test'][e['dataset']][label] = t['table']
        provenance[key] = dict(dataset=e['dataset'], model=e['model'], kind=e['kind'], label=label,
                               config=e['path'], config_sha256=e['sha256'], checkpoint=e['checkpoint'],
                               checkpoint_sha256=e['checkpoint_sha256'], best_epoch=e['best_epoch'],
                               epochs_run=e['epochs_run'], training_report=e['report'],
                               teacher_report=e.get('teacher_report'),
                               validation_replay_max_abs_difference=v['max_abs_difference'],
                               test_record=str((xroot / f'{key}.json').relative_to(ROOT)),
                               test_samples=t['test_samples'])
    for key, e in xp['reuse'].items():
        ablations['validation'][e['dataset']][e['label']] = e['validation_table']
        ablations['test'][e['dataset']][e['label']] = e['test_table']
        provenance[key] = {k: v for k, v in e.items() if not k.endswith('_table')} | dict(kind='ablation', reused=True)
    for s in SPLITS:
        for d in DATASETS:
            for m in EXISTING + ('ALFNet',):
                tables[s][d][m] = existing[s][d][m]
            ablations[s][d]['Full'] = existing[s][d]['ALFNet']
            assert existing['validation'][d]['ALFNet']['main']['LifeAvg']['MAE'] == \
                tp['anchors'][d]['table']['main']['LifeAvg']['MAE']
    return tp, xp, tables, ablations, provenance


def value(t, cell):
    g, c, m = cell
    return t[g][c][m]


def ranks(models):
    """Per cell: models ordered by value (lower is better)."""
    return {cell: sorted(models, key=lambda m: value(models[m], cell)) for cell in CELLS}


def alfnet_first(models):
    others = [m for m in models if m != 'ALFNet']
    return [cell for cell in CELLS if all(value(models['ALFNet'], cell) < value(models[o], cell) for o in others)]


def best_other(models, cell):
    return min((value(models[o], cell), o) for o in models if o != 'ALFNet')


def comparison_sets():
    sets = {'existing 8 (current paper)': EXISTING,
            'existing 8 + all 4 new': EXISTING + NEW}
    for a, b in itertools.product(('Merlin', 'HDTTS'), ('STONE', 'STOP')):
        sets[f'existing 8 + {name(a)} + {name(b)}'] = EXISTING + (a, b)
    for n in NEW:
        sets[f'existing 8 + {name(n)}'] = EXISTING + (n,)
    return sets


def counts(tables):
    out = {}
    for split in SPLITS:
        out[split] = {}
        for label, members in comparison_sets().items():
            per = {}
            for d in DATASETS:
                models = {m: tables[split][d][name(m)] for m in members} | {'ALFNet': tables[split][d]['ALFNet']}
                firsts = alfnet_first(models)
                lost = {}
                for cell in CELLS:
                    if cell not in firsts:
                        lost.setdefault(best_other(models, cell)[1], []).append('/'.join(cell[1:]))
                per[d] = dict(first=len(firsts), of=len(CELLS), meets_80=len(firsts) >= THRESHOLD * len(CELLS),
                              cells_won_by_others={name(k): v for k, v in lost.items()})
            total = sum(v['first'] for v in per.values())
            out[split][label] = dict(per_dataset=per, total=total, of=len(CELLS) * len(DATASETS),
                                     share=total / (len(CELLS) * len(DATASETS)),
                                     all_datasets_meet_80=all(v['meets_80'] for v in per.values()),
                                     total_meets_80=total >= THRESHOLD * len(CELLS) * len(DATASETS))
    return out


# --- workbooks ---------------------------------------------------------------


def styled_sheet(wb, title, header):
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet(title)
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDE6F0')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.freeze_panes = 'C2'
    return ws


def mark_rank(ws, rows, col, values):
    """Bold the lowest, underline the second lowest in one column block."""
    from openpyxl.styles import Font
    order = sorted(range(len(values)), key=lambda i: values[i])
    ws.cell(rows[order[0]], col).font = Font(bold=True, color='C00000')
    if len(order) > 1:
        ws.cell(rows[order[1]], col).font = Font(underline='single')


def write_wide(ws, dataset_models, order, sections):
    """Rows: dataset x model; columns: condition/metric. Rank marks per dataset block."""
    cols = [(g, c, m) for g in sections for c in SECTIONS[g] for m in METRICS]
    for d in DATASETS:
        models = dataset_models[d]
        names = [m for m in order if m in models]
        start = ws.max_row + 1
        for m in names:
            ws.append([DS_LABEL[d], m] + [round(value(models[m], cell), 4) for cell in cols])
        rows = list(range(start, start + len(names)))
        for j, cell in enumerate(cols, start=3):
            mark_rank(ws, rows, j, [value(models[m], cell) for m in names])
            for r in rows:
                ws.cell(r, j).number_format = '0.000'
    return cols


def comparison_workbook(tables, count):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    readme = wb.active
    readme.title = '说明'
    for line in (
            '补充对比实验（2026-11-03 计划 5867446b，测试计划 4742c349）：seed 32，单次运行。',
            '新增 4 个方法：Merlin、HD-TTS（2.2 不完整观测）；STONE、STOP（2.3 网络演化/固定参数泛化）。',
            '现有 8 个基线与 ALFNet 数值只读自 paper/pems08_wd_test_20261012/selected_tables.json。',
            '检查点按验证集 33 格 LifeAvg MAE 选择；测试每个检查点只评价一次，未参与任何选择。',
            '格式：每个数据集块内每列最低值红色加粗，次低值下划线。',
            '"第一"定义同 2026-10-12 汇总：ALFNet 在该格严格低于比较集中所有其他方法。',
            '80% 门槛：每数据集 33 格中至少 27 格（26.4 向上取整），四数据集合计至少 106/132 格。'):
        readme.append([line])
    readme.column_dimensions['A'].width = 120
    order = EXISTING + tuple(name(n) for n in NEW) + ('ALFNet',)
    order = tuple(name(m) for m in order)
    for split in SPLITS:
        ws = styled_sheet(wb, f'{split}_LifeAvg', ['dataset', 'model', 'MAE', 'RMSE', 'MAPE'])
        for d in DATASETS:
            models = {name(k): v for k, v in tables[split][d].items()}
            start = ws.max_row + 1
            names = [m for m in order if m in models]
            for m in names:
                ws.append([DS_LABEL[d], m] + [round(models[m]['main']['LifeAvg'][x], 4) for x in METRICS])
            rows = list(range(start, start + len(names)))
            for j, x in enumerate(METRICS, start=3):
                mark_rank(ws, rows, j, [models[m]['main']['LifeAvg'][x] for m in names])
        for g in SECTIONS:
            ws = styled_sheet(wb, f'{split}_{g}', ['dataset', 'model'] +
                              [f'{c}/{m}' for c in SECTIONS[g] for m in METRICS])
            write_wide(ws, {d: {name(k): v for k, v in tables[split][d].items()} for d in DATASETS}, order, [g])
    for split in SPLITS:
        ws = styled_sheet(wb, f'{split}_first_counts', ['comparison set'] + [DS_LABEL[d] for d in DATASETS] +
                          ['total /132', 'share', 'every dataset ≥27/33', 'total ≥106/132'])
        for label, c in count[split].items():
            ws.append([label] + [c['per_dataset'][d]['first'] for d in DATASETS] +
                      [c['total'], round(c['share'], 4), 'yes' if c['all_datasets_meet_80'] else 'no',
                       'yes' if c['total_meets_80'] else 'no'])
        ws.column_dimensions['A'].width = 34
        ws = styled_sheet(wb, f'{split}_cells_lost', ['comparison set', 'dataset', 'winner', 'cells'])
        for label in ('existing 8 + all 4 new',):
            for d in DATASETS:
                for winner, cells in count[split][label]['per_dataset'][d]['cells_won_by_others'].items():
                    ws.append([label, DS_LABEL[d], winner, ', '.join(cells)])
        ws.column_dimensions['D'].width = 120
    # Relative LifeAvg gain of ALFNet over the strongest other method (positive = ALFNet better).
    ws = styled_sheet(wb, 'test_LifeAvg_gain', ['comparison set', 'dataset', 'MAE gain %', 'RMSE gain %',
                                               'MAPE gain %', 'strongest other (MAE)'])
    for label, members in comparison_sets().items():
        for d in DATASETS:
            models = {m: tables['test'][d][name(m)] for m in members} | {'ALFNet': tables['test'][d]['ALFNet']}
            gains = []
            for x in METRICS:
                best = min(models[o]['main']['LifeAvg'][x] for o in models if o != 'ALFNet')
                gains.append(round(100 * (1 - models['ALFNet']['main']['LifeAvg'][x] / best), 3))
            strongest = min((models[o]['main']['LifeAvg']['MAE'], o) for o in models if o != 'ALFNet')[1]
            ws.append([label, DS_LABEL[d]] + gains + [name(strongest)])
            for j, gval in enumerate(gains, start=3):
                if gval < 0:
                    ws.cell(ws.max_row, j).font = Font(color='C00000')
    ws.column_dimensions['A'].width = 34
    return wb


def ablation_workbook(ablations):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    readme = wb.active
    readme.title = '说明'
    for line in (
            '补充消融（2026-11-03）：w/o SAHC（去除图循环补全、窗口内插值与参考中心化）、w/o OS（所有可学习计算不接收观测角色、状态进度与掩码）。',
            'Full 为当前最终 ALFNet（PEMS08 权重衰减 1e-3），数值只读自 selected_tables.json；seed 32、训练预算与配对初始化相同。',
            'METR-LA、PEMS-BAY 的 w/o SAHC 复用 2026-10-06 的 no_imputer（两数据集本无插值与中心化，干预等价），测试记录来自 2026-10-10。',
            'w/o OS 保持训练目标不变（含角色均衡项），只检验前向计算中的条件化；掩码仅保留补全时保留观测值与损失/评价/教师强制的目标掩码。',
            'delta% = (变体 − Full) / Full × 100；正值表示去除后误差变大（模块有益）。'):
        readme.append([line])
    readme.column_dimensions['A'].width = 130
    order = ('Full', 'w/o SAHC', 'w/o OS')
    for split in SPLITS:
        ws = styled_sheet(wb, f'{split}_LifeAvg', ['dataset', 'variant', 'MAE', 'RMSE', 'MAPE',
                                                  'ΔMAE %', 'ΔRMSE %', 'ΔMAPE %', 'worse cells /33'])
        for d in DATASETS:
            models = ablations[split][d]
            full = models['Full']
            for v in order:
                if v not in models:
                    continue
                t = models[v]['main']['LifeAvg']
                deltas = [None if v == 'Full' else round(100 * (t[x] / full['main']['LifeAvg'][x] - 1), 3) for x in METRICS]
                worse = None if v == 'Full' else sum(value(models[v], c) > value(full, c) for c in CELLS)
                ws.append([DS_LABEL[d], v] + [round(t[x], 4) for x in METRICS] + deltas + [worse])
                for j, dv in enumerate(deltas, start=6):
                    if dv is not None and dv < 0:
                        ws.cell(ws.max_row, j).font = Font(color='C00000')
        for g in SECTIONS:
            ws = styled_sheet(wb, f'{split}_{g}', ['dataset', 'variant'] +
                              [f'{c}/{m}' for c in SECTIONS[g] for m in METRICS])
            write_wide(ws, ablations[split], order, [g])
    return wb


def long_rows(split, d, label, t):
    for g, c, m in CELLS:
        yield dict(split=split, dataset=d, model=label, section=g, condition=c, metric=m, value=t[g][c][m])


def main():
    tp, xp, tables, ablations, provenance = load()
    count = counts(tables)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    comparison_workbook(tables, count).save(LATEX_DIR / 'comparison_20261103.xlsx')
    ablation_workbook(ablations).save(LATEX_DIR / 'ablation_20261103.xlsx')
    write(LATEX_DIR / 'first_place_counts.json', count)
    for folder, kind, data in ((BASE_DIR, 'baseline', tables), (ABL_DIR, 'ablation', ablations)):
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for s in SPLITS for d in DATASETS for label, t in data[s][d].items() for r in long_rows(s, d, label, t)]
        with (folder / 'cells_33_long.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lifeavg = [dict(split=s, dataset=d, model=label, **{x: t['main']['LifeAvg'][x] for x in METRICS})
                   for s in SPLITS for d in DATASETS for label, t in data[s][d].items()]
        with (folder / 'lifeavg_summary.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lifeavg[0]))
            w.writeheader()
            w.writerows(lifeavg)
        write(folder / 'results_and_provenance.json', dict(
            training_plan='configs/paper/extension_20261103/plan.json', training_plan_sha256=tp['plan_sha256'],
            test_plan='configs/paper/extension_test_20261103/plan.json', test_plan_sha256=xp['plan_sha256'],
            existing_tables=str(SELECTED.relative_to(ROOT)), existing_tables_sha256=sha(SELECTED),
            runs={k: v for k, v in provenance.items() if v['kind'] == kind},
            tables={s: data[s] for s in SPLITS},
            first_place_counts=count if kind == 'baseline' else None, test_accessed=True,
            note='seed 32 single run; test evaluated once per validation-selected checkpoint'))
    print('written', LATEX_DIR)
    for split in SPLITS:
        for label, c in count[split].items():
            print(split, f'{label:34s}', [c['per_dataset'][d]['first'] for d in DATASETS], c['total'], f"{c['share']:.3f}")


if __name__ == '__main__':
    main()
