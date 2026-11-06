"""Mean and standard deviation over seeds 32/42/52 of the 2026-11-06 three-seed study.

From frozen records only (no inference): the test plan's entries and per-run test
records (seeds 42/52 and any retrained seed 32) and the reused seed-32 tables the
plans carry. std is the sample standard deviation (ddof=1) over the three seeds.
Writes paper/seeds_20261106/{seeds_mean_std.json, seeds_long.csv, README_CN.md}
and paper/latex_neurips_20261020/results_seeds_20261106/seeds_{test,validation}_mean_std.xlsx.
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, write  # noqa: E402
from scripts import run_seeds_20261106 as training  # noqa: E402
from scripts import test_seeds_20261106 as testing  # noqa: E402

SECTIONS = {'main': ('Unseen', 'Steady', 'Join', 'Fault', 'LifeAvg'),
            'phase': ('Cold', 'Warm', 'Ready', 'Onset', 'Decay', 'Outage')}
METRICS = ('MAE', 'RMSE', 'MAPE')
CELLS = [(g, c, m) for g, cs in SECTIONS.items() for c in cs for m in METRICS]
DS_LABEL = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
OUT = ROOT / 'paper/seeds_20261106'
LATEX = ROOT / 'paper/latex_neurips_20261020/results_seeds_20261106'
SEEDS = (32, 42, 52)


def tables():
    tp = testing.check()
    troot = testing.output(tp)
    assert read(troot / 'COMPLETE.json')['passed']
    found = {}
    for key, e in tp['entries'].items():
        t = read(troot / f'{key}.json')
        assert t['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256'] and t['seed'] == e['seed']
        found[(e['dataset'], e['model'], e['seed'])] = dict(validation=e['table'], test=t['table'],
                                                            source=str((troot / f'{key}.json').relative_to(ROOT)),
                                                            reused=False)
    for e in tp['reuse'].values():
        assert (e['dataset'], e['model'], 32) not in found
        found[(e['dataset'], e['model'], 32)] = dict(validation=e['validation_table'], test=e['test_table'],
                                                     source=e['source'], reused=True)
    return tp, found


def main():
    tp, found = tables()
    summary, rows = {}, []
    for split in ('test', 'validation'):
        for ds in training.DATASETS:
            for m in training.MODELS:
                runs = [found[(ds, m, s)] for s in SEEDS]
                for g, c, metric in CELLS:
                    values = [r[split][g][c][metric] for r in runs]
                    mean, std = statistics.fmean(values), statistics.stdev(values)
                    summary.setdefault(split, {}).setdefault(ds, {}).setdefault(m, {})[f'{c}/{metric}'] = dict(
                        values=dict(zip(map(str, SEEDS), values)), mean=mean, std=std)
                    rows.append([split, ds, m, g, c, metric, *values, mean, std])
    sources = {f'{ds}_{m}_seed{s}': dict(source=v['source'], reused=v['reused']) for (ds, m, s), v in found.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'seeds_mean_std.json', dict(test_plan_sha256=tp['plan_sha256'],
                                            training_plan_sha256=tp['training_plan_sha256'], seeds=list(SEEDS),
                                            std='sample standard deviation (ddof=1) over three seeds',
                                            summary=summary, sources=sources))
    with (OUT / 'seeds_long.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'dataset', 'model', 'group', 'cell', 'metric', *[f'seed{s}' for s in SEEDS], 'mean', 'std'])
        w.writerows(rows)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    LATEX.mkdir(parents=True, exist_ok=True)
    for split in ('test', 'validation'):
        wb = Workbook()
        wb.remove(wb.active)
        for ds in training.DATASETS:
            ws = wb.create_sheet(DS_LABEL[ds])
            ws.append(['Model'] + [f'{c} {metric}' for _, c, metric in CELLS])
            for m in training.MODELS:
                cells = summary[split][ds][m]
                ws.append([m] + [f"{cells[f'{c}/{metric}']['mean']:.3f} ± {cells[f'{c}/{metric}']['std']:.3f}"
                                 for _, c, metric in CELLS])
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True)
            ws.column_dimensions['A'].width = 12
        wb.save(LATEX / f'seeds_{split}_mean_std.xlsx')
    reused = sum(v['reused'] for v in found.values())
    (OUT / 'README_CN.md').write_text(
        '# 对比算法三种子结果（2026-11-06）\n\n'
        f'- 方法：{", ".join(training.MODELS)}；数据集：{", ".join(DS_LABEL[d] for d in training.DATASETS)}；种子 32/42/52。\n'
        f'- 训练计划 `{tp["training_plan_sha256"][:8]}`，测试计划 `{tp["plan_sha256"][:8]}`；'
        f'新测试 {len(tp["entries"])} 个检查点（各一次），复用种子32表 {reused} 个。\n'
        '- 种子同时决定 K/V/T 节点划分；每个种子用与种子32完全相同的配置文件（同 SHA256）。\n'
        '- 例外：GinAR 在 METR-LA、PEMS-BAY 上 lr .006 在种子42发散（诊断见 artifacts/experiments/seeds_20261106/diagnostics/），'
        '按 2026-09-27 流量数据集先例统一改为 lr .001（四个数据集一致），三个种子均新训练；论文现有 lr .006 种子32结果未改动、不计入本均值。\n'
        '- 全部运行带非有限梯度步守卫（无事件时为空操作；有事件的运行在 seeds_mean_std.json 来源中可查）。\n'
        '- 数值为 mean ± std，std 为三种子样本标准差（ddof=1）。\n'
        '- `seeds_mean_std.json` 含每格三种子原值与来源；`seeds_long.csv` 为长表。\n')
    print('wrote', OUT, LATEX, 'reused seed-32 tables', reused)


if __name__ == '__main__':
    main()
