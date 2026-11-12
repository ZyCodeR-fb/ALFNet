"""Three-seed overview including ALFNet, MCPST and MetaDG, from frozen records only (no inference).

Per (dataset, method, seed in 32/42/52) the validation and test 33-cell tables come from:
* the 2026-11-06 study (scripts/report_seeds_20261106.py tables()) for UIGNN, GinAR, MoGERNN,
  SLPF, STGCN, STID, PatchSTG, STBP, Merlin, STONE (GinAR on METR-LA/PEMS-BAY at lr .001, all seeds);
* the 2026-11-10 study (scripts/test_seeds2_20261110.py plan: new entries + reused seed-32 tables)
  for ALFNet, MCPST, MetaDG. Before that test plan exists the script falls back to the seed-32
  comparison records (scripts/report_extension2_20261105.py load()) and says so (provisional);
* GinAR at lr .006 (paper run, seed 32, METR-LA/PEMS-BAY) is kept as a reference row only.
HD-TTS and STOP are not reported. First place: ALFNet strictly lower than every method of the
comparison set (8 paper baselines + Merlin + STONE + MCPST + MetaDG) -- three-seed mean vs
three-seed mean, and per seed on the identical node split (seed 32, 42, 52 separately).
Writes paper/latex_neurips_20261020/results_seeds2_20261110/{seeds_overview_20261110b.xlsx,
seeds_overview_20261110b.json}.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, write  # noqa: E402
from scripts import report_seeds_20261106 as seeds  # noqa: E402

DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
DS_LABEL = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
SEEDS = (32, 42, 52)
SPLITS = ('test', 'validation')
SECTIONS, METRICS, CELLS = seeds.SECTIONS, seeds.METRICS, seeds.CELLS
STUDY1 = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP', 'Merlin', 'STONE')
STUDY2 = ('ALFNet', 'MCPST', 'MetaDG')
COMPARISON = STUDY1 + ('MCPST', 'MetaDG')
GINAR_PAPER = 'GinAR (lr .006, paper, reference)'
ORDER = COMPARISON + (GINAR_PAPER, 'ALFNet')
OUT = ROOT / 'paper/latex_neurips_20261020/results_seeds2_20261110'
TEST_PLAN = ROOT / 'configs/paper/seeds2_test_20261110/plan.json'


def study2_tables():
    """{(dataset, method, seed): {validation, test}} of the 2026-11-10 study, or the seed-32 fallback."""
    if not TEST_PLAN.exists():
        from scripts import report_extension2_20261105 as comparison
        _, single, _ = comparison.load()
        return None, {(d, m, 32): {s: single[s][d][m] for s in SPLITS} for d in DATASETS for m in STUDY2}
    from scripts import test_seeds2_20261110 as testing
    tp = testing.check()
    troot = testing.output(tp)
    assert read(troot / 'COMPLETE.json')['passed']
    found = {}
    for key, e in tp['entries'].items():
        t = read(troot / f'{key}.json')
        assert t['passed'] and t['checkpoint_sha256'] == e['checkpoint_sha256'] and t['seed'] == e['seed']
        found[(e['dataset'], e['model'], e['seed'])] = dict(validation=e['table'], test=t['table'])
    for e in tp['reuse'].values():
        assert (e['dataset'], e['model'], 32) not in found
        found[(e['dataset'], e['model'], 32)] = dict(validation=e['validation_table'], test=e['test_table'])
    return tp, found


def collect():
    tp1, found1 = seeds.tables()
    tp2, found2 = study2_tables()
    from scripts import report_extension2_20261105 as comparison
    _, single, _ = comparison.load()
    data = {}
    for (d, m, s), v in list(found1.items()) + list(found2.items()):
        for split in SPLITS:
            data[(split, d, m, s)] = v[split]
    for split in SPLITS:
        for d in ('METRLA', 'PEMSBAY'):
            data[(split, d, GINAR_PAPER, 32)] = single[split][d]['GinAR']
    return tp1, tp2, data


def values(split, d, m, cell):
    g, c, x = cell
    return [DATA[(split, d, m, s)][g][c][x] if (split, d, m, s) in DATA else None for s in SEEDS]


def stats(vals):
    present = [v for v in vals if v is not None]
    if len(present) < len(SEEDS):
        return len(present), None, None
    return len(present), statistics.fmean(present), statistics.stdev(present)


def methods(d):
    return [m for m in ORDER if any(('test', d, m, s) in DATA for s in SEEDS)]


def first_counts(split='test'):
    """ALFNet strictly below every comparison method: mean vs mean, and per seed on the same split."""
    rules = {'mean_vs_mean': None, 'matched_seed32': 0, 'matched_seed42': 1, 'matched_seed52': 2}
    out = {}
    for rule, idx in rules.items():
        per = {}
        for d in DATASETS:
            won, lost, available = 0, {}, True
            for cell in CELLS:
                if idx is None:
                    a = stats(values(split, d, 'ALFNet', cell))[1]
                    ref = {m: stats(values(split, d, m, cell))[1] for m in COMPARISON}
                else:
                    a = values(split, d, 'ALFNet', cell)[idx]
                    ref = {m: values(split, d, m, cell)[idx] for m in COMPARISON}
                if a is None or any(v is None for v in ref.values()):
                    available = False
                    break
                best = min(ref, key=ref.get)
                if a < ref[best]:
                    won += 1
                else:
                    lost.setdefault(best, []).append('/'.join(cell[1:]))
            per[d] = dict(first=won, of=len(CELLS), cells_won_by_others=lost) if available else None
        done = [v for v in per.values() if v]
        out[rule] = dict(per_dataset=per, total=sum(v['first'] for v in done) if len(done) == len(DATASETS) else None,
                         of=len(CELLS) * len(DATASETS),
                         every_dataset_ge_27=all(v['first'] >= 27 for v in done) if len(done) == len(DATASETS) else None)
    return out


def sheet(wb, title, header, widths=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet(title)
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDE6F0')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.freeze_panes = 'C2'
    for col, w in (widths or {}).items():
        ws.column_dimensions[col].width = w
    return ws


def mark(ws, rows, col, vals):
    """Lowest mean bold red, second underlined, among rows with a three-seed mean (reference row excluded)."""
    from openpyxl.styles import Font
    ranked = sorted((v, r) for v, r in zip(vals, rows) if v is not None)
    if ranked:
        ws.cell(ranked[0][1], col).font = Font(bold=True, color='C00000')
    if len(ranked) > 1:
        ws.cell(ranked[1][1], col).font = Font(underline='single')


def fmt(mean, std):
    return None if mean is None else f'{mean:.3f} ± {std:.3f}'


def workbook(tp1, tp2, counts, provisional):
    from openpyxl import Workbook
    wb = Workbook()
    readme = wb.active
    readme.title = '说明'
    lines = [
        '三种子汇总（2026-11-10，第二版）：12 个对比方法 + ALFNet；种子 32/42/52，种子同时决定 K/V/T 节点划分。',
        f'2026-11-06 研究：训练计划 {tp1["training_plan_sha256"][:8]}，测试计划 {tp1["plan_sha256"][:8]}（UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE）。',
        ('2026-11-10 研究：' + (f'训练计划 {tp2["training_plan_sha256"][:8]}，测试计划 {tp2["plan_sha256"][:8]}（ALFNet、MCPST、MetaDG）。'
                               if tp2 else 'ALFNet、MCPST、MetaDG 尚无新种子，暂只有种子 32（暂定，不可用于结论）。')),
        'HD-TTS、STOP 不进入本报告。每个新检查点测试一次，测试前验证重放一致；测试结果未用于任何选择。',
        'GinAR 在 METR-LA、PEMS-BAY：三种子均为 lr .001（lr .006 在种子 42 发散，按 2026-09-27 流量数据集先例统一为 .001）；'
        f'论文现有 lr .006 种子 32 结果单列为 "{GINAR_PAPER}"，仅备查，不计入均值、不参与第一格比较。',
        '比较集（第一格）= UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE、MCPST、MetaDG。',
        'first_place：mean_vs_mean = ALFNet 三种子均值严格低于每个方法的三种子均值；matched_seedXX = 同一种子（同一节点划分）逐种子比较。',
        'mean/std 只在三个种子齐全时填写（std 为样本标准差 ddof=1）；格式：每个数据集块内每列最低均值红色加粗、次低下划线。',
    ]
    if provisional:
        lines.append('注意：当前为暂定版本（ALFNet/MCPST/MetaDG 缺种子 42/52），第一格计数中依赖缺失种子的规则留空。')
    for line in lines:
        readme.append([line])
    readme.column_dimensions['A'].width = 150
    for split in SPLITS:
        header = ['dataset', 'method', 'n seeds']
        for x in METRICS:
            header += [f'{x} s32', f'{x} s42', f'{x} s52', f'{x} mean', f'{x} std', f'{x} mean ± std']
        ws = sheet(wb, f'{split}_LifeAvg', header, {'A': 10, 'B': 30})
        for d in DATASETS:
            start = ws.max_row + 1
            ms = methods(d)
            means = {x: [] for x in METRICS}
            for m in ms:
                row = [DS_LABEL[d], m]
                n = None
                for x in METRICS:
                    v = values(split, d, m, ('main', 'LifeAvg', x))
                    n, mean, std = stats(v)
                    row += v + [mean, std, fmt(mean, std)]
                    means[x].append(None if m == GINAR_PAPER else mean)
                ws.append(row[:2] + [n] + row[2:])
            rows = list(range(start, start + len(ms)))
            for j, x in enumerate(METRICS):
                col = 4 + j * 6
                mark(ws, rows, col + 3, means[x])
                for r in rows:
                    for k in range(5):
                        ws.cell(r, col + k).number_format = '0.000'
        for g in SECTIONS:
            cols = [(g, c, x) for c in SECTIONS[g] for x in METRICS]
            ws = sheet(wb, f'{split}_{g}_mean_std', ['dataset', 'method', 'n seeds'] +
                       [f'{c}/{x}' for _, c, x in cols], {'A': 10, 'B': 30})
            for d in DATASETS:
                start = ws.max_row + 1
                ms = methods(d)
                grid = [[stats(values(split, d, m, cell)) for cell in cols] for m in ms]
                for m, st in zip(ms, grid):
                    ws.append([DS_LABEL[d], m, st[0][0]] + [fmt(s[1], s[2]) for s in st])
                rows = list(range(start, start + len(ms)))
                for j in range(len(cols)):
                    mark(ws, rows, 4 + j, [None if m == GINAR_PAPER else st[j][1] for m, st in zip(ms, grid)])
        ws = sheet(wb, f'{split}_long', ['dataset', 'method', 'group', 'cell', 'metric', 'seed32', 'seed42',
                                         'seed52', 'n seeds', 'mean', 'std'], {'A': 10, 'B': 30})
        for d in DATASETS:
            for m in methods(d):
                for cell in CELLS:
                    v = values(split, d, m, cell)
                    n, mean, std = stats(v)
                    ws.append([DS_LABEL[d], m, *cell, *v, n, mean, std])
    ws = sheet(wb, 'test_stability', ['dataset', 'method', 'LifeAvg MAE mean', 'std', 'CV %', 'min seed', 'max seed',
                                      'rank at seed 32', 'rank by mean', 'rank change'], {'A': 10, 'B': 12})
    cell = ('main', 'LifeAvg', 'MAE')
    for d in DATASETS:
        ms = [m for m in COMPARISON + ('ALFNet',) if stats(values('test', d, m, cell))[1] is not None]
        s32 = sorted(ms, key=lambda m: values('test', d, m, cell)[0])
        mean_rank = sorted(ms, key=lambda m: stats(values('test', d, m, cell))[1])
        for m in mean_rank:
            v = values('test', d, m, cell)
            _, mean, std = stats(v)
            ws.append([DS_LABEL[d], m, round(mean, 4), round(std, 4), round(100 * std / mean, 2),
                       f'{min(v):.3f} (s{SEEDS[v.index(min(v))]})', f'{max(v):.3f} (s{SEEDS[v.index(max(v))]})',
                       s32.index(m) + 1, mean_rank.index(m) + 1, s32.index(m) - mean_rank.index(m)])
    for split in SPLITS:
        c = counts[split]
        ws = sheet(wb, f'first_place_{split}', ['rule'] + [DS_LABEL[d] for d in DATASETS] +
                   ['total /132', 'every dataset ≥27/33'], {'A': 22})
        for rule, r in c.items():
            ws.append([rule] + [None if r['per_dataset'][d] is None else r['per_dataset'][d]['first'] for d in DATASETS]
                      + [r['total'], None if r['every_dataset_ge_27'] is None else
                         ('yes' if r['every_dataset_ge_27'] else 'no')])
        ws.append([])
        ws.append(['rule', 'dataset', 'winner', 'cells'])
        for rule, r in c.items():
            for d in DATASETS:
                for winner, cells in ((r['per_dataset'][d] or {}).get('cells_won_by_others') or {}).items():
                    ws.append([rule, DS_LABEL[d], winner, ', '.join(cells)])
        ws.column_dimensions['D'].width = 100
    ws = sheet(wb, 'test_ALFNet_vs_strongest', ['dataset', 'ALFNet LifeAvg MAE mean ± std', 'strongest other (mean)',
                                                'its mean ± std', 'gain % (mean vs mean)'],
               {'A': 10, 'B': 26, 'C': 24, 'D': 18})
    for d in DATASETS:
        a = stats(values('test', d, 'ALFNet', cell))
        if a[1] is None:
            continue
        best = min(COMPARISON, key=lambda m: stats(values('test', d, m, cell))[1])
        _, mean, std = stats(values('test', d, best, cell))
        ws.append([DS_LABEL[d], fmt(a[1], a[2]), best, fmt(mean, std), round(100 * (1 - a[1] / mean), 2)])
    return wb


def main():
    global DATA
    tp1, tp2, DATA = collect()
    counts = {split: first_counts(split) for split in SPLITS}
    provisional = tp2 is None
    OUT.mkdir(parents=True, exist_ok=True)
    workbook(tp1, tp2, counts, provisional).save(OUT / 'seeds_overview_20261110b.xlsx')
    summary = {split: {d: {m: {f'{c}/{x}': dict(zip(['n', 'mean', 'std'], stats(values(split, d, m, (g, c, x)))),
                                               values=dict(zip(map(str, SEEDS), values(split, d, m, (g, c, x)))))
                               for g, c, x in CELLS} for m in methods(d)} for d in DATASETS} for split in SPLITS}
    write(OUT / 'seeds_overview_20261110b.json', dict(
        provisional=provisional, study1_test_plan_sha256=tp1['plan_sha256'],
        study2_test_plan_sha256=None if tp2 is None else tp2['plan_sha256'], seeds=list(SEEDS),
        comparison_set=list(COMPARISON), excluded=['HD-TTS', 'STOP'], ginar_reference=GINAR_PAPER,
        std='sample standard deviation (ddof=1), only when all three seeds exist',
        first_place=counts, summary=summary))
    for split in SPLITS:
        for rule, c in counts[split].items():
            print(split, rule, [None if c['per_dataset'][d] is None else c['per_dataset'][d]['first'] for d in DATASETS],
                  c['total'])


DATA = {}

if __name__ == '__main__':
    main()
