"""Seed overview workbook for every compared method, from frozen records only (no inference).

Per (dataset, method, seed in 32/42/52) the validation and test 33-cell tables come from:
* the 2026-11-06 three-seed study (scripts/report_seeds_20261106.py tables(): test plan
  entries + reused seed-32 tables) for UIGNN, GinAR, MoGERNN, SLPF, STGCN, STID,
  PatchSTG, STBP, Merlin, STONE (GinAR on METR-LA/PEMS-BAY at lr .001, all seeds);
* seed 32 only, from the frozen comparison records (scripts/report_extension2_20261105.py
  load()), for ALFNet, HD-TTS, STOP, MCPST, MetaDG and GinAR at lr .006 (paper run,
  METR-LA/PEMS-BAY).
Missing seeds stay empty; mean/std (sample std, ddof=1) only when all three seeds exist,
so the workbook can be regenerated as further seeds are added (extend collect()).
Writes paper/latex_neurips_20261020/results_seeds_20261106/seeds_overview_20261110.xlsx
and seeds_overview_20261110.json (with provisional first-place counts).
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import write  # noqa: E402
from scripts import report_seeds_20261106 as seeds  # noqa: E402
from scripts import report_extension2_20261105 as comparison  # noqa: E402

DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
DS_LABEL = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
SEEDS = (32, 42, 52)
SPLITS = ('test', 'validation')
SECTIONS = seeds.SECTIONS
METRICS = seeds.METRICS
CELLS = seeds.CELLS
THREE_SEED = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP', 'Merlin', 'STONE')
SEED32_ONLY = ('HD-TTS', 'STOP', 'MCPST', 'MetaDG')
GINAR_PAPER = 'GinAR (lr .006, paper)'
ORDER = THREE_SEED + SEED32_ONLY + (GINAR_PAPER, 'ALFNet')
OUT = ROOT / 'paper/latex_neurips_20261020/results_seeds_20261106'


def collect():
    """{(split, dataset, method, seed): table}."""
    tp, found = seeds.tables()
    _, single, _ = comparison.load()
    data = {}
    for (d, m, s), v in found.items():
        for split in SPLITS:
            data[(split, d, m, s)] = v[split]
    for split in SPLITS:
        for d in DATASETS:
            for m in SEED32_ONLY + ('ALFNet',):
                data[(split, d, m, 32)] = single[split][d][m]
            if d in ('METRLA', 'PEMSBAY'):
                data[(split, d, GINAR_PAPER, 32)] = single[split][d]['GinAR']
            else:  # PEMS04/PEMS08 GinAR has always used lr .001: same run as the three-seed seed 32
                assert single[split][d]['GinAR'] == data[(split, d, 'GinAR', 32)]
            for m in THREE_SEED:
                if not (m == 'GinAR' and d in ('METRLA', 'PEMSBAY')):
                    assert single[split][d][m] == data[(split, d, m, 32)], (split, d, m)
    return tp, data


def methods(d):
    return [m for m in ORDER if any(('test', d, m, s) in DATA for s in SEEDS)]


def values(split, d, m, cell):
    g, c, x = cell
    return [DATA[(split, d, m, s)][g][c][x] if (split, d, m, s) in DATA else None for s in SEEDS]


def stats(vals):
    present = [v for v in vals if v is not None]
    if len(present) < len(SEEDS):
        return len(present), None, None
    return len(present), statistics.fmean(present), statistics.stdev(present)


def first_counts():
    """Provisional: ALFNet has only seed 32; seeds also redraw the node split."""
    out = {}
    for rule in ('matched_seed32', 'vs_three_seed_mean', 'vs_best_seed'):
        per = {}
        for d in DATASETS:
            won, lost = 0, {}
            for cell in CELLS:
                a = values('test', d, 'ALFNet', cell)[0]
                ref = {}
                for m in THREE_SEED:
                    v = values('test', d, m, cell)
                    ref[m] = v[0] if rule == 'matched_seed32' else (stats(v)[1] if rule == 'vs_three_seed_mean' else min(v))
                best = min(ref, key=ref.get)
                if a < ref[best]:
                    won += 1
                else:
                    lost.setdefault(best, []).append('/'.join(cell[1:]))
            per[d] = dict(first=won, of=len(CELLS), cells_won_by_others=lost)
        total = sum(v['first'] for v in per.values())
        out[rule] = dict(per_dataset=per, total=total, of=len(CELLS) * len(DATASETS),
                         every_dataset_ge_27=all(v['first'] >= 27 for v in per.values()))
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
    """Lowest mean bold red, second underlined, among rows with a value."""
    from openpyxl.styles import Font
    ranked = sorted((v, r) for v, r in zip(vals, rows) if v is not None)
    if ranked:
        ws.cell(ranked[0][1], col).font = Font(bold=True, color='C00000')
    if len(ranked) > 1:
        ws.cell(ranked[1][1], col).font = Font(underline='single')


def fmt(mean, std):
    return None if mean is None else f'{mean:.3f} ± {std:.3f}'


def workbook(tp, counts):
    from openpyxl import Workbook
    wb = Workbook()
    readme = wb.active
    readme.title = '说明'
    for line in (
            f'三种子汇总（2026-11-10）。训练计划 {tp["training_plan_sha256"][:8]}，测试计划 {tp["plan_sha256"][:8]}；每个新检查点测试一次，测试前验证重放一致。',
            '种子 32/42/52，种子同时决定 K/V/T 节点划分；三种子方法：UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE。',
            '未跑的种子留空：ALFNet、HD-TTS、STOP、MCPST、MetaDG 目前只有种子 32；mean/std 只在三个种子齐全时填写（std 为样本标准差 ddof=1）。',
            'GinAR 在 METR-LA、PEMS-BAY：三种子均为 lr .001（lr .006 在种子 42 发散，按 2026-09-27 流量数据集先例统一为 .001）；'
            '论文现有 lr .006 种子 32 结果单列为 "GinAR (lr .006, paper)"，不计入均值。PEMS04/PEMS08 的 GinAR 一直是 lr .001。',
            'METR-LA GinAR 种子 42：验证 LifeAvg MAE 7.02 正常，但测试 Steady 格 MAE 50.3（其余格 6–9），LifeAvg 测试 MAE 18.30；'
            '按协议不因测试结果剔除或重跑，均值与标准差照实计入。',
            'METR-LA UIGNN（批大小 4）每轮跳过 6 个无监督验证批，三个种子相同，为公共 runner 正常行为；全部 82 个新训练无非有限梯度事件、无训练跳批。',
            '格式：每个数据集块内每列最低均值红色加粗、次低下划线（仅在有三种子均值的方法之间比较）。',
            'first_place_provisional：ALFNet 只有种子 32，且种子改变节点划分，计数仅为暂定；ALFNet 三种子补齐后应以均值对均值重算。',
            '  matched_seed32 = ALFNet 种子 32 对比各方法种子 32（同一节点划分）；vs_three_seed_mean = 对比各方法三种子均值；'
            'vs_best_seed = 对比各方法三种子中最好的值（最保守）。比较集 = 上述 10 个三种子方法。'):
        readme.append([line])
    readme.column_dimensions['A'].width = 150
    for split in SPLITS:
        # LifeAvg: per-seed values + mean/std
        header = ['dataset', 'method', 'n seeds']
        for x in METRICS:
            header += [f'{x} s32', f'{x} s42', f'{x} s52', f'{x} mean', f'{x} std', f'{x} mean ± std']
        ws = sheet(wb, f'{split}_LifeAvg', header, {'A': 10, 'B': 22})
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
                    means[x].append(mean)
                ws.append(row[:2] + [n] + row[2:])
            rows = list(range(start, start + len(ms)))
            for j, x in enumerate(METRICS):
                col = 4 + j * 6
                mark(ws, rows, col + 3, means[x])
                for r in rows:
                    for k in range(5):
                        ws.cell(r, col + k).number_format = '0.000'
        # wide mean ± std for main / phase cells
        for g in SECTIONS:
            cols = [(g, c, x) for c in SECTIONS[g] for x in METRICS]
            ws = sheet(wb, f'{split}_{g}_mean_std', ['dataset', 'method', 'n seeds'] +
                       [f'{c}/{x}' for _, c, x in cols], {'A': 10, 'B': 22})
            for d in DATASETS:
                start = ws.max_row + 1
                ms = methods(d)
                grid = [[stats(values(split, d, m, cell)) for cell in cols] for m in ms]
                for m, st in zip(ms, grid):
                    ws.append([DS_LABEL[d], m, st[0][0]] + [fmt(s[1], s[2]) for s in st])
                rows = list(range(start, start + len(ms)))
                for j in range(len(cols)):
                    mark(ws, rows, 4 + j, [st[j][1] for st in grid])
        # long format: every cell, every seed
        ws = sheet(wb, f'{split}_long', ['dataset', 'method', 'group', 'cell', 'metric', 'seed32', 'seed42',
                                         'seed52', 'n seeds', 'mean', 'std'], {'A': 10, 'B': 22})
        for d in DATASETS:
            for m in methods(d):
                for cell in CELLS:
                    v = values(split, d, m, cell)
                    n, mean, std = stats(v)
                    ws.append([DS_LABEL[d], m, *cell, *v, n, mean, std])
    # stability and ranking
    ws = sheet(wb, 'test_stability', ['dataset', 'method', 'LifeAvg MAE mean', 'std', 'CV %', 'min seed', 'max seed',
                                      'rank at seed 32', 'rank by mean', 'rank change'], {'A': 10, 'B': 12})
    for d in DATASETS:
        cell = ('main', 'LifeAvg', 'MAE')
        s32 = sorted(THREE_SEED, key=lambda m: values('test', d, m, cell)[0])
        mean_rank = sorted(THREE_SEED, key=lambda m: stats(values('test', d, m, cell))[1])
        for m in mean_rank:
            v = values('test', d, m, cell)
            _, mean, std = stats(v)
            ws.append([DS_LABEL[d], m, round(mean, 4), round(std, 4), round(100 * std / mean, 2),
                       f'{min(v):.3f} (s{SEEDS[v.index(min(v))]})', f'{max(v):.3f} (s{SEEDS[v.index(max(v))]})',
                       s32.index(m) + 1, mean_rank.index(m) + 1, s32.index(m) - mean_rank.index(m)])
    ws = sheet(wb, 'first_place_provisional', ['rule'] + [DS_LABEL[d] for d in DATASETS] +
               ['total /132', 'every dataset ≥27/33'], {'A': 26})
    for rule, c in counts.items():
        ws.append([rule] + [c['per_dataset'][d]['first'] for d in DATASETS] +
                  [c['total'], 'yes' if c['every_dataset_ge_27'] else 'no'])
    ws.append([])
    ws.append(['rule', 'dataset', 'winner', 'cells'])
    for rule, c in counts.items():
        for d in DATASETS:
            for winner, cells in c['per_dataset'][d]['cells_won_by_others'].items():
                ws.append([rule, DS_LABEL[d], winner, ', '.join(cells)])
    ws.column_dimensions['D'].width = 100
    ws = sheet(wb, 'test_ALFNet_vs_mean', ['dataset', 'ALFNet s32 LifeAvg MAE', 'strongest 3-seed mean (MAE)',
                                           'its mean ± std', 'gain % vs mean', 'its best seed', 'gain % vs best seed'],
               {'A': 10, 'C': 26, 'D': 18, 'F': 14})
    for d in DATASETS:
        cell = ('main', 'LifeAvg', 'MAE')
        a = values('test', d, 'ALFNet', cell)[0]
        best = min(THREE_SEED, key=lambda m: stats(values('test', d, m, cell))[1])
        v = values('test', d, best, cell)
        _, mean, std = stats(v)
        ws.append([DS_LABEL[d], round(a, 4), best, fmt(mean, std), round(100 * (1 - a / mean), 2), round(min(v), 4),
                   round(100 * (1 - a / min(v)), 2)])
    return wb


def main():
    global DATA
    tp, DATA = collect()
    counts = first_counts()
    OUT.mkdir(parents=True, exist_ok=True)
    workbook(tp, counts).save(OUT / 'seeds_overview_20261110.xlsx')
    summary = {split: {d: {m: {f'{c}/{x}': dict(zip(['n', 'mean', 'std'], stats(values(split, d, m, (g, c, x)))),
                                               values=dict(zip(map(str, SEEDS), values(split, d, m, (g, c, x)))))
                               for g, c, x in CELLS} for m in methods(d)} for d in DATASETS} for split in SPLITS}
    write(OUT / 'seeds_overview_20261110.json', dict(
        training_plan_sha256=tp['training_plan_sha256'], test_plan_sha256=tp['plan_sha256'], seeds=list(SEEDS),
        std='sample standard deviation (ddof=1), only when all three seeds exist',
        first_place_provisional=counts, summary=summary))
    for rule, c in counts.items():
        print(rule, [c['per_dataset'][d]['first'] for d in DATASETS], c['total'])


DATA = {}

if __name__ == '__main__':
    main()
