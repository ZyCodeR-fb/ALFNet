"""Advisor-facing tables in the layout of the author's example workbook (one sheet per dataset x main/phase).

From frozen records only (no inference):
1. main_phase_3seed_20261112.xlsx -- test, 12 comparison methods + ALFNet, each cell "mean ± std" over seeds
   32/42/52 (sample std, ddof=1); per column the lowest mean bold, the second lowest underlined (as in the example).
   Source: scripts/report_seeds_overview2_20261110.py collect() (2026-11-06 and 2026-11-10 three-seed studies).
2. ablation_seed32_20261112.xlsx -- test, ALFNet (Full) + the four ablations of the paper table
   (paper/pems08_wd_ablation_test_20261012/four_dataset_results.json; "w/o State Age" is shown as
   "w/o State Progress" as in tables/ablation_lifeavg.tex) + w/o SAHC, w/o OS (2026-11-03 study,
   scripts/report_extension_20261103.py load()); seed 32 only, all paired with the same final checkpoint.
Style follows the example: 宋体 11, centred, 3 decimals.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read  # noqa: E402

OUT = ROOT / 'paper/latex_neurips_20261020/results_seeds2_20261110'
DATASETS = ('METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08')
DS_LABEL = {'METRLA': 'METR-LA', 'PEMSBAY': 'PEMS-BAY', 'PEMS04': 'PEMS04', 'PEMS08': 'PEMS08'}
SECTIONS = {'main': ('Unseen', 'Steady', 'Join', 'Fault', 'LifeAvg'),
            'phase': ('Cold', 'Warm', 'Ready', 'Onset', 'Decay', 'Outage')}
METRICS = ('MAE', 'RMSE', 'MAPE')
EXAMPLE_ORDER = ('UIGNN', 'GinAR', 'MoGERNN', 'SLPF', 'STGCN', 'STID', 'PatchSTG', 'STBP')
METHODS = EXAMPLE_ORDER + ('Merlin', 'STONE', 'MCPST', 'MetaDG', 'ALFNet')
ABLATIONS = (('ALFNet', 'four', 'ALFNet'), ('w/o State Progress', 'four', 'w/o State Age'),
             ('w/o Adaptive Routing', 'four', 'w/o Adaptive Routing'), ('w/o Spatial Context', 'four', 'w/o Spatial Context'),
             ('w/o ALFTemporal–FiLM', 'four', 'w/o ALFTemporal–FiLM'), ('w/o SAHC', 'ext', 'w/o SAHC'),
             ('w/o OS', 'ext', 'w/o OS'))
FONT = '宋体'


def cols(group):
    return [(c, m) for c in SECTIONS[group] for m in METRICS]


def write_sheet(ws, rows, group, rank_values, text=False):
    """rows: [(label, [cell values])]; rank_values: [(label, [numbers used for best/second])]."""
    from openpyxl.styles import Alignment, Font
    center = Alignment(horizontal='center', vertical='center')
    ws.append(['Model'] + [f'{c}/{m}' for c, m in cols(group)])
    for label, vals in rows:
        ws.append([label] + vals)
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, size=11)
            cell.alignment = center
            if cell.row > 1 and cell.column > 1 and not text:
                cell.number_format = '0.000'
    for j in range(len(cols(group))):
        ranked = sorted((vals[j], i) for i, (_, vals) in enumerate(rank_values))
        ws.cell(ranked[0][1] + 2, j + 2).font = Font(name=FONT, size=11, bold=True)
        ws.cell(ranked[1][1] + 2, j + 2).font = Font(name=FONT, size=11, underline='single')
    ws.column_dimensions['A'].width = 22 if text else 20
    for j in range(len(cols(group))):
        ws.column_dimensions[ws.cell(1, j + 2).column_letter].width = 17 if text else 13


def notes_sheet(wb, title, lines, table=None):
    from openpyxl.styles import Alignment, Font
    ws = wb.create_sheet(title)
    for line in lines:
        ws.append([line])
    if table:
        ws.append([])
        for r in table:
            ws.append(r)
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name=FONT, size=11)
            cell.alignment = Alignment(horizontal='left' if cell.column == 1 else 'center', vertical='center')
    ws.column_dimensions['A'].width = 34
    for col in 'BCDEFGH':
        ws.column_dimensions[col].width = 14
    return ws


def three_seed():
    from openpyxl import Workbook
    from scripts import report_seeds_overview2_20261110 as rep
    tp1, tp2, data = rep.collect()
    assert tp2 is not None
    seeds = (32, 42, 52)
    # cross-check: seed-32 values of the example's methods equal the paper tables (GinAR speed differs: lr .001 here)
    sel = read(ROOT / 'paper/pems08_wd_test_20261012/selected_tables.json')['test']
    for d in DATASETS:
        for m in EXAMPLE_ORDER + ('ALFNet',):
            if m == 'GinAR' and d in ('METRLA', 'PEMSBAY'):
                continue
            for g in SECTIONS:
                for c, x in cols(g):
                    assert data[('test', d, m, 32)][g][c][x] == sel[d][m][g][c][x], (d, m, c, x)
    wb = Workbook()
    wb.remove(wb.active)
    firsts = {}
    for d in DATASETS:
        won = 0
        for g in SECTIONS:
            text_rows, rank_rows = [], []
            for m in METHODS:
                vals = [[data[('test', d, m, s)][g][c][x] for s in seeds] for c, x in cols(g)]
                means = [statistics.fmean(v) for v in vals]
                text_rows.append((m, [f'{statistics.fmean(v):.3f} ± {statistics.stdev(v):.3f}' for v in vals]))
                rank_rows.append((m, means))
            for j in range(len(cols(g))):
                a = rank_rows[-1][1][j]
                won += all(a < r[1][j] for r in rank_rows[:-1])
            write_sheet(wb.create_sheet(f'{d}_{g}'), text_rows, g, rank_rows, text=True)
        firsts[d] = won
    lifeavg = []
    for d in DATASETS:
        for x in METRICS:
            means = {m: statistics.fmean(data[('test', d, m, s)]['main']['LifeAvg'][x] for s in seeds) for m in METHODS}
            order = sorted(means, key=means.get)
            other = next(m for m in order if m != 'ALFNet')
            lifeavg.append([DS_LABEL[d], x, order.index('ALFNet') + 1, other,
                            f"{100 * (1 - means['ALFNet'] / means[other]):.2f}%"])
    notes_sheet(wb, '说明', [
        '测试集结果，三个种子（32/42/52）均值 ± 样本标准差；种子同时决定 K/V/T 节点划分。',
        '每列按均值：最低加粗，次低下划线（与示例表一致）。',
        '比较方法：示例中的 8 个 + Merlin、STONE、MCPST、MetaDG；HD-TTS、STOP 不纳入。',
        'GinAR 在 METR-LA、PEMS-BAY 三个种子均为 lr .001（lr .006 在种子 42 发散，与 PEMS04/08 统一）；'
        '因此其种子 32 数值与论文现有表（lr .006）不同，其余方法的种子 32 数值与论文现有表逐格一致（已校验）。',
        'ALFNet 四个数据集使用当前最终配置；MetaDG PEMS-BAY 种子 42 来自冻结恢复（守卫限额 10%）。',
        f'来源：测试计划 {tp1["plan_sha256"][:8]}（10 个方法）与 {tp2["plan_sha256"][:8]}（ALFNet、MCPST、MetaDG）；'
        '明细与逐种子数值见同目录 seeds_overview_20261110b.xlsx、seeds_analysis_20261112.xlsx。',
    ], table=[['ALFNet 严格第一格（均值对均值）'] + [DS_LABEL[d] for d in DATASETS] + ['合计'],
              ['第一格 / 33'] + [firsts[d] for d in DATASETS] + [f'{sum(firsts.values())} / 132'],
              [], ['LifeAvg（均值）', '指标', 'ALFNet 排名', '最强对比方法', 'ALFNet 相对降低']] + lifeavg)
    path = OUT / 'main_phase_3seed_20261112.xlsx'
    wb.save(path)
    return path, firsts


def ablation():
    from openpyxl import Workbook
    from scripts import report_extension_20261103 as extension
    four = read(ROOT / 'paper/pems08_wd_ablation_test_20261012/four_dataset_results.json')
    sel = read(ROOT / 'paper/pems08_wd_test_20261012/selected_tables.json')['test']
    final = read(ROOT / 'paper/pems08_wd_test_20261012/selected_configuration_manifest.json')['entries']
    _, _, _, ext, _ = extension.load()
    ext_plan = read(ROOT / 'configs/paper/extension_20261103/plan.json')
    wb = Workbook()
    wb.remove(wb.active)
    worse = {}
    for d in DATASETS:
        # every ablation is paired with the same final seed-32 checkpoint
        assert four['provenance']['test'][d]['ALFNet']['checkpoint_sha256'] == final[d]['checkpoint_sha256']
        assert ext_plan['anchors'][d]['checkpoint_sha256'] == final[d]['checkpoint_sha256']
        tables = {}
        for label, src, key in ABLATIONS:
            tables[label] = four['tables']['test'][d][key] if src == 'four' else ext['test'][d][key]
        for t in (tables['ALFNet'], ext['test'][d]['Full']):
            assert all(t[g][c][x] == sel[d]['ALFNet'][g][c][x] for g in SECTIONS for c, x in cols(g))
        for g in SECTIONS:
            rows = [(label, [tables[label][g][c][x] for c, x in cols(g)]) for label, _, _ in ABLATIONS]
            write_sheet(wb.create_sheet(f'{d}_{g}'), rows, g, rows)
        full = tables['ALFNet']
        for label, _, _ in ABLATIONS[1:]:
            t = tables[label]
            worse.setdefault(label, {})[d] = (
                sum(t[g][c][x] > full[g][c][x] for g in SECTIONS for c, x in cols(g)),
                [100 * (t['main']['LifeAvg'][x] / full['main']['LifeAvg'][x] - 1) for x in METRICS])
    table = [['消融相对 ALFNet 变差的格数（/33）'] + [DS_LABEL[d] for d in DATASETS]]
    for label, _, _ in ABLATIONS[1:]:
        table.append([label] + [worse[label][d][0] for d in DATASETS])
    table += [[], ['LifeAvg 相对 ALFNet 变化（MAE / RMSE / MAPE，正值=变差）'] + [DS_LABEL[d] for d in DATASETS]]
    for label, _, _ in ABLATIONS[1:]:
        table.append([label] + [' / '.join(f'{v:+.2f}%' for v in worse[label][d][1]) for d in DATASETS])
    notes_sheet(wb, '说明', [
        '测试集结果，种子 32（消融只训练了种子 32）；所有消融与同一个 ALFNet 最终检查点配对（同一节点划分、同一配置，只去掉对应组件）。',
        '每列：最低加粗，次低下划线（与示例表一致）。ALFNet 行即论文主表中的 ALFNet（种子 32）。',
        '前四项为论文表 2 的消融（来源 paper/pems08_wd_ablation_test_20261012；"w/o State Progress" 对应源数据中的 "w/o State Age"）；'
        'w/o Graph Refinement 按作者要求不展示，结果仍保留在源文件中。',
        'w/o SAHC、w/o OS 为 2026-11-03 补跑（训练计划 5867446b、测试计划 4742c349；METR-LA/PEMS-BAY 的 w/o SAHC 复用 2026-10-06 的 no_imputer 运行，验证重放一致）。',
        '源文件：paper/extension_ablations_20261103/、paper/latex_neurips_20261020/results_extension_20261103/。',
    ], table=table)
    path = OUT / 'ablation_seed32_20261112.xlsx'
    wb.save(path)
    return path, worse


def main():
    p1, firsts = three_seed()
    p2, worse = ablation()
    print(p1, firsts)
    print(p2)
    for label, v in worse.items():
        print(' ', label, {d: v[d][0] for d in DATASETS})


if __name__ == '__main__':
    main()
