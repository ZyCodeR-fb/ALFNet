"""Analysis of the completed three-seed study (ALFNet + 12 comparison methods), from frozen records only.

Inputs: scripts/report_seeds_overview2_20261110.py collect() (2026-11-06 study of 10 methods, 2026-11-10 study
of ALFNet/MCPST/MetaDG incl. recovery 694a8c24; HD-TTS and STOP not reported), plus the training/test records of
the 2026-11-10 study for the run-health sheet. No inference, no test access, no selection.
First place = ALFNet strictly lower than every method of the comparison set (8 paper baselines + Merlin + STONE
+ MCPST + MetaDG; GinAR at lr .001). Requirement used in this project: >= 80% of the 33 cells, i.e. >= 27 per dataset.
Rules: mean_vs_mean (primary: three-seed mean vs three-seed mean), matched_seed32/42/52 (same node split),
new_seeds_mean (seeds 42 and 52 only: seed 32's ALFNet configurations were retained with test feedback),
majority_matched (strictly first on >= 2 of the 3 matched seeds), mean_vs_best_seed (ALFNet mean vs each
method's best single seed; most conservative).
Writes paper/latex_neurips_20261020/results_seeds2_20261110/{seeds_analysis_20261112.xlsx, seeds_analysis_20261112.json}.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metrla_hparams import read, write  # noqa: E402
from scripts import report_seeds_overview2_20261110 as rep  # noqa: E402

DATASETS, SEEDS, CELLS, METRICS = rep.DATASETS, rep.SEEDS, rep.CELLS, rep.METRICS
DS_LABEL, COMPARISON = rep.DS_LABEL, rep.COMPARISON
OUT = rep.OUT
NEED = 27
RULES = ('mean_vs_mean', 'matched_seed32', 'matched_seed42', 'matched_seed52', 'new_seeds_mean',
         'majority_matched', 'mean_vs_best_seed')
RULE_LABEL = {
    'mean_vs_mean': '主口径：三种子均值对均值',
    'matched_seed32': '逐种子同划分：种子 32',
    'matched_seed42': '逐种子同划分：种子 42',
    'matched_seed52': '逐种子同划分：种子 52',
    'new_seeds_mean': '敏感性：只用种子 42/52 的均值（种子 32 配置保留曾参考测试）',
    'majority_matched': '稳健性：3 个同划分种子中至少 2 个严格第一',
    'mean_vs_best_seed': '最保守：ALFNet 均值对各方法最好的单个种子',
}


def V(split, d, m, cell):
    return rep.values(split, d, m, cell)


def mean(v, idx=(0, 1, 2)):
    return statistics.fmean(v[i] for i in idx)


def sd(v):
    return statistics.stdev(v)


def matched_wins(split, d, cell):
    a = V(split, d, 'ALFNet', cell)
    return [i for i in range(len(SEEDS)) if a[i] < min(V(split, d, m, cell)[i] for m in COMPARISON)]


def first(split, d, cell, rule):
    a = V(split, d, 'ALFNet', cell)
    ref = {m: V(split, d, m, cell) for m in COMPARISON}
    if rule == 'mean_vs_mean':
        return mean(a) < min(mean(v) for v in ref.values())
    if rule.startswith('matched_seed'):
        i = SEEDS.index(int(rule[-2:]))
        return a[i] < min(v[i] for v in ref.values())
    if rule == 'new_seeds_mean':
        return mean(a, (1, 2)) < min(mean(v, (1, 2)) for v in ref.values())
    if rule == 'majority_matched':
        return len(matched_wins(split, d, cell)) >= 2
    if rule == 'mean_vs_best_seed':
        return mean(a) < min(min(v) for v in ref.values())
    raise ValueError(rule)


def counts(split):
    out = {}
    for rule in RULES:
        per = {d: sum(first(split, d, c, rule) for c in CELLS) for d in DATASETS}
        out[rule] = dict(per_dataset=per, total=sum(per.values()), of=len(CELLS) * len(DATASETS),
                         every_dataset_ge_27=all(v >= NEED for v in per.values()))
    return out


def best_other(split, d, cell, idx=None):
    vals = {m: (mean(V(split, d, m, cell)) if idx is None else V(split, d, m, cell)[idx]) for m in COMPARISON}
    m = min(vals, key=vals.get)
    return m, vals[m]


def run_health():
    """Training/test records of the 24 new runs and the 12 reused seed-32 runs (2026-11-10 study)."""
    from scripts import test_seeds2_20261110 as testing
    tp = testing.check()
    troot = testing.output(tp)
    rows = []
    for key, e in sorted(tp['entries'].items(), key=lambda kv: (kv[1]['model'], kv[1]['dataset'], kv[1]['seed'])):
        t, v = read(troot / f'{key}.json'), read(troot / 'validation' / f'{key}.json')
        report = read(ROOT / e['report'])
        rows.append(dict(model=e['model'], dataset=e['dataset'], seed=e['seed'],
                         source='recovery 694a8c24 (guard limit 10%)' if key in tp['recovered'] else 'plan d6f66d20',
                         epochs_run=e['epochs_run'], best_epoch=e['best_epoch'],
                         early_stopping_start_epoch=report['early_stopping_start_epoch'],
                         guarded_steps=len(e['nonfinite_gradient_steps']), train_skipped=e['train_skipped'],
                         validation_skipped_per_epoch=','.join(map(str, e['validation_skipped'])),
                         replay_diff_before_test=v['max_abs_difference'], test_samples=t['test_samples'],
                         checkpoint=e['checkpoint']))
    for key, e in sorted(tp['reuse'].items()):
        rows.append(dict(model=e['model'], dataset=e['dataset'], seed=32,
                         source=f"reused seed 32 (training plan {e['training_plan_sha256'][:8]})", epochs_run=None,
                         best_epoch=e['epoch'], early_stopping_start_epoch=None, guarded_steps=None,
                         train_skipped=None, validation_skipped_per_epoch=None,
                         replay_diff_before_test='preflight replay 0.0', test_samples=None, checkpoint=e['checkpoint']))
    return tp, rows


def sheet(wb, title, header, widths=None, freeze='C2'):
    from openpyxl.styles import Alignment, Font, PatternFill
    ws = wb.create_sheet(title)
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='DDE6F0')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.freeze_panes = freeze
    for col, w in (widths or {}).items():
        ws.column_dimensions[col].width = w
    return ws


def build(tp, health, cnt):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    red = PatternFill('solid', fgColor='F8D7DA')
    green = PatternFill('solid', fgColor='D4EDDA')
    wb = Workbook()
    ws = wb.active
    ws.title = '说明'
    for line in (
            '三种子结果分析（2026-11-12），全部来自冻结记录，无新推理、无测试访问；只用于描述，不用于任何选择。',
            f'测试计划 {tp["plan_sha256"][:8]}（ALFNet、MCPST、MetaDG，24 个新检查点各测试一次，测试前验证重放差 0）；'
            '其余 10 个方法来自 2026-11-06 研究（测试计划 92f8956b）。',
            '比较集（第一格）：UIGNN、GinAR、MoGERNN、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、STONE、MCPST、MetaDG；HD-TTS、STOP 不纳入。',
            'GinAR 在 METR-LA/PEMS-BAY 三种子均为 lr .001；论文现有 lr .006 种子 32 仅在概览工作簿中单列备查，不参与比较。',
            '要求：每个数据集 33 格中 ALFNet 严格第一 ≥80%，即 ≥27/33（与此前各轮一致）。',
            '规则：' + '；'.join(f'{k} = {v}' for k, v in RULE_LABEL.items()),
            'ALFNet 配置：四个数据集沿用最终配置（2026-11-10 审计后作者决定保留）。种子 32 的配置保留曾参考测试结果，'
            '种子 42/52 是固定配置后的新划分重复，因此单列 new_seeds_mean 作为无选择偏差的敏感性。',
            'MetaDG PEMS-BAY 种子 42：主计划中第 0 轮 56 次非有限梯度步（2.9% > 1% 限额）失败；按冻结恢复 694a8c24'
            '（限额 10%，在看到 2.9% 后提出，已披露）重跑，第 0 轮仍为 56 次（2.87%），之后各轮 0 次。PEMS08 MetaDG 种子 52 有 1 次（1% 内）。',
            'METR-LA MCPST 每轮跳过 2 个无监督验证批（三个种子相同，公共 runner 行为）；METR-LA GinAR 种子 42 测试 Steady 异常（见 2026-11-06 记录），照实计入。',
            '训练预算差异（既有设置，非本轮引入）：ALFNet 从第 40 轮起才计早停，MoGERNN 从第 20 轮起，其余方法从第 0 轮起；最多 100 轮、patience 10 对所有方法相同。',
            '格式：绿色 = 满足 ≥27；红色 = 不满足。cell_robustness 中 margin% = ALFNet 均值相对最强对手均值的差（负值表示 ALFNet 更好）。'):
        ws.append([line])
    ws.column_dimensions['A'].width = 160
    for split in ('test', 'validation'):
        s = sheet(wb, f'first_place_{split}', ['rule', '含义'] + [DS_LABEL[d] for d in DATASETS] +
                  ['total /132', '每个数据集 ≥27'], {'A': 20, 'B': 58}, freeze='C2')
        for rule in RULES:
            c = cnt[split][rule]
            s.append([rule, RULE_LABEL[rule]] + [c['per_dataset'][d] for d in DATASETS] +
                     [c['total'], 'yes' if c['every_dataset_ge_27'] else 'no'])
            r = s.max_row
            for j, d in enumerate(DATASETS):
                s.cell(r, 3 + j).fill = green if c['per_dataset'][d] >= NEED else red
            s.cell(r, 8).fill = green if c['every_dataset_ge_27'] else red
    s = sheet(wb, 'lost_cells_test', ['dataset', 'group', 'cell', 'metric', 'ALFNet mean ± std', 'winner (mean)',
                                      'winner mean ± std', 'gap % (ALFNet vs winner)', 'ALFNet first at matched seeds'],
              {'A': 10, 'E': 18, 'F': 14, 'G': 18, 'I': 22})
    for d in DATASETS:
        for cell in CELLS:
            if first('test', d, cell, 'mean_vs_mean'):
                continue
            a = V('test', d, 'ALFNet', cell)
            w, wm = best_other('test', d, cell)
            wv = V('test', d, w, cell)
            s.append([DS_LABEL[d], *cell, f'{mean(a):.3f} ± {sd(a):.3f}', w, f'{wm:.3f} ± {sd(wv):.3f}',
                      round(100 * (mean(a) / wm - 1), 2),
                      ','.join(str(SEEDS[i]) for i in matched_wins('test', d, cell)) or 'none'])
    s = sheet(wb, 'cell_robustness_test', ['dataset', 'group', 'cell', 'metric', 'ALFNet s32', 'ALFNet s42',
                                           'ALFNet s52', 'ALFNet mean', 'strongest other (mean)', 'its mean',
                                           'margin %', 'first by mean', 'matched-seed wins (of 3)',
                                           'first, seeds 42/52 mean'], {'A': 10, 'I': 16})
    for d in DATASETS:
        for cell in CELLS:
            a = V('test', d, 'ALFNet', cell)
            w, wm = best_other('test', d, cell)
            k = len(matched_wins('test', d, cell))
            ok = first('test', d, cell, 'mean_vs_mean')
            s.append([DS_LABEL[d], *cell, *[round(x, 4) for x in a], round(mean(a), 4), w, round(wm, 4),
                      round(100 * (mean(a) / wm - 1), 2), 'yes' if ok else 'no', k,
                      'yes' if first('test', d, cell, 'new_seeds_mean') else 'no'])
            r = s.max_row
            s.cell(r, 12).fill = green if ok else red
            if ok and k <= 1:
                s.cell(r, 13).font = Font(bold=True, color='C00000')
    for split in ('test', 'validation'):
        s = sheet(wb, f'LifeAvg_ranking_{split}', ['dataset', 'method'] +
                  [f'{x} mean ± std' for x in METRICS] + [f'{x} rank' for x in METRICS], {'A': 10, 'B': 12})
        for d in DATASETS:
            ms = list(COMPARISON) + ['ALFNet']
            ranks = {x: sorted(ms, key=lambda m: mean(V(split, d, m, ('main', 'LifeAvg', x)))) for x in METRICS}
            for m in ranks['MAE']:
                vals = [V(split, d, m, ('main', 'LifeAvg', x)) for x in METRICS]
                s.append([DS_LABEL[d], m] + [f'{mean(v):.3f} ± {sd(v):.3f}' for v in vals] +
                         [ranks[x].index(m) + 1 for x in METRICS])
                if m == 'ALFNet':
                    for c in s[s.max_row]:
                        c.font = Font(bold=True)
    s = sheet(wb, 'LifeAvg_gain_test', ['dataset', 'metric', 'ALFNet mean ± std', 'strongest other', 'its mean ± std',
                                        'gain % (mean vs mean)', 'ALFNet first at seeds', 'per-seed gain % vs that seed’s best'],
              {'A': 10, 'C': 18, 'D': 14, 'E': 18, 'G': 20, 'H': 30})
    for d in DATASETS:
        for x in METRICS:
            cell = ('main', 'LifeAvg', x)
            a = V('test', d, 'ALFNet', cell)
            w, wm = best_other('test', d, cell)
            per = []
            for i in range(len(SEEDS)):
                bm, bv = best_other('test', d, cell, idx=i)
                per.append(f's{SEEDS[i]} {100 * (1 - a[i] / bv):+.2f}% ({bm})')
            s.append([DS_LABEL[d], x, f'{mean(a):.3f} ± {sd(a):.3f}', w, f'{wm:.3f} ± {sd(V("test", d, w, cell)):.3f}',
                      round(100 * (1 - mean(a) / wm), 2),
                      ','.join(str(SEEDS[i]) for i in matched_wins('test', d, cell)) or 'none', '; '.join(per)])
    s = sheet(wb, 'pairwise_LifeAvgMAE_test', ['dataset', 'method', 'its mean', 'ALFNet mean', 'ALFNet gain % (mean)',
                                               'seeds where ALFNet better (of 3)'], {'A': 10, 'B': 12})
    for d in DATASETS:
        cell = ('main', 'LifeAvg', 'MAE')
        a = V('test', d, 'ALFNet', cell)
        for m in sorted(COMPARISON, key=lambda m: mean(V('test', d, m, cell))):
            v = V('test', d, m, cell)
            s.append([DS_LABEL[d], m, round(mean(v), 4), round(mean(a), 4), round(100 * (1 - mean(a) / mean(v)), 2),
                      sum(a[i] < v[i] for i in range(len(SEEDS)))])
    s = sheet(wb, 'run_health_20261110', list(health[0]), {'A': 10, 'B': 10, 'D': 34, 'M': 60})
    for r in health:
        s.append(list(r.values()))
    return wb


def main():
    tp1, tp2, data = rep.collect()
    assert tp2 is not None, 'the 2026-11-10 test plan must exist (not the seed-32 fallback)'
    rep.DATA = data
    for d in DATASETS:
        for m in COMPARISON + ('ALFNet',):
            for cell in CELLS:
                assert None not in V('test', d, m, cell) and None not in V('validation', d, m, cell), (d, m, cell)
    cnt = {split: counts(split) for split in ('test', 'validation')}
    tp, health = run_health()
    build(tp, health, cnt).save(OUT / 'seeds_analysis_20261112.xlsx')
    lost = {d: [dict(cell='/'.join(c), winner=best_other('test', d, c)[0],
                     gap_percent=100 * (mean(V('test', d, 'ALFNet', c)) / best_other('test', d, c)[1] - 1))
                for c in CELLS if not first('test', d, c, 'mean_vs_mean')] for d in DATASETS}
    write(OUT / 'seeds_analysis_20261112.json', dict(
        study1_test_plan_sha256=tp1['plan_sha256'], study2_test_plan_sha256=tp2['plan_sha256'],
        comparison_set=list(COMPARISON), requirement='>= 27 of 33 cells per dataset', first_place=cnt,
        lost_cells_test_mean_vs_mean=lost, run_health=health))
    for split in ('test', 'validation'):
        for rule in RULES:
            c = cnt[split][rule]
            print(split, rule, [c['per_dataset'][d] for d in DATASETS], c['total'], c['every_dataset_ge_27'])


if __name__ == '__main__':
    main()
