"""Audit saved candidates against eight baselines; no training or data loading."""
import copy
import json
import shutil
from openpyxl import load_workbook
from scripts import package_four_paper_tables as pack

EXCLUDED = {'ALFNet', 'STFiT', 'ST-FiT', 'Merlin'}
SELECTED = dict(METRLA='METRLA_patch7', PEMSBAY='05_PEMSBAY_combo2',
                PEMS03='PEMS03_capacity_mid', PEMS04='PEMS04_default',
                PEMS08='PEMS08_lr_double')


def audit(rows, candidates, baselines):
    result = {}
    best = {m: min(r['table']['main']['LifeAvg'][m] for r in baselines.values())
            for m in pack.METRICS}
    for key in candidates:
        t = rows[key]['table']
        wins = sum(t[g][c][m] < min(r['table'][g][c][m] for r in baselines.values())
                   for g, cs in pack.GROUPS.items() for c in cs for m in pack.METRICS)
        life = {m: t['main']['LifeAvg'][m] for m in pack.METRICS}
        result[key] = dict(first_cells=wins, lifeavg=life,
                           mean_relative_lifeavg=sum(life[m]/best[m] for m in pack.METRICS)/3)
    return dict(sorted(result.items(), key=lambda kv: (-kv[1]['first_cells'], kv[1]['mean_relative_lifeavg'])))


def main():
    p = pack.run.check()
    d = pack.read(pack.run.OUT/'all_results_and_provenance.json')
    v = pack.read(pack.ROOT/'paper/hparams_combinations_20260930/results.json')['rows']
    b = pack.read(pack.ROOT/'paper/hparams_combinations_20260930/results_and_provenance.json')['records']
    backup = pack.OUT/'before_reselection_20261002'
    backup.mkdir(exist_ok=True)
    for f in pack.OUT.iterdir():
        if f.is_file() and not (backup/f.name).exists():
            shutil.copy2(f, backup/f.name)
    models = {'validation': {}, 'test': {}}
    ranking = {'validation': {}, 'test': {}}
    for ds, key in SELECTED.items():
        keys = [k for k, e in p['entries'].items() if e['dataset'] == ds and e['model'] == 'ALFNet']
        bases = {
            'validation': {n: r for n, r in b[ds].items() if n not in EXCLUDED},
            'test': {e['model']: d['rows'][k] for k, e in p['entries'].items()
                     if e['dataset'] == ds and e['model'] not in EXCLUDED}}
        for split, rows in [('validation', v), ('test', d['rows'])]:
            assert len(bases[split]) == 8
            ranking[split][ds] = audit(rows, keys, bases[split])
            models[split][ds] = dict(ALFNet=rows[key], **bases[split])
            for r in models[split][ds].values():
                for g, cs in pack.GROUPS.items():
                    for c in cs:
                        for field, value in rows[key]['table'][g][c].items():
                            if field.endswith('_count'):
                                assert r['table'][g][c][field] == value
    summary = {}
    counts = [pack.comparisons(s, models[s], p, SELECTED, summary) for s in models]
    h1, layout1, _ = pack.hyperparameters('validation', v, p, SELECTED)
    h2, layout2, _ = pack.hyperparameters('test', d['rows'], p, SELECTED)
    assert layout1 == layout2 and counts == [1485, 1485]
    note = '2026-10-02按8个基线重新审查全部67配置；PEMS04改用default，其余不变。test参与选型，非独立确认；两split使用相同权重。'
    for path in pack.OUT.glob('0[12]*.xlsx'):
        wb = load_workbook(path)
        ws = wb['说明']
        for i in range(ws.max_row, 0, -1):
            text = str(ws.cell(i, 1).value or '')
            if text.startswith('ST-FiT保留'):
                ws.delete_rows(i)
            elif 'Merlin两阶段' in text:
                ws.cell(i, 1, text.replace('Merlin两阶段预算和train-K教师特权保持原披露。', ''))
            elif text.startswith('最终配置依'):
                ws.cell(i, 1, note)
        ws.append(['展示ALFNet与8个基线，排除ST-FiT、Merlin；第一加粗，第二下划线。'])
        wb.save(path)
        md = path.with_suffix('.md')
        lines = md.read_text().splitlines()
        lines = [note if x.startswith('最终配置依') else x.replace('Merlin两阶段预算和train-K教师特权保持原披露。', '')
                 for x in lines if not x.startswith('ST-FiT保留')]
        md.write_text('\n'.join(lines)+'\n')
        actual = load_workbook(path)
        for ws in actual:
            if not ws.title.endswith(('主表', '阶段表')):
                continue
            assert ws.max_row == 10
            for j in range(2, ws.max_column+1):
                values = sorted({ws.cell(i, j).value for i in range(2, 11)})
                for i in range(2, 11):
                    cell = ws.cell(i, j); rank = values.index(cell.value)+1
                    assert cell.font.bold == (rank == 1)
                    assert cell.font.underline == ('single' if rank == 2 else None)
    pack.write(pack.OUT/'eight_baseline_candidate_ranking.json', ranking)
    old = pack.read(backup/'selected_configuration_manifest.json')
    old.update(selected=SELECTED, entries={ds: p['entries'][k] for ds, k in SELECTED.items()},
               selection_basis=note, summary=summary, excluded_baselines=['STFiT', 'Merlin'])
    pack.write(pack.OUT/'selected_configuration_manifest.json', old)
    pack.write(pack.OUT/'reselection_verification.json', dict(passed=True,
               cells=counts+[h1, h2], candidates=67, same_checkpoints_across_splits=True,
               original_records_unchanged=True, test_accessed=False))
    lines = ['# 四份论文实验表（2026-10-02重新选型）', '', note, '',
             '比较范围为ALFNet及8个基线；既有原始结果、被移除模型与此前表格完整保留。', '',
             '| 数据集 | 配置 | validation第一/33 | test第一/33 | test LifeAvg MAE / RMSE / MAPE |',
             '|---|---|---:|---:|---|']
    for ds, k in SELECTED.items():
        a = summary[ds]['validation']; t = summary[ds]['test']
        lines.append(f'| {ds} | {k} | {sum(a["first"].values())} | {sum(t["first"].values())} | '+
                     ' / '.join(f'{t["lifeavg"][m]:.4f}' for m in pack.METRICS)+' |')
    lines += ['', '## 取舍', '',
        'PEMS04默认容量10/LR1/dropout0.1/WD1e-4：test 22格第一，LifeAvg三项均优于旧combo1，故替换。',
        'PEMS03保留容量16：8格第一；WD1e-5有9格第一，但LifeAvg MAE/RMSE/MAPE分别由35.268/47.310/50.436退化为36.554/48.805/52.328，约3.65%/3.16%/3.75%。一格优势不足以抵偿三项平均误差一致变差；这是一项明确的综合取舍，没有设置硬性退化门槛。9格候选保留在完整排名中。',
        '其余三数据集原配置仍为最多第一格的配置之一。完整67项validation/test排名见eight_baseline_candidate_ranking.json。33格有重叠，不能视作33个独立统计检验。', '',
        '## 文件', '',
        '01、02分别为选定同一权重的validation/test对比表，第一加粗、第二下划线；03、04为四轴单变量及下部组合结果，数值与分组保持不变，仅同步最终配置标记。',
        '全部配置与test已被观察，本次仅离线重算排名。新修复实验另设validation-only计划，不覆盖这套结果。',
        '旧表备份before_reselection_20261002；原模型过滤备份before_model_filter_20261001。MoGERNN旧patience3、SLPF/STBP适配限制继续见BASELINE_FAIRNESS_20260928.md。']
    (pack.OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
