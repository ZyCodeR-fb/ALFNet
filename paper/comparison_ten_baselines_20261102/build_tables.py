"""Merge saved ST-FiT / Merlin cells into the final comparison tables (offline only).

No training, inference or test access: every value is read from files already
under paper/. ALFNet and the eight Table-1 baselines come from the final
selection (pems08_wd_test_20261012); ST-FiT and Merlin come from the
2026-10-01 unified evaluation (before_model_filter_20261001), whose shared
baseline cells must match the final tables exactly before anything is merged.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font

PAPER = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATASETS = ["METRLA", "PEMSBAY", "PEMS04", "PEMS08"]
DISPLAY = {"METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY", "PEMS04": "PEMS04", "PEMS08": "PEMS08"}
EIGHT = ["UIGNN", "GinAR", "MoGERNN", "SLPF", "STGCN", "STID", "PatchSTG", "STBP"]
ADDED = ["STFiT", "Merlin"]
ORDER = ["UIGNN", "GinAR", "Merlin", "MoGERNN", "STFiT", "SLPF", "STGCN", "STID",
         "PatchSTG", "STBP", "ALFNet"]
NAME = {"STFiT": "ST-FiT"}
MAIN = ["Unseen", "Steady", "Join", "Fault", "LifeAvg"]
PHASE = ["Cold", "Warm", "Ready", "Onset", "Decay", "Outage"]
METRICS = ["MAE", "RMSE", "MAPE"]
SOURCES = {
    "test": (PAPER / "pems08_wd_test_20261012/test_main_phase.csv",
             PAPER / "final_four_tables_20261001/before_model_filter_20261001/02_comparison_test.csv"),
    "validation": (PAPER / "pems08_wd_test_20261012/validation_main_phase.csv",
                   PAPER / "final_four_tables_20261001/before_model_filter_20261001/01_comparison_validation.csv"),
}


def load(path):
    table = {}
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["model"], row["section"], row["condition"], row["metric"])
            table[key] = float(row["value"])
    return table


def merged_split(split):
    final_path, extended_path = SOURCES[split]
    final, extended = load(final_path), load(extended_path)
    checked = 0
    for key, value in final.items():
        if key[1] in EIGHT:
            if extended.get(key) != value:
                sys.exit(f"{split}: baseline cell differs between sources: {key}")
            checked += 1
    merged = {k: v for k, v in final.items() if k[0] in DATASETS}
    for key, value in extended.items():
        if key[0] in DATASETS and key[1] in ADDED:
            merged[key] = value
    cells = [(s, c, m) for s, conds in (("main", MAIN), ("phase", PHASE)) for c in conds for m in METRICS]
    for d in DATASETS:
        for model in ORDER:
            for cell in cells:
                if (d, model) + cell not in merged:
                    sys.exit(f"{split}: missing {d} {model} {cell}")
    return merged, cells, checked


def ranks(values):
    distinct = sorted(set(values.values()))
    first = distinct[0]
    second = distinct[1] if len(distinct) > 1 else None
    return first, second


def first_counts(merged, cells, competitors):
    counts, lost = {}, {}
    for d in DATASETS:
        wins, losses = 0, []
        for cell in cells:
            a = merged[(d, "ALFNet") + cell]
            others = {m: merged[(d, m) + cell] for m in competitors}
            if all(a < v for v in others.values()):
                wins += 1
            else:
                best = min(others, key=others.get)
                losses.append((cell, best, a, others[best]))
        counts[d], lost[d] = wins, losses
    return counts, lost


def lifeavg_gain(merged, d, metric, competitors):
    a = merged[(d, "ALFNet", "main", "LifeAvg", metric)]
    others = {m: merged[(d, m, "main", "LifeAvg", metric)] for m in competitors}
    best = min(others, key=others.get)
    return best, others[best], 100.0 * (1.0 - a / others[best])


def write_table(ws, merged, d, section, conditions):
    header = ["模型"] + [f"{c}/{m}" for c in conditions for m in METRICS]
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
    for model in ORDER:
        ws.append([NAME.get(model, model)] + [merged[(d, model, section, c, m)]
                                               for c in conditions for m in METRICS])
    for col, (c, m) in enumerate(((c, m) for c in conditions for m in METRICS), start=2):
        column = {model: merged[(d, model, section, c, m)] for model in ORDER}
        first, second = ranks(column)
        for row, model in enumerate(ORDER, start=2):
            target = ws.cell(row=row, column=col)
            target.number_format = "0.000"
            if column[model] == first:
                target.font = Font(bold=True)
            elif column[model] == second:
                target.font = Font(underline="single")
    for row in range(2, len(ORDER) + 2):
        if ws.cell(row=row, column=1).value in ("ST-FiT", "Merlin"):
            ws.cell(row=row, column=1).font = Font(italic=True)
    ws.column_dimensions["A"].width = 12
    for col in range(2, len(header) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 13
    ws.freeze_panes = "B2"


def build(split):
    merged, cells, checked = merged_split(split)
    before, _ = first_counts(merged, cells, EIGHT)
    after, lost = first_counts(merged, cells, EIGHT + ADDED)
    wb = openpyxl.Workbook()
    notes = wb.active
    notes.title = "说明"
    split_cn = "测试集" if split == "test" else "验证集"
    lines = [
        f"四数据集对比（{split_cn}，seed 32）：ALFNet 与 10 个比较方法（原表 1 的 8 个 + ST-FiT、Merlin）。",
        f"ALFNet 与 8 个基线取自 {SOURCES[split][0].relative_to(PAPER).as_posix()}（论文表 1 与图 2 的依据）。",
        f"ST-FiT、Merlin 取自 {SOURCES[split][1].relative_to(PAPER).as_posix()}（2026-10-01 统一评估，计划 SHA e696f342…）。",
        f"合并前核对：两个来源共有的 8 个基线 {checked} 个数值逐一相等。",
        "第一名加粗，第二个不同数值加下划线，按未舍入值排名；MAPE 单位为 %，MAE/RMSE 为原单位，速度与流量不跨数据集比较。",
        "ST-FiT、Merlin 行名为斜体。Merlin 的教师模型在训练期可访问已知节点被遮蔽的历史，并保留节点身份参数；"
        "ST-FiT 评估含 Gumbel 随机采样（固定种子重放）。详见 paper/BASELINE_FAIRNESS_20260928.md。",
        "ST-FiT、Merlin 的权重已在 2026-10-17 清理中删除（archive/cleanup_20261017/plan.json），本表只使用已保存的结果；"
        "如需在其上补做图 3/4/6 一类分析，需要重新训练。",
        "本文件由 build_tables.py 离线生成，未训练、未推理、未访问新的测试数据。",
    ]
    for line in lines:
        notes.append([line])
    notes.column_dimensions["A"].width = 140

    summary = wb.create_sheet("第一名统计")
    summary.append(["数据集", "ALFNet 第一/33（8 个基线）", "ALFNet 第一/33（10 个比较方法）",
                    "新增方法超过 ALFNet 的格数", "其中 ST-FiT 最优", "其中 Merlin 最优", "被超过的格（条件/指标→该格最优方法）"])
    for cell in summary[1]:
        cell.font = Font(bold=True)
    for d in DATASETS:
        newly = [(cell, best, a, v) for cell, best, a, v in lost[d]
                 if all(a < merged[(d, m) + cell] for m in EIGHT)]
        st = sum(1 for _, b, _, _ in newly if b == "STFiT")
        me = sum(1 for _, b, _, _ in newly if b == "Merlin")
        detail = "；".join(f"{c[1]}/{c[2]}→{NAME.get(b, b)}" for c, b, _, _ in newly)
        summary.append([DISPLAY[d], before[d], after[d], len(newly), st, me, detail])
    summary.append([])
    summary.append(["数据集", "指标", "ALFNet LifeAvg", "最强比较方法（8 个）", "降幅%（8 个）",
                    "最强比较方法（10 个）", "降幅%（10 个）"])
    for cell in summary[summary.max_row]:
        cell.font = Font(bold=True)
    for d in DATASETS:
        for metric in METRICS:
            b8, v8, g8 = lifeavg_gain(merged, d, metric, EIGHT)
            b10, v10, g10 = lifeavg_gain(merged, d, metric, EIGHT + ADDED)
            summary.append([DISPLAY[d], metric, merged[(d, "ALFNet", "main", "LifeAvg", metric)],
                            f"{NAME.get(b8, b8)} {v8:.3f}", round(g8, 2),
                            f"{NAME.get(b10, b10)} {v10:.3f}", round(g10, 2)])
    for col, width in zip("ABCDEFG", (11, 24, 28, 24, 16, 22, 90)):
        summary.column_dimensions[col].width = width

    for d in DATASETS:
        write_table(wb.create_sheet(f"{DISPLAY[d]}_主表"), merged, d, "main", MAIN)
        write_table(wb.create_sheet(f"{DISPLAY[d]}_阶段表"), merged, d, "phase", PHASE)
    path = OUT / f"{split}_comparison_10methods.xlsx"
    wb.save(path)

    with open(OUT / f"{split}_comparison_10methods.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "section", "condition", "metric", "value"])
        for d in DATASETS:
            for model in ORDER:
                for cell in cells:
                    writer.writerow([d, NAME.get(model, model), *cell, repr(merged[(d, model) + cell])])
    return {"split": split, "baseline_cells_checked": checked,
            "first_cells_8": before, "first_cells_10": after,
            "lifeavg_gain": {d: {m: {"8": lifeavg_gain(merged, d, m, EIGHT),
                                     "10": lifeavg_gain(merged, d, m, EIGHT + ADDED)}
                                 for m in METRICS} for d in DATASETS}}


if __name__ == "__main__":
    report = [build("test"), build("validation")]
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in report:
        print(r["split"], "checked", r["baseline_cells_checked"], "first8", r["first_cells_8"], "first10", r["first_cells_10"])
        for d in DATASETS:
            print("  ", d, {m: (r["lifeavg_gain"][d][m]["8"][0], round(r["lifeavg_gain"][d][m]["8"][2], 2),
                               r["lifeavg_gain"][d][m]["10"][0], round(r["lifeavg_gain"][d][m]["10"][2], 2))
                           for m in METRICS})
