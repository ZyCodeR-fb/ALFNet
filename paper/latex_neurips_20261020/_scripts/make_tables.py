"""Generate the LaTeX tabular bodies for the ALFNet paper from frozen result files.

Run from the repository root:

    python paper/latex_neurips_20261020/_scripts/make_tables.py
    python paper/latex_neurips_20261020/_scripts/make_tables.py --check

Only saved tables/JSON/CSV are read; nothing is trained or evaluated. Ranking marks
(red bold = best, blue underline = second) are computed on unrounded values within
each dataset/metric column of the displayed rows.

Tables 1-2 (main comparison): test set, mean ± sample std (ddof=1) over seeds
32/42/52 of the frozen three-seed studies, per scenario (Unseen/Steady/Join/Fault),
each cell "MAE / RMSE / MAPE" (LifeAvg and the phases are in the appendix tables,
_scripts/make_appendix.py). Table 3 (ablation): seed 32, LifeAvg, five ablations.
`--write-md` replaces the HTML tables of the Chinese v6 draft with the generated
ones; `--check` verifies that the draft contains exactly the generated HTML.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "paper", "latex_neurips_20261020", "tables")
DATASETS = ["METRLA", "PEMSBAY", "PEMS04", "PEMS08"]
DISPLAY = {"METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY", "PEMS04": "PEMS04", "PEMS08": "PEMS08"}
METRICS = ["MAE", "RMSE", "MAPE"]
SEEDS = (32, 42, 52)
SCENARIOS = ["Unseen", "Steady", "Join", "Fault", "LifeAvg"]
MAIN_SCENARIOS = SCENARIOS[:4]  # columns of Tables 1-2; LifeAvg is reported in the appendix
PHASES = ["Cold", "Warm", "Ready", "Onset", "Decay", "Outage"]
# Comparison methods grouped as in Section 2 (chronological inside a group); label = venue, year.
GROUPS = [
    ("未观测位置的归纳预测", [("MoGERNN", "TR-C, 2025"), ("SLPF", "TMLR, 2025")]),
    ("不完整观测下的预测", [("GinAR", "KDD, 2024"), ("Merlin", "KDD, 2025")]),
    ("传感网络变化与跨路网泛化", [("STONE", "KDD, 2024"), ("MCPST", "arXiv, 2026"), ("STBP", "ICLR, 2026")]),
    ("时空预测主干", [("STGCN", "IJCAI, 2018"), ("STID", "CIKM, 2022"), ("PatchSTG", "KDD, 2025"),
                ("MetaDG", "AAAI, 2026")]),
]
BASELINES = [m for _, members in GROUPS for m, _ in members]
VENUE = {m: v for _, members in GROUPS for m, v in members}
MAIN_TABLES = [("main_speed", ["METRLA", "PEMSBAY"]), ("main_flow", ["PEMS04", "PEMS08"])]
UNIT = {"METRLA": "速度", "PEMSBAY": "速度", "PEMS04": "流量", "PEMS08": "流量"}
# Ablations grouped by the component they test: (row label, source, key in the frozen file).
ABLATION_GROUPS = [
    ("OS 表示", [("w/o OS", "ext", "w/o OS")]),
    ("SAHC", [("w/o SAHC", "ext", "w/o SAHC")]),
    ("ALFTemporal", [("w/o Spatial Context", "four", "w/o Spatial Context"),
                     ("w/o ALFTemporal–FiLM", "four", "w/o ALFTemporal–FiLM")]),
    ("ALFGraph", [("w/o Adaptive Routing", "four", "w/o Adaptive Routing")]),
]

SRC = {
    "manifest": "paper/pems08_wd_test_20261012/selected_configuration_manifest.json",
    "ablation": "paper/pems08_wd_ablation_test_20261012/four_dataset_results.json",
    "hparams": "paper/final_figures_20261017/validation/fig5_values.csv",
}
DATA_MANIFEST = "configs/paper/v4_data_manifests/{}_run_manifest.json"
# Node and time-step counts come from the recorded payload shape [T, N, C]. For
# PEMS-BAY the nominal date range has 52,128 five-minute slots while the canonical
# payload has 52,116 rows; the payload shape is what the experiments used.


def path(rel):
    return os.path.join(ROOT, *rel.split("/"))


def sha256(rel):
    h = hashlib.sha256()
    with open(path(rel), "rb") as handle:
        h.update(handle.read())
    return h.hexdigest()


def load(rel):
    with open(path(rel), encoding="utf-8") as handle:
        return json.load(handle)


def marks(values):
    """Return {key: 'b'|'u'|''} for best/second (lower is better, unrounded)."""
    order = sorted(values, key=values.get)
    out = {key: "" for key in values}
    out[order[0]] = "b"
    if len(order) > 1:
        out[order[1]] = "u"
    return out


def cell(value, mark, digits=3):
    text = f"{value:.{digits}f}"
    if mark == "b":
        return f"\\textcolor{{red}}{{\\textbf{{{text}}}}}"
    if mark == "u":
        return f"\\textcolor{{blue}}{{\\underline{{{text}}}}}"
    return text


def html_cell(value, mark, digits=3):
    text = f"{value:.{digits}f}"
    if mark == "b":
        return f'<b style="color:#c00000">{text}</b>'
    if mark == "u":
        return f'<u style="color:#1f4e9c">{text}</u>'
    return text


def three_seed():
    """(plans, mean, std): mean[ds][method][cell][metric] over SEEDS on the test set (frozen records)."""
    sys.path.insert(0, ROOT)
    from scripts import report_seeds_overview2_20261110 as rep
    tp1, tp2, data = rep.collect()
    mean, std = {}, {}
    for ds in DATASETS:
        mean[ds], std[ds] = {}, {}
        for m in BASELINES + ["ALFNet"]:
            mean[ds][m], std[ds][m] = {}, {}
            for group, cells in (("main", SCENARIOS), ("phase", PHASES)):
                for c in cells:
                    vals = {x: [data[("test", ds, m, s)][group][c][x] for s in SEEDS] for x in METRICS}
                    mean[ds][m][c] = {x: statistics.fmean(v) for x, v in vals.items()}
                    std[ds][m][c] = {x: statistics.stdev(v) for x, v in vals.items()}
    return {"first": tp1["plan_sha256"], "second": tp2["plan_sha256"]}, mean, std


def scenario_table(pair, mean, std):
    """One table for two datasets (stacked panels): rows = grouped methods, columns = scenarios."""
    rows = BASELINES + ["ALFNet"]
    mk = {ds: {c: {x: marks({r: mean[ds][r][c][x] for r in rows}) for x in METRICS} for c in MAIN_SCENARIOS}
          for ds in pair}
    n = len(MAIN_SCENARIOS) + 1
    tex = ["\\begin{tabular}{l" + "c" * len(MAIN_SCENARIOS) + "}", "\\toprule",
           "方法 & " + " & ".join(MAIN_SCENARIOS) + " \\\\"]
    html = ["<table>", "  <thead>", "    <tr><th>方法</th>" + "".join(f"<th>{c}</th>" for c in MAIN_SCENARIOS) + "</tr>",
            "  </thead>", "  <tbody>"]

    def label(r):
        return r if r == "ALFNet" else f"{r} ({VENUE[r]})"

    for ds in pair:
        tex += ["\\midrule", f"\\multicolumn{{{n}}}{{c}}{{{DISPLAY[ds]}（{UNIT[ds]}）}} \\\\"]
        html.append(f'    <tr><td colspan="{n}" align="center"><i>{DISPLAY[ds]}（{UNIT[ds]}）</i></td></tr>')
        for _, members in GROUPS + [("", [("ALFNet", "")])]:
            tex.append("\\midrule")
            for r, _ in members:
                tcells, hcells = [], []
                for c in MAIN_SCENARIOS:
                    tcells.append(" / ".join(
                        cell(mean[ds][r][c][x], mk[ds][c][x][r], 2) + f"{{\\tiny$\\pm${std[ds][r][c][x]:.2f}}}"
                        for x in METRICS))
                    hcells.append(" / ".join(
                        html_cell(mean[ds][r][c][x], mk[ds][c][x][r], 2) + f"<sub>±{std[ds][r][c][x]:.2f}</sub>"
                        for x in METRICS))
                tex.append(f"{label(r)} & " + " & ".join(tcells) + " \\\\")
                html.append(f"    <tr><td>{label(r)}</td>" + "".join(f"<td>{h}</td>" for h in hcells) + "</tr>")
    tex += ["\\bottomrule", "\\end{tabular}"]
    html += ["  </tbody>", "</table>"]
    return "\n".join(tex) + "\n", "\n".join(html)


def comparison_summary(mean):
    """Counts and gains quoted in the text (ALFNet vs the best of the 11 comparison methods, on means)."""
    out = {}
    for ds in DATASETS:
        cells = {}
        for c in SCENARIOS + PHASES:
            for x in METRICS:
                best = min(BASELINES, key=lambda m: mean[ds][m][c][x])
                a, b = mean[ds]["ALFNet"][c][x], mean[ds][best][c][x]
                cells[f"{c}/{x}"] = {"alfnet": a, "best_baseline": best, "best_value": b,
                                     "gain_percent": 100 * (1 - a / b), "first": a < b}
        count = lambda names: sum(cells[f"{c}/{x}"]["first"] for c in names for x in METRICS)
        out[ds] = {"first_scenarios_of_12": count(SCENARIOS[:4]), "first_lifeavg_of_3": count(["LifeAvg"]),
                   "first_phases_of_18": count(PHASES), "first_all_of_33": count(SCENARIOS + PHASES),
                   "cells": cells}
    return out


def phase_gain_csv(summary):
    """Input for the phase figure (drawn separately): ALFNet vs the best comparison method per phase cell."""
    lines = ["dataset,phase,metric,alfnet_mean,best_baseline,best_baseline_mean,gain_percent"]
    for ds in DATASETS:
        for c in PHASES:
            for x in METRICS:
                e = summary[ds]["cells"][f"{c}/{x}"]
                lines.append(f"{ds},{c},{x},{e['alfnet']:.6f},{e['best_baseline']},{e['best_value']:.6f},"
                             f"{e['gain_percent']:.4f}")
    return "\n".join(lines) + "\n"


def ablation_table():
    """Seed 32, test LifeAvg; every variant is paired with the same final checkpoint configuration."""
    sys.path.insert(0, ROOT)
    from scripts import report_extension_20261103 as extension
    four = load(SRC["ablation"])["tables"]["test"]
    ext = extension.load()[3]["test"]
    rows = [("ALFNet", "four", "ALFNet")] + [r for _, members in ABLATION_GROUPS for r in members]
    values = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for name, src, key in rows:
            entry = (four if src == "four" else ext)[ds][key]
            values[ds][name] = {met: entry["main"]["LifeAvg"][met] for met in METRICS}
        full = ext[ds]["Full"]["main"]["LifeAvg"]
        if any(full[met] != values[ds]["ALFNet"][met] for met in METRICS):
            raise ValueError(f"{ds}: the two ablation studies use different full models")
    names = [r[0] for r in rows]
    m = {ds: {met: marks({r: values[ds][r][met] for r in names}) for met in METRICS} for ds in DATASETS}
    tex = ["\\begin{tabular}{ll" + "rrr" * len(DATASETS) + "}", "\\toprule"]
    head = ["\\multirow{2}{*}{对应组件}", "\\multirow{2}{*}{变体}"]
    head += [f"\\multicolumn{{3}}{{c}}{{{DISPLAY[ds]}}}" for ds in DATASETS]
    tex.append(" & ".join(head) + " \\\\")
    tex.append(" ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(DATASETS))))
    tex.append(" & & " + " & ".join(["MAE$\\downarrow$", "RMSE$\\downarrow$", "MAPE(\\%)$\\downarrow$"]
                                    * len(DATASETS)) + " \\\\")
    html = ["<table>", "  <thead>",
            '    <tr><th rowspan="2">对应组件</th><th rowspan="2">变体</th>'
            + "".join(f'<th colspan="3">{DISPLAY[ds]}</th>' for ds in DATASETS) + "</tr>",
            "    <tr>" + "<th>MAE↓</th><th>RMSE↓</th><th>MAPE(%)↓</th>" * len(DATASETS) + "</tr>",
            "  </thead>", "  <tbody>"]
    for group, members in [("—", [("ALFNet", "four", "ALFNet")])] + ABLATION_GROUPS:
        tex.append("\\midrule")
        for i, (name, _, _) in enumerate(members):
            first = (f"\\multirow{{{len(members)}}}{{*}}{{{group}}}" if len(members) > 1 else group) if i == 0 else ""
            tex.append(f"{first} & {name} & " + " & ".join(
                cell(values[ds][name][met], m[ds][met][name]) for ds in DATASETS for met in METRICS) + " \\\\")
            html.append(f"    <tr><td>{group if i == 0 else ''}</td><td>{name}</td>" + "".join(
                f"<td>{html_cell(values[ds][name][met], m[ds][met][name])}</td>"
                for ds in DATASETS for met in METRICS) + "</tr>")
    tex += ["\\bottomrule", "\\end{tabular}"]
    html += ["  </tbody>", "</table>"]
    change = {ds: {name: {met: 100 * (values[ds][name][met] / values[ds]["ALFNet"][met] - 1) for met in METRICS}
                   for name in names[1:]} for ds in DATASETS}
    return "\n".join(tex) + "\n", "\n".join(html), {"lifeavg": values, "change_percent_vs_full": change}


def dataset_table(provenance):
    lines = ["\\begin{tabular}{lrrll}", "\\toprule",
             "数据集 & 节点数 & 时间点数 & 预测变量 & 空间分区依据 \\\\", "\\midrule"]
    layout_name = {"kdtree_on_verified_wgs84": "地理坐标，KD-tree",
                   "graph_native_component_bfs": "图拓扑，BFS"}
    variable = {"v": "速度", "q": "流量"}
    for ds in DATASETS:
        rel = DATA_MANIFEST.format(ds)
        m = load(rel)
        steps, nodes, _ = m["payload"]["shape"]
        if nodes != m["canonical_sensor_ids"]["count"]:
            raise ValueError(f"{ds}: payload nodes {nodes} != sensor count")
        start = dt.datetime.fromisoformat(m["time"]["start"][:19])
        end = dt.datetime.fromisoformat(m["time"]["end"][:19])
        derived = int((end - start).total_seconds() // 300) + 1
        provenance["datasets"][ds] = {
            "manifest": rel, "manifest_sha256": sha256(rel), "nodes": nodes,
            "time_range": [m["time"]["start"], m["time"]["end"]],
            "steps_from_range": derived, "steps_displayed": steps,
            "steps_source": "manifest payload.shape[0]",
            "unseen_count": m["unseen"]["count"],
            "layout_algorithm": m["layout"]["algorithm"],
        }
        lines.append(f"{DISPLAY[ds]} & {nodes} & {steps:,} & {variable[m['target']['name']]} & "
                     f"{layout_name[m['layout']['algorithm']]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def power10(value):
    exponent = round(math.log10(value))
    if not math.isclose(value, 10.0 ** exponent, rel_tol=1e-9):
        raise ValueError(f"not a power of ten: {value}")
    return f"$10^{{{exponent}}}$"


def plain(value):
    text = f"{value:g}"
    return text


def settings_table(provenance):
    man = load(SRC["manifest"])["entries"]
    lines = ["\\begin{tabular}{lrrrrlrr}", "\\toprule",
             "数据集 & 分区数 & 分区容量 & 学习率倍率 & 权重衰减 & 窗口插值 & 解码残差倍率 & 中心化强度 \\\\",
             "\\midrule"]
    for ds in DATASETS:
        e = man[ds]
        hp = e["hyperparameters"]
        parts = 2 ** load(DATA_MANIFEST.format(ds))["layout"]["recur_times"]
        lines.append(" & ".join([
            DISPLAY[ds], str(parts), str(hp["capacity"]), plain(hp["lr_multiplier"]),
            power10(hp["weight_decay"]), "开启" if e["interpolation"] else "关闭",
            plain(e["decoder_scale"]), plain(e["center_alpha"])]) + " \\\\")
        provenance["settings"][ds] = {"key": e["key"], "partitions": parts, **{
            k: hp[k] for k in ("capacity", "lr_multiplier", "graph_lr", "temporal_lr", "gate_lr",
                               "dropout", "weight_decay", "batch_size", "max_epoch", "patience",
                               "early_stopping_start_epoch", "loss")},
            "interpolation": e["interpolation"], "decoder_scale": e["decoder_scale"],
            "center_alpha": e["center_alpha"], "checkpoint_sha256": e["checkpoint_sha256"]}
        report = os.path.join(os.path.dirname(path(e["checkpoint"])), "run_report.json")
        if os.path.exists(report):
            with open(report, encoding="utf-8") as handle:
                provenance["settings"][ds]["parameter_count"] = json.load(handle).get("parameter_count")
            provenance["settings"][ds]["run_report"] = os.path.relpath(report, ROOT).replace(os.sep, "/")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def hparam_table(provenance):
    rows = list(csv.DictReader(open(path(SRC["hparams"]), encoding="utf-8")))
    axes = ["capacity", "lr", "dropout", "weight_decay"]
    found = {r["axis"] for r in rows}
    alias = {a: a for a in found}
    if "lr" not in found:
        alias["lr"] = next(a for a in found if a.startswith("lr"))
    grid = {}
    for ds in DATASETS:
        grid[ds] = {}
        for axis in axes:
            sel = [r for r in rows if r["dataset"] == ds and r["axis"] == alias.get(axis, axis)]
            values = sorted({float(r["parameter"]) for r in sel})
            anchors = [v for v in values if all(abs(float(r["change"])) < 1e-12
                                                for r in sel if float(r["parameter"]) == v)]
            if len(anchors) != 1:
                raise ValueError(f"{ds}/{axis}: anchor not unique {anchors}")
            grid[ds][axis] = {"values": values, "anchor": anchors[0]}
    provenance["hyperparameter_grid"] = grid

    def fmt(axis, v):
        if axis == "capacity":
            return str(int(v))
        if axis == "weight_decay":
            return power10(v)
        return plain(v)

    lines = ["\\begin{tabular}{llllll}", "\\toprule",
             "数据集 & 锚点（容量／LR／dropout／权重衰减） & 容量候选 & LR 候选 & dropout 候选 & 权重衰减候选 \\\\",
             "\\midrule"]
    for ds in DATASETS:
        g = grid[ds]
        anchor = "／".join(fmt(a, g[a]["anchor"]) for a in axes)
        cands = ["、".join(fmt(a, v) for v in g[a]["values"]) for a in axes]
        lines.append(" & ".join([DISPLAY[ds], anchor] + cands) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


DRAFT = "paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md"


def draft_tables(text):
    """{table number: (start, end)} of the HTML table that follows each '**表N' caption in the draft."""
    found = {}
    for cap in re.finditer(r"^\*\*表(\d)\u3000", text, flags=re.M):
        start = text.index("<table>", cap.end())
        found[int(cap.group(1))] = (start, text.index("</table>", start) + len("</table>"))
    return found


def sync_draft(html, write):
    """Compare (or with write=True replace) the draft's HTML tables with the generated ones."""
    text = open(path(DRAFT), encoding="utf-8").read()
    problems = 0
    for number in sorted(html, reverse=True):
        span = draft_tables(text).get(number)
        if span is None:
            print(f"[表{number}] caption or <table> not found in the draft")
            problems += 1
        elif text[span[0]:span[1]] != html[number]:
            if write:
                text = text[:span[0]] + html[number] + text[span[1]:]
                print(f"[表{number}] draft table replaced")
            else:
                print(f"[表{number}] draft differs from the generated table")
                problems += 1
        else:
            print(f"[表{number}] draft table equals the generated table")
    if write:
        with open(path(DRAFT), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-md", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    provenance = {"generated_by": "paper/latex_neurips_20261020/_scripts/make_tables.py",
                  "sources": {k: {"path": v, "sha256": sha256(v)} for k, v in SRC.items()},
                  "datasets": {}, "settings": {}}
    plans, mean, std = three_seed()
    summary = comparison_summary(mean)
    outputs, html = {}, {}
    for number, (name, pair) in enumerate(MAIN_TABLES, start=1):
        outputs[name + ".tex"], html[number] = scenario_table(pair, mean, std)
    outputs["ablation_lifeavg.tex"], html[len(MAIN_TABLES) + 1], ablation = ablation_table()
    outputs.update({
        "datasets.tex": dataset_table(provenance),
        "settings.tex": settings_table(provenance),
        "hparam_grid.tex": hparam_table(provenance),
    })
    provenance["three_seed_comparison"] = {
        "seeds": list(SEEDS), "statistics": "mean and sample std (ddof=1) over seeds, test set",
        "test_plans_sha256": plans, "methods": BASELINES + ["ALFNet"],
        "groups": [[g, [m for m, _ in members]] for g, members in GROUPS], "venues": VENUE, "summary": summary}
    provenance["ablation_seed32"] = ablation
    for name, text in outputs.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("% Generated by _scripts/make_tables.py from frozen result files; do not edit by hand.\n")
            handle.write(text)
    with open(os.path.join(OUT, "phase_gain_3seed.csv"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(phase_gain_csv(summary))
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
    print("wrote", ", ".join(sorted(outputs)), "phase_gain_3seed.csv and provenance.json")
    for ds in DATASETS:
        print(ds, {k: v for k, v in summary[ds].items() if k != "cells"})
    if args.check or args.write_md:
        sys.exit(1 if sync_draft(html, args.write_md) else 0)


if __name__ == "__main__":
    main()
