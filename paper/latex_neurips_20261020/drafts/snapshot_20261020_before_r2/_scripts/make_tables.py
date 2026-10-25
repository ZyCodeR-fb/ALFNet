"""Generate the LaTeX tabular bodies for the ALFNet paper from frozen result files.

Run from the repository root:

    python paper/latex_neurips_20261020/_scripts/make_tables.py
    python paper/latex_neurips_20261020/_scripts/make_tables.py --check

Only saved tables/JSON/CSV are read; nothing is trained or evaluated. Ranking marks
(bold = best, underline = second) are computed on unrounded values within each
dataset/metric column of the displayed rows. `--check` compares every displayed
number and mark with the HTML tables of the Chinese v6 draft.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "paper", "latex_neurips_20261020", "tables")
DATASETS = ["METRLA", "PEMSBAY", "PEMS04", "PEMS08"]
DISPLAY = {"METRLA": "METR-LA", "PEMSBAY": "PEMS-BAY", "PEMS04": "PEMS04", "PEMS08": "PEMS08"}
METRICS = ["MAE", "RMSE", "MAPE"]
BASELINES = ["UIGNN", "GinAR", "MoGERNN", "SLPF", "STGCN", "STID", "PatchSTG", "STBP"]
ABLATIONS = ["w/o State Age", "w/o Adaptive Routing", "w/o Spatial Context", "w/o ALFTemporal–FiLM"]

SRC = {
    "selected_tables": "paper/pems08_wd_test_20261012/selected_tables.json",
    "summary": "paper/pems08_wd_test_20261012/summary.json",
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


def cell(value, mark):
    text = f"{value:.3f}"
    if mark == "b":
        return f"\\textbf{{{text}}}"
    if mark == "u":
        return f"\\underline{{{text}}}"
    return text


def grid_table(rows, values, first_col_head):
    """rows: list of row names; values[ds][row][metric] -> float."""
    m = {ds: {met: marks({r: values[ds][r][met] for r in rows}) for met in METRICS} for ds in DATASETS}
    lines = ["\\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", "\\toprule"]
    head = [f"\\multirow{{2}}{{*}}{{{first_col_head}}}"]
    head += [f"\\multicolumn{{3}}{{c}}{{{DISPLAY[ds]}}}" for ds in DATASETS]
    lines.append(" & ".join(head) + " \\\\")
    rules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS)))
    lines.append(rules)
    lines.append(" & " + " & ".join(["MAE$\\downarrow$", "RMSE$\\downarrow$", "MAPE(\\%)$\\downarrow$"] * len(DATASETS)) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        cells = [cell(values[ds][r][met], m[ds][met][r]) for ds in DATASETS for met in METRICS]
        lines.append(f"{r} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", m


def main_table():
    test = load(SRC["selected_tables"])["test"]
    rows = BASELINES + ["ALFNet"]
    values = {ds: {r: {met: test[ds][r]["main"]["LifeAvg"][met] for met in METRICS} for r in rows} for ds in DATASETS}
    return grid_table(rows, values, "模型"), values, rows


def ablation_table():
    tab = load(SRC["ablation"])["tables"]["test"]
    rows = ["ALFNet"] + ABLATIONS
    values = {}
    for ds in DATASETS:
        values[ds] = {}
        for r in rows:
            entry = tab[ds][r]
            la = entry["main"]["LifeAvg"] if "main" in entry else entry
            values[ds][r] = {met: la[met] for met in METRICS}
    return grid_table(rows, values, "变体"), values, rows


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


def check_against_v6(name, values, mark_map, rows):
    """Compare 3-decimal values and marks with the HTML table in the v6 draft."""
    draft = open(path("paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md"), encoding="utf-8").read()
    tables = re.findall(r"<tbody>(.*?)</tbody>", draft, flags=re.S)
    html = tables[0] if name == "main" else tables[1]
    parsed = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        tds = re.findall(r"<td>(.*?)</td>", tr, flags=re.S)
        label = tds[0].strip()
        parsed[label] = []
        for td in tds[1:]:
            mk = "b" if "<b>" in td else ("u" if "<u>" in td else "")
            parsed[label].append((re.sub(r"<[^>]+>", "", td).strip(), mk))
    problems = 0
    for r in rows:
        expected = [(f"{values[ds][r][met]:.3f}", mark_map[ds][met][r]) for ds in DATASETS for met in METRICS]
        if parsed.get(r) != expected:
            problems += 1
            print(f"[{name}] mismatch {r}: draft={parsed.get(r)} generated={expected}")
    print(f"[{name}] {len(rows)} rows x 12 cells compared with v6 draft: {'OK' if not problems else problems}")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    provenance = {"generated_by": "paper/latex_neurips_20261020/_scripts/make_tables.py",
                  "sources": {k: {"path": v, "sha256": sha256(v)} for k, v in SRC.items()},
                  "datasets": {}, "settings": {}}
    (main_tex, main_marks), main_vals, main_rows = main_table()
    (abl_tex, abl_marks), abl_vals, abl_rows = ablation_table()
    outputs = {
        "main_lifeavg.tex": main_tex,
        "ablation_lifeavg.tex": abl_tex,
        "datasets.tex": dataset_table(provenance),
        "settings.tex": settings_table(provenance),
        "hparam_grid.tex": hparam_table(provenance),
    }
    provenance["summary_lifeavg_gain_percent"] = load(SRC["summary"])
    for name, text in outputs.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("% Generated by _scripts/make_tables.py from frozen result files; do not edit by hand.\n")
            handle.write(text)
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
    print("wrote", ", ".join(sorted(outputs)), "and provenance.json")
    if args.check:
        bad = check_against_v6("main", main_vals, main_marks, main_rows)
        bad += check_against_v6("ablation", abl_vals, abl_marks, abl_rows)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
