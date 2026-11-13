"""Generate the appendix tables (tables/app_*.tex) from frozen result files and keep the draft's copies in sync.

Run from the repository root (after make_tables.py):

    python paper/latex_neurips_20261020/_scripts/make_appendix.py [--check | --write-md]

The appendix text lives in drafts/manuscript_v6_cn.md (sections "附录A", "附录B") and is converted by
md_to_tex.py; this script only writes the tabular files and, with --write-md, the HTML copy of every
appendix table under its "**表An" caption in the draft (--check reports differences instead).

Nothing is trained or evaluated. Two statistics are kept apart:
  * three-seed tables (LifeAvg, phases): test set, mean ± sample std (ddof=1) over seeds 32/42/52,
    11 comparison methods + ALFNet, 2 decimals as in Tables 1-2; every mean/std is checked against
    results_seeds2_20261110/main_phase_3seed_20261112.xlsx (which stores "mean ± std" text with 3 decimals,
    not per-seed values);
  * ablation tables: test set, seed 32 only, Full + five ablations, 3 decimals as in Table 3; every value is
    checked against results_seeds2_20261110/ablation_seed32_20261112.xlsx.
Ranking marks follow make_tables.py (red bold = best, blue underline = second, on unrounded values within a
dataset/column of the displayed rows). The router and efficiency numbers come from
paper/experiments_20261112/ (scripts/router_weights_20261112.py, scripts/efficiency_20261112.py).
The unseen-ratio table is the METR-LA seed-32 test result of scripts/test_ratio_20261113.py (pool 50 = protocol).
"""
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402

ROOT = mt.ROOT
PAPER = os.path.join(ROOT, "paper", "latex_neurips_20261020")
XLSX = os.path.join(PAPER, "results_seeds2_20261110")
EXP = os.path.join(ROOT, "paper", "experiments_20261112")
JOIN, FAULT = mt.PHASES[:3], mt.PHASES[3:]
ROWS = mt.BASELINES + ["ALFNet"]
ABLATIONS = [("ALFNet", "four", "ALFNet")] + [r for _, members in mt.ABLATION_GROUPS for r in members]
METRIC_HEAD = ["MAE", "RMSE", "MAPE(\\%)"]
SLUG = {"METRLA": "metrla", "PEMSBAY": "pemsbay", "PEMS04": "pems04", "PEMS08": "pems08"}


def pm(mean, std, mark):
    return mt.cell(mean, mark, 2) + f"{{\\tiny$\\pm${std:.2f}}}"


def header(cells, first="方法"):
    """Two header rows: cell names spanning three metric columns, then the metric names."""
    top = " & ".join(f"\\multicolumn{{3}}{{c}}{{{c}}}" for c in cells)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(cells)))
    return [f"{first} & {top} \\\\", rules, " & " + " & ".join(METRIC_HEAD * len(cells)) + " \\\\"]


def method_rows(fmt):
    out = []
    for _, members in mt.GROUPS + [("", [("ALFNet", "")])]:
        out.append("\\midrule")
        out += [f"{m} & " + fmt(m) + " \\\\" for m, _ in members]
    return out


def lifeavg_table(mean, std):
    """LifeAvg, three seeds: two stacked panels (speed, flow), three metrics per dataset."""
    tex = ["\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule"]
    for i, (_, pair) in enumerate(mt.MAIN_TABLES):
        mk = {ds: {x: mt.marks({r: mean[ds][r]["LifeAvg"][x] for r in ROWS}) for x in mt.METRICS} for ds in pair}
        if i:
            tex.append("\\midrule")
        tex += header([f"{mt.DISPLAY[ds]}（{mt.UNIT[ds]}）" for ds in pair])
        tex += method_rows(lambda m: " & ".join(
            pm(mean[ds][m]["LifeAvg"][x], std[ds][m]["LifeAvg"][x], mk[ds][x][m]) for ds in pair for x in mt.METRICS))
    return "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def phase_table(ds, mean, std):
    """One dataset, three seeds: Join phases (top panel) and Fault phases (bottom panel)."""
    mk = {c: {x: mt.marks({r: mean[ds][r][c][x] for r in ROWS}) for x in mt.METRICS} for c in mt.PHASES}
    tex = ["\\begin{tabular}{l" + "c" * 9 + "}", "\\toprule"]
    for i, (track, cells) in enumerate((("Join", JOIN), ("Fault", FAULT))):
        if i:
            tex.append("\\midrule")
        tex += header([f"{track}: {c}" for c in cells])
        tex += method_rows(lambda m: " & ".join(
            pm(mean[ds][m][c][x], std[ds][m][c][x], mk[c][x][m]) for c in cells for x in mt.METRICS))
    return "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def ablation_values():
    from scripts import report_extension_20261103 as extension
    four = mt.load(mt.SRC["ablation"])["tables"]["test"]
    ext = extension.load()[3]["test"]
    return {ds: {name: (four if src == "four" else ext)[ds][key] for name, src, key in ABLATIONS}
            for ds in mt.DATASETS}


def ablation_table(values, group, cells):
    """Seed 32: Full + five ablations, one panel per dataset; columns = cells x metrics."""
    n = 3 * len(cells)
    tex = ["\\begin{tabular}{l" + "r" * n + "}", "\\toprule"] + header(cells, first="变体")
    for ds in mt.DATASETS:
        mk = {c: {x: mt.marks({r: values[ds][r][group][c][x] for r, _, _ in ABLATIONS}) for x in mt.METRICS}
              for c in cells}
        tex += ["\\midrule", f"\\multicolumn{{{n + 1}}}{{c}}{{{mt.DISPLAY[ds]}}} \\\\", "\\midrule"]
        for r, _, _ in ABLATIONS:
            tex.append(f"{'ALFNet (Full)' if r == 'ALFNet' else r} & " + " & ".join(
                mt.cell(values[ds][r][group][c][x], mk[c][x][r]) for c in cells for x in mt.METRICS) + " \\\\")
    return "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def check_workbooks(mean, std, values):
    """Every displayed number must agree with the advisor workbooks (returns the number of compared cells)."""
    from openpyxl import load_workbook
    n = 0
    wb = load_workbook(os.path.join(XLSX, "main_phase_3seed_20261112.xlsx"))
    for ds in mt.DATASETS:
        for group, cells in (("main", mt.SCENARIOS), ("phase", mt.PHASES)):
            rows = list(wb[f"{ds}_{group}"].iter_rows(values_only=True))
            col = {h: i for i, h in enumerate(rows[0])}
            body = {r[0]: r for r in rows[1:]}
            for m in ROWS:
                for c in cells:
                    for x in mt.METRICS:
                        want = f"{mean[ds][m][c][x]:.3f} ± {std[ds][m][c][x]:.3f}"
                        assert body[m][col[f"{c}/{x}"]] == want, (ds, m, c, x, body[m][col[f"{c}/{x}"]], want)
                        n += 1
    wb = load_workbook(os.path.join(XLSX, "ablation_seed32_20261112.xlsx"))
    for ds in mt.DATASETS:
        for group, cells in (("main", mt.SCENARIOS), ("phase", mt.PHASES)):
            rows = list(wb[f"{ds}_{group}"].iter_rows(values_only=True))
            col = {h: i for i, h in enumerate(rows[0])}
            body = {r[0]: r for r in rows[1:]}
            for r, _, _ in ABLATIONS:
                for c in cells:
                    for x in mt.METRICS:
                        assert math.isclose(body[r][col[f"{c}/{x}"]], values[ds][r][group][c][x], rel_tol=1e-12), (ds, r, c, x)
                        n += 1
    return n


def router_summary():
    """Mean gate per (dataset, role, expert) over seeds (NEW/OFF also over the 12 progress values) + progress effect."""
    rows = list(csv.DictReader(open(os.path.join(EXP, "router", "router_gates.csv"), encoding="utf-8")))
    acc, by_p = {}, {}
    for r in rows:
        acc.setdefault((r["dataset"], r["role"], r["expert"]), []).append(float(r["gate"]))
        by_p[(r["dataset"], r["seed"], r["role"], r["expert"], round(float(r["progress"]), 6))] = float(r["gate"])
    gate = {k: statistics.fmean(v) for k, v in acc.items()}
    progress = max(abs(by_p[(d, s, role, e, 1.0)] - by_p[(d, s, role, e, round(1 / 12, 6))])
                   for (d, s, role, e, p) in by_p if role in ("NEW", "OFF") and p == 1.0)
    role = max(abs(gate[(d, r, e)] - gate[(d, "ON", e)]) for (d, r, e) in gate)
    return gate, progress, role


def efficiency_table():
    data = mt.load("paper/experiments_20261112/efficiency/METRLA.json")
    tex = ["\\begin{tabular}{lrrr}", "\\toprule",
           "方法 & 参数量（M） & 单批推理时延（ms，中位数 [Q1, Q3]） & 峰值显存（MiB） \\\\", "\\midrule"]
    order = ["MoGERNN", "Merlin", "STBP", "STGCN", "ALFNet"]
    rows = {r["model"]: r for r in data["rows"]}
    for m in order:
        r = rows[m]
        if m == "ALFNet":
            tex.append("\\midrule")
        tex.append(f"{m} & {r['parameters'] / 1e6:.2f} & {r['latency_ms_median']:.1f} "
                   f"[{r['latency_ms_q1']:.1f}, {r['latency_ms_q3']:.1f}] & {r['peak_memory_mib']:.0f} \\\\")
    return "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n", rows


def ratio_table():
    """METR-LA, seed 32, test: four unseen-node pools x (ALFNet + four representative methods)."""
    files = glob.glob(os.path.join(ROOT, "artifacts", "evaluation", "ratio_test_20261113", "*", "test_results.json"))
    assert len(files) == 1, files
    data = json.load(open(files[0], encoding="utf-8"))
    tables, models = data["tables"], ["MoGERNN", "Merlin", "STBP", "STGCN", "ALFNet"]
    cells = mt.SCENARIOS
    n = 3 * len(cells)
    tex = ["\\begin{tabular}{l" + "r" * n + "}", "\\toprule"] + header(cells)
    for pool in sorted(tables, key=int):
        t = tables[pool]
        mk = {c: {x: mt.marks({m: t[m]["main"][c][x] for m in models}) for x in mt.METRICS} for c in cells}
        tex += ["\\midrule", f"\\multicolumn{{{n + 1}}}{{c}}{{未观测节点 {pool} 个（{int(pool) / 207 * 100:.1f}\\%）}} \\\\",
                "\\midrule"]
        for m in models:
            tex.append(f"{m} & " + " & ".join(mt.cell(t[m]["main"][c][x], mk[c][x][m]) for c in cells for x in mt.METRICS)
                       + " \\\\")
    return "\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n", {
        "file": os.path.relpath(files[0], ROOT), "plan_sha256": data["plan_sha256"], "seed": data["seed"], "split": data["split"]}


def tex_to_html(tex):
    """HTML copy of a generated tabular (for the Markdown draft): same cells, marks and order."""
    def conv(c):
        c = c.strip()
        c = re.sub(r"\\textcolor\{red\}\{\\textbf\{([^{}]*)\}\}", r'<b style="color:#c00000">\1</b>', c)
        c = re.sub(r"\\textcolor\{blue\}\{\\underline\{([^{}]*)\}\}", r'<u style="color:#1f4e9c">\1</u>', c)
        c = re.sub(r"\{\\tiny\$\\pm\$([^{}]*)\}", r"<sub>±\1</sub>", c)
        return c.replace("\\%", "%")
    html, head, tag = ["<table>", "  <thead>"], True, "th"
    for line in tex.split("\n"):
        line = line.strip()
        if line.startswith("\\midrule") and head:
            html += ["  </thead>", "  <tbody>"]
            head, tag = False, "td"
        if not line.endswith("\\\\"):
            continue
        row = []
        for c in line[:-2].split(" & "):
            m = re.fullmatch(r"\s*\\multicolumn\{(\d+)\}\{c\}\{(.*)\}\s*", c)
            row.append(f'<{tag} colspan="{m.group(1)}" align="center">{conv(m.group(2))}</{tag}>' if m
                       else f"<{tag}>{conv(c)}</{tag}>")
        html.append("    <tr>" + "".join(row) + "</tr>")
    assert not head
    out = "\n".join(html + ["  </tbody>", "</table>"])
    assert "\\" not in out, out[:400]
    return out


# appendix table number -> tabular file (order of appearance in the draft)
NUMBERS = ["app_lifeavg", "app_phase_metrla", "app_phase_pemsbay", "app_phase_pems04", "app_phase_pems08",
           "app_ablation_scenario", "app_ablation_phase", "app_ratio"]  # app_efficiency.tex is still written but no longer in the paper


def sync_draft(html, write):
    """Compare (or with write=True replace) the HTML table after each '**表An' caption of the draft."""
    text = open(mt.path(mt.DRAFT), encoding="utf-8").read()
    problems = 0
    for number in sorted(html, reverse=True):
        cap = re.search(rf"^\*\*表A{number}\u3000", text, flags=re.M)
        if cap is None or "<table>" not in text[cap.end():]:
            print(f"[表A{number}] caption or <table> not found in the draft")
            problems += 1
            continue
        start = text.index("<table>", cap.end())
        end = text.index("</table>", start) + len("</table>")
        if text[start:end] == html[number]:
            print(f"[表A{number}] draft table equals the generated table")
        elif write:
            text = text[:start] + html[number] + text[end:]
            print(f"[表A{number}] draft table replaced")
        else:
            print(f"[表A{number}] draft differs from the generated table")
            problems += 1
    if write:
        with open(mt.path(mt.DRAFT), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return problems


def main():
    plans, mean, std = mt.three_seed()
    values = ablation_values()
    compared = check_workbooks(mean, std, values)
    gate, progress, role = router_summary()
    eff_tex, eff = efficiency_table()
    ratio_tex, ratio = ratio_table()
    outputs = {"app_lifeavg.tex": lifeavg_table(mean, std),
               "app_ablation_scenario.tex": ablation_table(values, "main", mt.SCENARIOS),
               "app_ablation_phase.tex": ablation_table(values, "phase", mt.PHASES),
               "app_efficiency.tex": eff_tex, "app_ratio.tex": ratio_tex}
    for ds in mt.DATASETS:
        outputs[f"app_phase_{SLUG[ds]}.tex"] = phase_table(ds, mean, std)
    for name, text in outputs.items():
        with open(os.path.join(mt.OUT, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("% Generated by _scripts/make_appendix.py from frozen result files; do not edit by hand.\n")
            handle.write(text)
    with open(os.path.join(mt.OUT, "appendix_provenance.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"generated_by": "paper/latex_neurips_20261020/_scripts/make_appendix.py",
                   "three_seed": {"seeds": list(mt.SEEDS), "statistics": "mean and sample std (ddof=1), test set",
                                  "test_plans_sha256": plans, "methods": ROWS,
                                  "workbook": "results_seeds2_20261110/main_phase_3seed_20261112.xlsx (mean ± std text, no per-seed values)"},
                   "ablation": {"seed": 32, "variants": [r for r, _, _ in ABLATIONS],
                                "workbook": "results_seeds2_20261110/ablation_seed32_20261112.xlsx"},
                   "workbook_cells_compared": compared,
                   "router": {"max_gate_change_over_progress": progress, "max_role_difference_vs_ON": role},
                   "efficiency": eff, "unseen_ratio": ratio}, handle, ensure_ascii=False, indent=2)
    print("wrote", ", ".join(sorted(outputs)), "; workbook cells compared:", compared)
    print("router: role diff", round(role, 4), "progress effect", round(progress, 4))
    if "--check" in sys.argv or "--write-md" in sys.argv:
        html = {i: tex_to_html(outputs[name + ".tex"]) for i, name in enumerate(NUMBERS, start=1)}
        sys.exit(1 if sync_draft(html, "--write-md" in sys.argv) else 0)


if __name__ == "__main__":
    main()
