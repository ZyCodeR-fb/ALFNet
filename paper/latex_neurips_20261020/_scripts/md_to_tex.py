"""Convert the Chinese v6 draft (Markdown) into the LaTeX section files.

Run from the repository root:

    python paper/latex_neurips_20261020/_scripts/md_to_tex.py [--draft MD] [--out DIR]

The Markdown draft is the editing source; sections/*.tex are regenerated from it and
then uploaded to Overleaf. Re-running overwrites sections/, so merge any edits made
directly in Overleaf back into the draft first.
Citations [n] are mapped to BibTeX keys, figures/tables to \ref labels, display
equations to numbered equation environments, *text* to \\emph, and HTML/pipe tables to the generated
tabular files under tables/. HTML comments (author notes) become LaTeX comments.
"""
import os
import re
import sys

ROOT = os.getcwd()
BASE = os.path.join(ROOT, "paper", "latex_neurips_20261020")
DRAFT = os.path.join(BASE, "drafts", "manuscript_v6_cn.md")
OUT = os.path.join(BASE, "sections")
for flag in ("--draft", "--out"):
    if flag in sys.argv:
        value = os.path.abspath(sys.argv[sys.argv.index(flag) + 1])
        if flag == "--draft":
            DRAFT = value
        else:
            OUT = value

CITE = {
    1: "yin2025xxltraffic", 2: "zhou2025mogernn", 3: "roth2022forecasting", 4: "mei2023uncertainty", 5: "gupta2023frigate", 6: "lei2025stfit", 7: "yu2024ginar", 8: "yu2025merlin", 9: "marisca2024graph", 10: "che2018recurrent", 11: "chen2021trafficstream", 12: "wang2023pattern", 13: "liu2026general", 14: "wang2026cold", 15: "zhao2026structure", 16: "wang2024stone", 17: "ma2025robust", 18: "cini2022filling", 19: "fang2025efficient", 20: "zhang2023crossformer", 21: "perez2018film", 22: "wang2023graph", 23: "li2018diffusion", 24: "wu2021inductive", 25: "zheng2023increase", 26: "xu2025kits", 27: "su2024spatial", 28: "su2026generalising", 29: "prabowo2024traffic", 30: "liu2025spatiotemporal", 31: "liu2023pristi", 32: "cui2020graph", 33: "wang2023traffic", 34: "zuo2023graph", 35: "huang2026masmamba", 36: "yu2025ginarplus", 37: "chen2025expand", 38: "yu2018spatio", 39: "wu2019graph", 40: "bai2020adaptive", 41: "cini2023taming", 42: "shao2022spatial", 43: "jiang2023pdformer", 44: "lee2024testam", 45: "yun2024flexmoe", 46: "guo2019attention", 47: "fofanah2026pimcst", 48: "zou2026meta",
}
FIG = {
    1: ("fig:arch", "figures/ALFNet.pdf", r"\textwidth"),
    2: ("fig:condition", "figures/fig2_phase_gain_3seed.pdf", r"\textwidth"),
    3: ("fig:trajectory", "figures/fig3_response_4ds_val.pdf", r"\textwidth"),
    4: ("fig:layout", "figures/fig4_layout_4ds_val.pdf", r"\textwidth"),
    5: ("fig:hparam", "figures/fig5_hparams_val.pdf", r"\textwidth"),
    6: ("fig:ratio", "figures/fig6_ratio_test.pdf", r"\textwidth"),
    7: ("fig:cases", "figures/fig7_cases_4ds_val.pdf", r"\textwidth"),
    "A1": ("fig:app-traj-diff", "figures/figA_trajectory_diff_val.pdf", r"\linewidth"),
    "A2": ("fig:app-router", "figures/figA1_router_gates.pdf", r"\linewidth"),
}
TAB = {1: ("tab:main_speed", "tables/main_speed"), 2: ("tab:main_flow", "tables/main_flow"),
       3: ("tab:ablation", "tables/ablation_lifeavg")}
# appendix: the text is in the draft (sections "附录A", "附录B"); the tabular files come from make_appendix.py
APP_TAB = {1: "tab:app-lifeavg", 2: "tab:app-phase-metrla", 3: "tab:app-phase-pemsbay", 4: "tab:app-phase-pems04",
           5: "tab:app-phase-pems08", 6: "tab:app-abl-scenario", 7: "tab:app-abl-phase", 8: "tab:app-ratio"}
APP_TAB_FILE = {1: "app_lifeavg", 2: "app_phase_metrla", 3: "app_phase_pemsbay", 4: "app_phase_pems04",
                5: "app_phase_pems08", 6: "app_ablation_scenario", 7: "app_ablation_phase", 8: "app_ratio"}
APP_TAB_NO_RESIZE = set()
APP_FIG = {n: FIG[f"A{n}"][0] for n in (1, 2)}
PIPE_TABLES = ["tables/datasets", "tables/settings", "tables/hparam_grid"]
FILES = {"摘要": "abstract", "1": "introduction", "2": "related_work", "3": "method",
         "4": "experiments", "5": "conclusion"}
SECTION_LABEL = {"1": "sec:intro", "2": "sec:related", "3": "sec:method", "4": "sec:exp", "5": "sec:conclusion"}


def cite(match):
    keys = []
    for part in re.split(r"\s*,\s*", match.group(1)):
        if "–" in part:
            a, b = map(int, part.split("–"))
            keys += [CITE[i] for i in range(a, b + 1)]
        else:
            keys.append(CITE[int(part)])
    return r"~\citep{" + ",".join(keys) + "}"


def protect_math(text):
    """Split text into (is_math, chunk) pieces on $...$ (display math handled earlier)."""
    out, pos = [], 0
    for m in re.finditer(r"\$[^$]+\$", text):
        out.append((False, text[pos:m.start()]))
        out.append((True, m.group(0)))
        pos = m.end()
    out.append((False, text[pos:]))
    return out


def inline(text):
    pieces = []
    for is_math, chunk in protect_math(text):
        if is_math:
            pieces.append(chunk)
            continue
        chunk = chunk.replace("%", r"\%").replace("→", r"$\to$")
        chunk = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", chunk)
        chunk = re.sub(r"(?<![\\*])\*([^*]+?)(?<!\\)\*(?!\*)", r"\\emph{\1}", chunk)
        chunk = chunk.replace("\\*", "*")
        chunk = re.sub(r"\s*\[(\d+(?:\s*[,–]\s*\d+)*)\]", cite, chunk)
        chunk = re.sub(r"第([2-5])节", lambda m: "第\\ref{" + SECTION_LABEL[m.group(1)] + "}节", chunk)
        chunk = re.sub(r"(?<![\d.])([2-5])\.([1-8])(?![\d%\\])",
                       lambda m: "\\ref{sec:" + m.group(1) + "_" + m.group(2) + "}", chunk)
        chunk = re.sub(r"表A([1-9])–A([1-9])", lambda m: "表\\ref{" + APP_TAB[int(m.group(1))] + "}--\\ref{"
                       + APP_TAB[int(m.group(2))] + "}", chunk)
        chunk = re.sub(r"表A([1-9])(?![\d])", lambda m: "表\\ref{" + APP_TAB[int(m.group(1))] + "}", chunk)
        chunk = re.sub(r"图A([1-9])(?![\d])", lambda m: "图\\ref{" + APP_FIG[int(m.group(1))] + "}", chunk)
        chunk = re.sub(r"图([1-7])(?![\d])", lambda m: "图\\ref{" + FIG[int(m.group(1))][0] + "}", chunk)
        chunk = re.sub(r"表([1-3])(?![\d])", lambda m: "表\\ref{" + TAB[int(m.group(1))][0] + "}", chunk)
        chunk = re.sub(r"式\((\d+)\)", lambda m: "式~\\eqref{eq:" + m.group(1) + "}", chunk)
        pieces.append(chunk)
    return "".join(pieces)


def convert_block(lines, pipe_counter):
    out, i = [], 0
    pending_caption = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("<!--"):
            comment = line
            while "-->" not in comment and i + 1 < len(lines):
                i += 1
                comment += " " + lines[i]
            out.append("% " + comment.replace("<!--", "").replace("-->", "").strip())
            i += 1
            continue
        if line.startswith("$$"):
            body = []
            i += 1
            while not lines[i].startswith("$$"):
                body.append(lines[i])
                i += 1
            i += 1
            text = "\n".join(body)
            tag = re.search(r"\\tag\{(\d+)\}", text)
            text = re.sub(r"\s*\\tag\{\d+\}", "", text).strip()
            label = f"\\label{{eq:{tag.group(1)}}}" if tag else ""
            out += ["\\begin{equation}" + label, text, "\\end{equation}"]
            continue
        if line.startswith("!["):
            # figure; caption is the next non-empty paragraph starting with **图N
            j = i + 1
            notes = []
            while j < len(lines) and not lines[j].startswith("**图"):
                if lines[j].startswith("<!--"):
                    notes.append(lines[j].replace("<!--", "").replace("-->", "").strip())
                j += 1
            cap = lines[j]
            n = re.match(r"\*\*图(A?\d)", cap).group(1)
            label, path, width = FIG[n if n.startswith("A") else int(n)]
            m = re.match(r"\*\*图A?\d\u3000(.+?)\*\*\s*(.*)", cap)
            caption = "\\textbf{" + inline(m.group(1)) + "}" + inline(m.group(2))
            out += ["\\begin{figure}[tp]" if n.startswith("A") else "\\begin{figure}[t]", "\\centering"]
            out += ["% " + note for note in notes]
            out += [f"\\includegraphics[width={width}]{{{path}}}",
                    f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{figure}"]
            i = j + 1
            continue
        if line.startswith("**表"):
            pending_caption = line
            i += 1
            continue
        if line.startswith("<table>"):
            while not lines[i].startswith("</table>"):
                i += 1
            i += 1
            n = re.match(r"\*\*表(A?\d)", pending_caption).group(1)
            m = re.match(r"\*\*表A?\d\u3000(.+?)\*\*\s*(.*)", pending_caption)
            caption = "\\textbf{" + inline(m.group(1)) + "}" + inline(m.group(2))
            if n.startswith("A"):
                k = int(n[1:])
                body = f"\\input{{tables/{APP_TAB_FILE[k]}}}"
                out += ["\\begin{table}[p]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{APP_TAB[k]}}}",
                        "\\small", body if k in APP_TAB_NO_RESIZE else f"\\resizebox{{\\linewidth}}{{!}}{{{body}}}",
                        "\\end{table}"]
                pending_caption = None
                continue
            label, path = TAB[int(n)]
            out += ["\\begin{table}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                    "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
                    f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{{path}}}}}", "\\end{table}"]
            pending_caption = None
            continue
        if line.startswith("|"):
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            path = PIPE_TABLES[pipe_counter[0]]
            pipe_counter[0] += 1
            out += ["\\begin{center}", "\\small", f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{{path}}}}}",
                    "\\end{center}"]
            continue
        heading = re.match(r"^(#{2,3}) (?:([\dAB])(?:\.(\d))? )?(.+)$", line)
        if heading:
            level, major, minor, title = heading.groups()
            if level == "###":
                out.append(f"\\subsection{{{inline(title)}}}\\label{{sec:{major}_{minor}}}")
            i += 1
            continue
        out.append(inline(line))
        i += 1
    return out


def main():
    text = open(DRAFT, encoding="utf-8").read()
    text, _, tail = text.partition("## 参考文献")  # the appendix follows the reference list
    if "\n## 附录" in tail:
        text += tail[tail.index("\n## 附录") + 1:]
    head_comment = re.match(r"\s*# .*?\n\n(<!--.*?-->)", text, flags=re.S)
    chunks = re.split(r"^(## .+)$", text, flags=re.M)
    os.makedirs(OUT, exist_ok=True)
    pipe_counter = [0]
    written, appendix = [], []
    for idx in range(1, len(chunks), 2):
        heading = chunks[idx][3:].strip()
        body = chunks[idx + 1].strip("\n").split("\n")
        number = heading.split(" ")[0]
        name = FILES.get(number, FILES.get(heading))
        if heading.startswith("附录"):  # "附录A 标题": both appendix sections go to one file
            letter, title = heading[2], heading.split(" ", 1)[1]
            appendix += [f"\\section{{{title}}}\\label{{app:{letter}}}"] + convert_block(body, pipe_counter) + [""]
            continue
        if name is None:
            continue
        lines = []
        if number in SECTION_LABEL:
            title = heading.split(" ", 1)[1]
            lines.append(f"\\section{{{title}}}\\label{{{SECTION_LABEL[number]}}}")
        lines += convert_block(body, pipe_counter)
        content = "\n".join(lines).strip() + "\n"
        content = re.sub(r"\n{3,}", "\n\n", content)
        # A display equation belongs to the sentence before it; text that explains
        # the equation ("其中…") continues the same paragraph.
        content = re.sub(r"\n\n(\\begin\{equation\})", r"\n\1", content)
        content = re.sub(r"(\\end\{equation\})\n\n(?=其中|softmax|该损失|这一步|\$\\mathcal\{L\})",
                         r"\1\n", content)
        with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        written.append(name)
    if appendix:
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(appendix).strip() + "\n")
        with open(os.path.join(OUT, "appendix.tex"), "w", encoding="utf-8", newline="\n") as handle:
            handle.write("% Generated by _scripts/md_to_tex.py from the draft; tables by _scripts/make_appendix.py.\n" + content)
        written.append("appendix")
    if pipe_counter[0] != len(PIPE_TABLES):
        sys.exit(f"expected {len(PIPE_TABLES)} pipe tables, converted {pipe_counter[0]}")
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
