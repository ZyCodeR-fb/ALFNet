"""Convert the Chinese v6 draft (Markdown) into the LaTeX section files.

Run from the repository root:

    python paper/latex_neurips_20261020/_scripts/md_to_tex.py [--draft MD] [--out DIR]

The Markdown draft is the editing source; sections/*.tex are regenerated from it and
then uploaded to Overleaf. Re-running overwrites sections/, so merge any edits made
directly in Overleaf back into the draft first.
Citations [n] are mapped to BibTeX keys, figures/tables to \ref labels, display
equations to numbered equation environments, and HTML/pipe tables to the generated
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
    1: "cini2022filling", 2: "liu2023pristi", 3: "zuo2023graph", 4: "yu2024ginar",
    5: "wu2021inductive", 6: "mei2023uncertainty", 7: "zhou2025mogernn", 8: "liu2025spatiotemporal",
    9: "zhao2026structure", 10: "yu2018spatio", 11: "wu2019graph", 12: "bai2020adaptive",
    13: "jiang2023pdformer", 14: "fang2025efficient", 15: "shao2022spatial", 16: "perez2018film",
    17: "liu2026general", 18: "li2018diffusion", 19: "cho2014learning", 20: "zhang2023crossformer",
    21: "shazeer2017outrageously", 22: "wang2023graph", 23: "guo2019attention", 24: "wang2026cold",
}
FIG = {
    1: ("fig:arch", "figures/alfnet.png", r"\textwidth"),
    2: ("fig:condition", "figures/fig2_condition_stage_test.pdf", r"\textwidth"),
    3: ("fig:trajectory", "figures/fig3_trajectory_val.pdf", r"\textwidth"),
    4: ("fig:layout", "figures/fig4_layout_val.pdf", r"\textwidth"),
    5: ("fig:hparam", "figures/fig5_hyperparameters_val.pdf", r"\textwidth"),
    6: ("fig:cases", "figures/fig6_cases_val.pdf", r"\textwidth"),
}
TAB = {1: ("tab:main", "tables/main_lifeavg"), 2: ("tab:ablation", "tables/ablation_lifeavg")}
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
        chunk = chunk.replace("%", r"\%").replace("\\*", "*").replace("→", r"$\to$")
        chunk = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", chunk)
        chunk = re.sub(r"\s*\[(\d+(?:\s*[,–]\s*\d+)*)\]", cite, chunk)
        chunk = re.sub(r"第([2-5])节", lambda m: "第\\ref{" + SECTION_LABEL[m.group(1)] + "}节", chunk)
        chunk = re.sub(r"(?<![\d.])([2-5])\.([1-8])(?![\d%\\])",
                       lambda m: "\\ref{sec:" + m.group(1) + "_" + m.group(2) + "}", chunk)
        chunk = re.sub(r"图([1-6])(?![\d])", lambda m: "图\\ref{" + FIG[int(m.group(1))][0] + "}", chunk)
        chunk = re.sub(r"表([12])(?![\d])", lambda m: "表\\ref{" + TAB[int(m.group(1))][0] + "}", chunk)
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
            n = int(re.match(r"\*\*图(\d)", cap).group(1))
            label, path, width = FIG[n]
            m = re.match(r"\*\*图\d\u3000(.+?)\*\*\s*(.*)", cap)
            caption = "\\textbf{" + inline(m.group(1)) + "}" + inline(m.group(2))
            out += ["\\begin{figure}[t]", "\\centering"]
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
            n = int(re.match(r"\*\*表(\d)", pending_caption).group(1))
            label, path = TAB[n]
            m = re.match(r"\*\*表\d\u3000(.+?)\*\*\s*(.*)", pending_caption)
            caption = "\\textbf{" + inline(m.group(1)) + "}" + inline(m.group(2))
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
        heading = re.match(r"^(#{2,3}) (?:(\d)(?:\.(\d))? )?(.+)$", line)
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
    text = text.split("## 参考文献")[0]
    head_comment = re.match(r"\s*# .*?\n\n(<!--.*?-->)", text, flags=re.S)
    chunks = re.split(r"^(## .+)$", text, flags=re.M)
    os.makedirs(OUT, exist_ok=True)
    pipe_counter = [0]
    written = []
    for idx in range(1, len(chunks), 2):
        heading = chunks[idx][3:].strip()
        body = chunks[idx + 1].strip("\n").split("\n")
        number = heading.split(" ")[0]
        name = FILES.get(number, FILES.get(heading))
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
    if pipe_counter[0] != len(PIPE_TABLES):
        sys.exit(f"expected {len(PIPE_TABLES)} pipe tables, converted {pipe_counter[0]}")
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
