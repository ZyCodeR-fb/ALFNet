# ALFNet 论文 LaTeX 工程：Overleaf 更新与协作说明

本文件与 `CHANGES_FOR_ADVISOR_CN.md` 只供作者本人查看，**不上传 Overleaf**，不与导师共享。

当前版本是**中文工作稿**，使用 NeurIPS 2026 官方样式的 preprint 模式（显示作者）。与导师定稿后再整篇译为英文。

## 1. 更新已有的 Overleaf 项目

Overleaf 项目已经建立并分享给导师，之后不再打包 zip，而是在现有项目中上传文件覆盖旧版：

1. 在 Overleaf 左侧文件树中进入对应文件夹，点 **Upload**，选择本地文件；同名文件会被覆盖。
2. 每次修改后需要上传的文件见第 4 节；没有变化的文件不必上传。
3. **覆盖前先看一眼导师是否在 Overleaf 里直接改过正文。** 上传会覆盖同名文件中的全部内容；如果导师改过，先在 Overleaf 的 History 中查看改动，把需要保留的内容合并回本地 md 稿，再重新生成并上传。
4. 编译器保持 **XeLaTeX**，Main document 为 `main.tex`。

本机没有 TeX 环境，工程只做了静态检查（括号与环境配对、`\ref`/`\label` 对应、引用全部能在 `references.bib` 中找到、`\input` 与图片文件存在）。上传后请在 Overleaf 重新编译，留意日志中的警告（尤其是新公式较长时的 overfull 行），把问题告诉我修改。

## 2. 文件结构

| 路径 | 内容 | 是否上传 Overleaf |
|---|---|---|
| `main.tex` | 导言区（NeurIPS 样式、ctex 中文、natbib 数字编号）、标题、作者占位、各节 `\input` | 是 |
| `neurips_2026.sty` | NeurIPS 2026 官方样式，未修改 | 是 |
| `sections/*.tex` | 摘要、引言、相关工作、方法、实验、结论；**由 md 稿生成，不要手改** | 是 |
| `tables/*.tex` | 由脚本从冻结结果文件生成的表格主体 | 是 |
| `figures/` | 图 1（作者 PPT 版 `alfnet.png`）与图 2–6（`final_figures_20261017` 的 PDF 矢量图） | 是 |
| `references.bib` | 24 条文献，元数据已逐条核对 | 是 |
| `drafts/manuscript_v6_cn.md` | **编辑源头**：中文 v6 Markdown 稿 | 否 |
| `tables/provenance.json` | 每张表的数据来源路径与 SHA256 | 否 |
| `_scripts/` | `md_to_tex.py`（md→各节 tex）、`make_tables.py`（结果文件→表格，并与 md 逐格比对）、`make_overleaf_zip.py`（初次打包用，现已不需要） | 否 |
| `notes/`、`review_round1/`、`ALIGNMENT_MATRIX_CN.md`、本文件、`CHANGES_FOR_ADVISOR_CN.md` | 对应矩阵、修改记录、引用核对、架构图核对清单、模拟评审 | 否 |
| `_template_neurips2026/` | 官方模板原件，含正式投稿必需的 `checklist.tex` | 否 |

## 3. 与导师协作

1. 讨论用 Overleaf 的**评论**功能（选中文字后添加评论）。
2. 导师若直接在 Overleaf 改正文，请把改动告诉我或下载 Source，我把它合并回 md 稿，再重新生成 tex；否则下次上传会覆盖这些改动。
3. **仓库中的 md 稿与结果文件始终是可追溯的源头。** 表格数值只能由 `_scripts/make_tables.py` 从冻结结果重新生成；不要在 Overleaf 里直接改表中数字。

## 4. 重新生成与上传（只在仓库中执行）

```bash
python paper/latex_neurips_20261020/_scripts/make_tables.py --check
```

```bash
python paper/latex_neurips_20261020/_scripts/md_to_tex.py
```

第一条只读取已保存的结果文件，重新写出 `tables/` 并与 md 稿中的表 1、表 2 逐格比对；第二条由 md 稿重新生成 `sections/*.tex`。

2026-10-27 第五轮需要上传覆盖的文件：`sections/abstract.tex`、`sections/method.tex`、`sections/experiments.tex`。`main.tex`、`references.bib`、`figures/` 与其余表格没有变化。

修改历史由 git 记录（分支 `paper/final`），不再另存快照。注意仓库 `.gitignore` 忽略所有名为 `figures/` 的目录，本工程的 `figures/` 是用 `git add -f` 加入的；以后替换图片后同样需要 `git add -f`。

## 5. 尚未包含的内容

- 作者、单位、邮箱（`main.tex` 中的占位）；
- 致谢与资助声明（`main.tex` 中已留注释的 `ack` 环境）；
- NeurIPS Paper Checklist：正式投稿必需，原件在 `_template_neurips2026/checklist.tex`，导师阅读版暂未加入；
- 附录：按作者决定暂不写；
- 英文版：定稿后整篇翻译，届时去掉 ctex 设置，改回 pdfLaTeX。
