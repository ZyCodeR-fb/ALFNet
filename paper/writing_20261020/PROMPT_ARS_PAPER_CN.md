# 新会话任务：用 ARS 完成 ALFNet 论文（补写方法 → LaTeX → 模拟评审 → 修改）

项目仓库：`A:/projects/ALFNet`。请全程用简体中文与我沟通；论文正文先用中文写，等和导师确定后再翻译为英文（顶会投稿语言）。

实验已经全部完成并冻结。初版中文论文为 `paper/manuscript_draft_cn-2-v5.md`，其中有引言、相关工作、实验与参考文献，**没有第 3 节方法，也没有摘要和结论**。本任务是用已安装的 ARS 插件（academic-research-skills，Claude Code 用户级插件）完成整篇论文：

1. 补写方法；
2. 生成可导入 Overleaf 的 LaTeX 工程，供我转发导师共同修改；
3. 模拟评审，按评审意见修改，直到各章互相对应。

## 一、先恢复上下文（不要跳过）

1. 按根目录 `CLAUDE.md` 的顺序阅读 `docs/ai/WORKFLOW.md`、`docs/ai/PROJECT_CONTEXT.md`、`docs/ai/HANDOFF.md` 及其链接的任务 handoff。然后读 `docs/FINAL_PROJECT_INDEX.md` 和 `docs/RESEARCH_LESSONS.md`。
2. 检查 Git 分支、HEAD 和工作区。不切换分支，不覆盖已有改动，不 commit 或 push。
3. 按 WORKFLOW 在 `docs/ai/` 下为本任务新建任务 handoff（例如 `T-20261020-paper-writing`），并在 `docs/ai/HANDOFF.md` 路由表登记。每个阶段结束更新进度。
4. 必读写作材料：

| 用途 | 文件 |
|---|---|
| 当前正文（v5，保留不动） | `paper/manuscript_draft_cn-2-v5.md` |
| 正文自审记录 | `paper/manuscript_v5_audit_20261012.md` |
| 实验稿与附录 | `paper/experiments_draft_cn.md`、`paper/experiments_appendix_cn.md` |
| 相关工作与检索证据 | `paper/related_work_cn.md`、`paper/related_work_evidence_cn.md`、`paper/related_work_search_20260914.json` |
| 对比模型选择与公平性 | `paper/baseline_selection_cn.md`、`paper/BASELINE_FAIRNESS_20260928.md`、`docs/comparison_model_audit.md` |
| 最终配置与主表权威 | `paper/pems08_wd_test_20261012/selected_configuration_manifest.json`、`selected_tables.json` 及同目录工作簿 |
| 正文图 2–6 与底层 CSV | `paper/final_figures_20261017/` |
| 方法写作依据（实现审计、符号、公式、代码映射） | `paper/architecture_20261019/STAGE1_IMPLEMENTATION_AUDIT_CN.md`、`paper/architecture_20261019/v2_20261019/README_CN.md`（名称、符号、公式、四数据集开关、代码行号）、`paper/architecture_20261019/nanobanana_v4_20261019/README_CN.md` 与 `DESIGN_CN.md`（英文图注、模块名、复核过的源码行号） |
| 架构图（暂用） | `paper/architecture_20261019/nanobanana_v4_20261019/alfnet_architecture_v4_gpt_P1_textfix.png`。我之后会用 PPT 复现一版替换；R1（`imagegen_v3/alfnet_imagegen_r1.png`）架构相同，仅样式不同 |
| 导师汇报（了解导师已知信息） | `paper/advisor_report_cn_v5.md`、`paper/advisor_paper_talk_cn.md` |

权威顺序：最终配置清单与实际代码 → 已核验的映射文档 → 论文草稿 → 历史设计。共享文档里有大量废弃候选和历史参数，不能把它们写成最终模型。

## 二、硬性边界

- 不训练、不评估、不访问测试集、不改模型代码、配置、权重或结果。所有数字只能来自 `paper/` 下已有的表格、JSON 或 CSV。新稿中每个数字都要能追溯到具体文件。发现草稿数字与权威文件不一致时，列出证据，不要自行"修正"实验。
- 模拟评审若指出需要新实验或新基线，只记录为"建议/待导师决定"。不要启动任何实验。
- 不把 GRIN/BiGRIL、DCRNN/GRU、Crossformer 两阶段注意力、PatchSTG 分块或 FiLM 写成原创算子。贡献表述沿用 v5 引言的口径：贡献在于条件化的组织方式。
- 方法中的每个公式、维度和开关，都要以 v2 README 与源码核对一致。例如：
  - 共享 GRIN 对所有节点执行，窗口插值仅覆盖 q>0 节点的缺失位置；
  - 每节点只有 1 个 temporal patch；
  - P=32，C=7/16/10/6；
  - 4 个 latent tokens；
  - FiLM 只调制两层编码器末状态，且只作用一次；
  - 五专家 dense softmax 路由；
  - 解码初值 y₀ = X̄ᴸ，反馈未修正的 yₖ；
  - 先 Restore（+αv），再 Role-Scaled Refinement；
  - 训练目标 0.9 L_mean + 0.1 L_role + L_route，其中逐元素损失为 (e/σ)·tanh(0.1e)。
- 引用必须真实。参考文献只用已在草稿或相关工作证据中出现、或经 ARS citation-check 查证的条目，不编造 DOI、页码或作者。
- `paper/manuscript_draft_cn-2-v5.md` 不改动。所有新稿件写在新目录（见第四节）。

## 三、ARS 使用约定

- ARS 默认 APA 7.0 引用、繁体中文双语摘要。本任务覆盖为：英文论文，机器学习会议的数字编号引用（natbib numbers / `plainnat` 或模板自带样式），不需要中文摘要。沟通用简体中文。
- 建议的模式顺序（可按需调整，先说明再做）：
  1. `academic-paper` 的 `plan` / `outline`（即 `/ars-plan`、`/ars-outline`）：方法章结构与证据映射；
  2. 写方法、摘要和结论；
  3. `format-convert`：生成 LaTeX；
  4. `/ars-citation-check`：检查引用；
  5. `academic-paper-reviewer` 的 full 模式（即 `/ars-reviewer`）：模拟评审；
  6. `revision`：按评审意见修改；
  7. 再评审：验证问题是否解决。

  文献调研已完成，不需要跑 deep-research 全流程。若评审指出缺少 2025–2026 年的重要相关工作，可以用 `/ars-lit-review` 或 `/ars-3w` 定向补查，补查结果先给我确认再写入。
- ARS 的跨模型验证（`ARS_CROSS_MODEL`）不开启。

## 四、产出物与目录

新建 `paper/latex_neurips_20261020/`，作为 LaTeX 源头，结构如下：

```
main.tex                 # 使用 NeurIPS 官方样式的 preprint 模式（显示作者，便于导师阅读）
sections/{abstract,introduction,related_work,method,experiments,conclusion}.tex
appendix/...             # 补充实验、协议细节、超参数、复现信息
figures/                 # 架构图占位 + 图 2–6（来自 final_figures_20261017，优先 PDF/矢量）
tables/                  # 由 selected_tables.json 等生成的表格 .tex
references.bib
README_overleaf_CN.md    # 导入 Overleaf 与协作说明
```

- **模板**：NeurIPS 官方 LaTeX 样式。它是 ML 方向最通用的顶会模板：单栏，正文页数限制加上不限页数的附录，适合本文较多的实验；之后改投 ICLR、KDD 或 AAAI 时迁移成本低。
  - 样式文件请先确认当年最新版本；需要下载时先征得我同意，并说明文件名、来源和大小。
  - 也可以让我在 Overleaf 模板库新建 NeurIPS 项目后，再上传你的文件。
  - 检查本机是否有可用的 TeX 环境（latexmk、pdflatex、xelatex 或 tectonic）。有就本地编译，并检查编译日志里是否有警告；没有就说明需要在 Overleaf 编译。
- 架构图暂用上面的 PNG，图注用 nanobanana_v4 README 中的英文图注，正文注释标明"待替换为 PPT 版"。
- 表格优先由脚本从权威 JSON 生成，避免手抄错误。生成脚本放在该目录的 `_scripts/` 下。
- 另写一份 `CHANGES_FOR_ADVISOR_CN.md`，给导师看的中文修改说明：新增了什么、与 v5 的差异、尚待讨论的问题。
- 打包 `paper/latex_neurips_20261020_overleaf.zip`，便于在 Overleaf 中用 New Project → Upload Project 导入。在 `README_overleaf_CN.md` 写清协作流程：
  1. 导入后用 Share 邀请导师；
  2. 用评论功能讨论（修订追踪等功能取决于 Overleaf 套餐，不要承诺）；
  3. 讨论定稿后下载 zip，放回仓库对照 diff；
  4. 仓库目录始终作为可追溯的源头。

## 五、核心要求：全文对应关系（贯穿始终）

在动笔前先建立 `paper/latex_neurips_20261020/ALIGNMENT_MATRIX_CN.md`。每个阶段结束时更新，模拟评审时也作为必查项：

| 列 | 内容 |
|---|---|
| 问题 Pi | 引言提出的每个研究问题/挑战（如：窗口内观测支持变化、零历史节点、加入/中断进展、空间传播是否随支持条件调整、跨节点时间上下文……）|
| 相关工作缺口 | 2.1/2.2/2.3 中哪一段指出现有方法对 Pi 的不足 |
| 方法模块 | 方法中哪个模块解决 Pi（共享历史补全、观测条件化传播/五专家路由、ALFTemporal + FiLM、Role-Scaled Refinement、训练目标中的 L_role 等）|
| 实验证据 | 哪个 RQ、表或图验证 Pi（4.2–4.8，以及消融、配对干预、布局分析）|
| 对比模型 | 该方向在相关工作中对应哪些方法、实验选了哪个、哪些未选及理由（与 4.1"比较方法"一致）|
| 状态 | 对应完整 / 缺失 / 不一致，并写明修改动作 |

检查规则：

- 引言提出的每个问题，方法中都有对应模块，实验中都有对应证据；没有证据的主张要删除或弱化。
- 方法中的每个模块，都能回答引言的某个问题，并有消融或干预实验支持；没有消融支撑的模块（如 Refinement 在正文消融中被移除）要如实说明其角色及附录证据。
- 相关工作的三个方向与对比模型的选取逐一对应。被讨论但未比较的方法（如 ST-FiT、SIML-MSIGN、GCN-M、PDFormer、AGCRN、Graph WaveNet），要有一致的未选理由。
- 引言中的贡献与结论表述，要与实验中的实际结果一致，包括负面边界（例如零历史条件下的优势并非在所有数据集上成立）。
- 术语前后统一：观测支持、生命周期角色 AAS/NAS/VS/FS、场景 Unseen/Steady/Join/Fault、LifeAvg、模块名（以 v2 README 名称表为准）。

## 六、阶段与检查点（每个检查点停下来等我确认）

1. **上下文与对应矩阵**：输出事实摘要、矩阵初版、发现的不一致和缺口，以及方法章大纲（含符号表和公式清单）。→ 检查点 1
2. **补写方法章和摘要、结论**：先以中文稿 v6 写在新目录，或直接写英文，写前说明选择。方法须与架构图逐模块对应，四数据集差异放在表格或附录。→ 检查点 2
3. **全文英文 LaTeX**：按 NeurIPS 模板生成，编译，检查引用；同时生成给导师的修改说明和 Overleaf 包。→ 检查点 3（我导入 Overleaf 检查）
4. **模拟评审**：用 academic-paper-reviewer 的 full 模式，把对应矩阵作为必查清单。评审意见按严重程度分类，分别标出"可文字修改"与"需新实验（只记录）"。→ 检查点 4
5. **修改与再评审**：执行文字层面修改，用 re-review 验证。重复步骤 4–5，直到没有高严重度的文字问题。最后更新矩阵、修改说明和 Overleaf 包。

每一步都区分"已验证 / 已修改未验证 / 未开始"。不要用"看起来更好"代替逐项核对。
