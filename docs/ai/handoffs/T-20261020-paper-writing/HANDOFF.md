# T-20261020-paper-writing

任务说明：[PROMPT_ARS_PAPER_CN.md](../../../../paper/writing_20261020/PROMPT_ARS_PAPER_CN.md)。用 ARS（academic-research-skills 3.23.1）补写方法、摘要、结论 → NeurIPS LaTeX → 模拟评审 → 修改。实验已冻结；不训练、不评估、不访问测试集、不改模型/配置/权重/结果；不 commit/push。

## 作者第二批意见修改 — 2026-10-25（已获授权提交 git，不推送）

- **用户指令**：引言不写模块来自他人工作，必要时只写“借鉴其思想”；方法章删 3.6、删文献编号；实验 4.1 删未比较方法理由与基线细节，训练与实现精简，不写测试结果参与配置决策；表 2 图注删 PEMS08 权重衰减说明；结论删“贡献不在于基础算子”、局限只留一两点；同类问题一并修改；不存快照，适当时提交 git。
- **Completed**：脚本 30 处替换 + 5 段删除（逐条见 `paper/latex_neurips_20261020/notes/V6_CHANGELOG_CN.md` 第四轮）；重新生成 sections/*.tex；README、CHANGES_FOR_ADVISOR 更新。
- **Verified**：texlint 0 问题；`make_tables.py --check` 一致；方法章 `\citep` 为 0；无 `sec:3_6` 引用残留。
- **Facts**：`.gitignore` 的 `figures/` 规则使 `paper/latex_neurips_20261020/figures/` 未进入 aa277c7；本次用 `git add -f` 加入。cho2014learning、shazeer2017outrageously、wang2023graph 不再被引用。
- **Modified but unverified**：未编译 PDF（本机无 TeX）。
- **Next**：作者上传 sections/*.tex 覆盖 Overleaf 并编译；自行重绘图 1。

## 作者意见修改 — 2026-10-25（md 与 LaTeX 同步；不再打包 zip）

- **用户指令**：按作者列出的摘要/引言/方法问题（一.1–一.4、二.1–二.7、三.1–三.3）及同类问题修改；参考 `references/paper/{MoGERNN,SIML-MSIGN}.md` 的写法；`manuscript_v6_cn.md` 与 LaTeX 同时修改；Overleaf 项目已分享导师，作者自行上传覆盖，不再生成 zip。
- **Completed**：快照 `paper/latex_neurips_20261020/drafts/snapshot_20261020_before_r2/`（md、sections、tables、_scripts）；第 2、4 节由 tex 逆向同步回 md（往返逐字节一致）；重写摘要、引言、方法章（图 1 移到第 3 节开头，3.1–3.6 重排，29 个编号公式，方法章不含其他模型名称）、结论；第 2、4 节 51 处术语替换；`md_to_tex.py` 去掉文件头注释、支持 `--draft/--out` 与 `式(n)`→`\eqref`；`make_tables.py` 增加 `ROW_LABEL`（w/o State Age 显示为 w/o State Progress）；更新 README（md 为源头、上传清单）、`notes/V6_CHANGELOG_CN.md`、`notes/FIGURE1_NOTES_CN.md`、`CHANGES_FOR_ADVISOR_CN.md`。
- **Verified**：`make_tables.py --check` 表 1、表 2 与 md 逐格一致；texlint 0 问题（61 labels、124 refs、112 cites）；摘要/引言数值与 `summary.json`、`test/fig2_values.csv`（Join/Fault 24 格全为正）一致；方法公式逐项对照源码（`source_grin.py`、`lifecycle_moge.py`、`ALFGraph.py`、`lcbt_v4.py`、`patchstg.py`、`structural_node_encoder.py`、`ALFNet.py`、`flow_support_transfer.py`、`flow_unified.py`、`v4_lifecycle_loader.py`）。
- **Modified but unverified**：未编译 PDF（本机无 TeX）；新公式行宽需在 Overleaf 编译后确认。
- **Next**：作者上传 `sections/*.tex` 与 `tables/ablation_lifeavg.tex` 覆盖 Overleaf 并编译；按 `notes/FIGURE1_NOTES_CN.md` 修改图 1 标注。

## 阶段 5 文字修改 — 2026-10-20（用户将发送导师）

- **用户指令**：只改文字层面的必改项与建议项，同步 LaTeX 与 Overleaf 包；实验暂不做，看导师安排。三个待定问题未单独表态，按保守口径处理：R6 不加入已有消融结果、S7 维持删除 ST-FiT、保留标题。
- **Completed**：直接修改 `sections/{abstract,introduction,method,experiments,conclusion}.tex`（逐条见 `review_round1/REVISION_CHECK_CN.md`）；英文直引号统一为中文引号；Overleaf 包重建（20 文件，SHA256 ef3812aa…）；更新 `CHANGES_FOR_ADVISOR_CN.md`（作者自用）、README、矩阵 v1.4。
- **Verified**：脚本确认 36 处修改落地、5 处旧表述已删除；新增数字与 `four_dataset_results.json`、`selected_tables.json` 等一致；静态 LaTeX 检查 0 问题；`make_tables.py --check` 通过。
- **Modified but unverified**：未重新编译 PDF（本机无 TeX，需在 Overleaf 用 XeLaTeX 编译确认）；未运行 ARS 正式 re-review 三门流程。`drafts/manuscript_v6_cn.md` 停留在 v6，不再同步。
- **Not started**：所有记录的实验（R2/R3/R4/R7/R8/R9/R12/S5/S10）；英文翻译；NeurIPS checklist；作者信息。
- **Next**：用户在 Overleaf 编译后发给导师；根据导师意见决定实验与标题。

## 阶段 4 / 检查点 4 — 2026-10-20（用户已确认：只做文字修改）

- **Completed**：ARS academic-paper-reviewer full 模式第 1 轮，目录 `paper/latex_neurips_20261020/review_round1/`：`FIELD_ANALYSIS_CN.md`（Phase 0）、五席位 `*.phase1.md`/`*.phase2.md`、`synthesis.md`、`panel-input.json`/`panel-provenance.json`、`acronym_report.md`、`EDITORIAL_DECISION_CN.md`（决定、阻断问题、共识/分歧、R1–R12 必改与 S1–S12 建议，每条标注"文字/实验"）。
- **Verified**：`check_phase_conformance.py` 五席位 Phase 1（`--phase1-only`）与 Phase 2 全部 PASS（Phase 1 仅有按空格计词的 TRIGGER-SHORT 提示；方法学席位附 RECEIPT-ATTESTATION 提示）；`check_panel_synthesis.py` PASS（D1/D3 block → F2 → major_revision，C1=VALIDATED）；`review_panel_provenance.py build/validate` PASS（插件合约文件为 CRLF，LF 规范化后与规范哈希一致，在 scratchpad 的 LF 副本上运行，插件未改）；所有 text 锚点引文与评审副本逐字一致。
- **Provenance（如实记录）**：五席位与作者同一模型、同一会话，`role_separated=true`，其余五轴均为 false；不是独立评审。Phase 0 配置卡未单独停下确认（用户要求直接进入阶段 4）。
- **Not produced**：`revision-roadmap/1.0` 机器文件、Schema 6 carrier。
- **Not started**：阶段 5 文字修改与再评审；所有标为"实验"的条目只记录，未执行。
- **Next**：用户确认阶段 5 的修改范围（尤其 R6 是否加入已有消融结果、S7 与此前删除 ST-FiT 的决定是否冲突、R9/R10 主张与标题如何调整）。

## 阶段 3 收尾 — 2026-10-20

- **用户反馈**：Overleaf 已用 XeLaTeX 编译出 PDF（`D:\ZyCodeR\下载\ALFNet_动态观测支持下的归纳时空预测.pdf`），未发现问题；工作文档（README、修改说明）不进 Overleaf 包，只留仓库；提供 6 个 DOI；PEMS-BAY 时间点数若与官方不一致则修改。
- **Completed / verified**：6 个 DOI 经 Crossref 核对并补入页码；PEMS-BAY 52,116 与 manifest `payload.shape` 一致（名义范围 52,128），`make_tables.py` 改为读取 payload 形状，`--check` 通过；Overleaf 包改为 20 个编译文件（无 README、修改说明、provenance.json），SHA256 a2f0460d…；删除各节文件中指向转换脚本的注释与图1注释中的内部流程字样；静态 lint 0 问题。本机无 PDF 渲染工具，未独立复核 PDF 版面。
- **Next**：阶段 4 模拟评审。

## 阶段 3 / 检查点 3 — 2026-10-20（用户已确认）

- **用户决定**：同意下载 NeurIPS 2026 官方模板；阶段 3 先做**中文 LaTeX**（XeLaTeX + ctex），定稿后整篇译英文。
- **Downloaded**：`paper/latex_neurips_20261020/_template_neurips2026/Formatting_Instructions_For_NeurIPS_2026.zip`（media.neurips.cc，20,259 字节，SHA256 82473931…），只解压（neurips_2026.tex/.sty、checklist.tex），未执行任何文件；`neurips_2026.sty` 原样复制到工程根目录。
- **Completed**：`main.tex`（preprint 模式、natbib numbers+plainnat、ctex scheme=plain、Termes/Heros、仅在导言区覆盖摘要标题为"摘要"）；`sections/*.tex`（由 `_scripts/md_to_tex.py` 从 v6 一次性转换）；`tables/*.tex` 与 `provenance.json`（`_scripts/make_tables.py`）；`figures/`（作者 PPT 图＋图2–6 PDF）；`references.bib`（24 条，逐条联网核对）；`README_overleaf_CN.md`；`CHANGES_FOR_ADVISOR_CN.md`；`notes/CITATION_AUDIT_CN.md`；`paper/latex_neurips_20261020_overleaf.zip`（23 文件，480,348 字节，SHA256 bcabbcdb…，testzip 通过）。
- **Verified**：`make_tables.py --check` 表1/表2 168 值与标记与 v6 一致；四份 run_report 参数量 1,037,529；静态 lint（scratchpad `texlint.py`）：12 个 tex 文件括号/环境配对、47 标签与 46 引用一致、60 处引用均在 bib、24 条全部被引用、所有 `\input`/图片存在、无异常字符。转换时发现并修正一处把"4.5%"误转为小节引用的问题。
- **Modified but unverified**：**未编译**（本机无 TeX），XeLaTeX 编译、字体与版面效果需在 Overleaf 验证；6 篇会议论文 DOI 未查到；未做撤稿筛查；PEMS-BAY 时间点数 52,116 出处待核。
- **Not started**：模拟评审（阶段 4）、修改与再评审（阶段 5）、英文翻译、NeurIPS checklist、作者/致谢信息。
- **Next**：用户导入 Overleaf，用 XeLaTeX 编译并反馈日志；确认后进入阶段 4 模拟评审。

## 阶段 2 修改轮 — 2026-10-20

- **用户反馈**：有英文缩写的术语后补英文；ST-FiT 不在正文出现；4.9 并入结论；改用作者 PPT 架构图 `paper/latex_neurips_20261020/imgs/alfnet.png`（符号与箭头由作者文字定稿后修改）；修改后进入阶段 3。
- **Completed / verified**：v6 已按四点修改（见 `notes/V6_CHANGELOG_CN.md` 第二轮）；角色中文名改为活跃/新接入/未观测/失效以免与场景名冲突；参考文献删 ST-FiT 并重排为 [1]–[24]，脚本核对引用与文献表一一对应；两张 HTML 表 168 个数值与 v5 完全一致；ST-FiT/Merlin 在 v6 中零出现。新增 `notes/FIGURE1_NOTES_CN.md`（PPT 图与公式的 8 处差异，供作者修图）。
- **Stage 3 preparation**：官方 NeurIPS 2026 模板 `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`（HEAD：200，20,259 字节，Last-Modified 2026-06-23）；尚未下载，等待用户同意。
- **Not started**：LaTeX 工程、bib、表格脚本、citation-check、Overleaf 包、导师说明、模拟评审。

## 阶段 2 / 检查点 2 — 2026-10-20（用户已反馈）

- **用户决定（检查点 1）**：B1 ST-FiT、Merlin 直接移除（用户：当时训练结果无效），只写正文四数据集八基线，不写附录；B2 表2 保持 PEMS04 w/o State Age 39.757/53.136/24.957 与"11/12"，正文不提另一条记录；B3 未比较理由写入正文；中文 v6；恢复预测记为 `Ỹ`；摘要先写中文；共享补全与图修正消融不放；不考虑页数。事实记录保留在矩阵 §4。
- **Completed**：`paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md`（摘要、第3节方法 3.1–3.8 与式(1)–(15)、第5节结论为新写；第1、2、4节术语统一与文字修正）；`notes/V6_CHANGELOG_CN.md`；`notes/CITATION_LOG_CN.md`（新增 [19]–[24] 联网核对，[25] 沿用相关工作证据）；矩阵升级为 v1。
- **Verified**：v6 与 v5 第1、2、4节数字逐项比对，差异仅为新增引用编号、小节交叉引用和由 `fig3_values.csv` 算出的 4.0%/6.1%；方法中的维度、开关与公式在写作前复读了 GRU 单元、BiGRIL、年龄规则、两阶段注意力轴（第一阶段沿分区轴、潜在 token 按分区独立）、时间嵌入与结构特征、训练期 cohort（四场景均在已知节点模拟）。
- **Modified but unverified**：v6 全部文字改动尚未经模拟评审；[25] 及 v5 原有 [1]–[18] 未重新联网核对。
- **Not started**：英文 LaTeX、NeurIPS 样式获取、`references.bib`、citation-check、Overleaf 包、导师修改说明、模拟评审。
- **Commands**：只读源码/配置（sed、grep）；Python 比对 v5/v6 数字；WebFetch/WebSearch 查证引用（DBLP、OpenReview 被拦截）。无训练、推理或测试访问；无 commit。
- **Next**：用户检查 v6 后确认进入阶段 3。

## 阶段 1 / 检查点 1 — 2026-10-20（已由用户确认）

- **Checked state**：分支 `paper/final`，HEAD `f06367e08e05e33f2364e7639dc614ab53c983f3`，与 `origin/paper/final` 0/0（未 fetch）。开工时已有改动全部保留：tracked 删除 `paper/manuscript_draft_cn-v4.md`、`paper/manuscript_draft_cn.md`（用户既有改动，未触碰）；未跟踪 `paper/architecture_20261019/claude_handoff_20261019/`、`nanobanana_v4_20261019/`、`paper/writing_20261020/`。
- **Facts（已验证）**：
  - v5 表1 全部 108 个 LifeAvg 值与 `paper/pems08_wd_test_20261012/selected_tables.json` 逐值一致；降幅与 `summary.json` 一致。表2 全部 60 值与 `paper/pems08_wd_ablation_test_20261012/four_dataset_results.json` 一致。图2 非第一格（BAY 4、04 4、08 3）与 `final_figures_20261017/test/fig2_values.csv` 一致。图3/图4/图6 正文数值与 `final_figures_20261017/validation/{fig3,fig4}_values.csv`、`case_selection.json` 一致。
  - 方法关键公式已与源码/最终配置逐项对照（详见 `paper/latex_neurips_20261020/notes/METHOD_OUTLINE_CN.md` 公式清单的"核对状态"列）：refiner、FiLM head/注入、soft router 与五支持、残差 AR 解码、插值/中心化/恢复顺序、Focal 元素损失、0.9/0.1 混合与 router 项、四数据集 C/插值/d/α、β=0.1、角色倍率 [0,1,1,0.25]、P=32（recur_times=5）、4 latent、1 层、1 temporal patch。
  - 四份最终 run_report.json（经 manifest checkpoint 目录定位，位于 artifacts/，非 paper/）`parameter_count` 均为 1,037,529；本地 best.pt 不存在。
  - 本机无 latexmk/pdflatex/xelatex/tectonic/bibtex → 阶段 3 需在 Overleaf 编译。
- **发现的重大问题（待用户/导师决定，未自行修正）**：
  1. ST-FiT 与 Merlin 已在相同协议下训练并完成 test（2026-10-01，117 项 test），后于同日按用户要求从比较表移除。按现有 CSV 离线比对：LifeAvg 第一不变；但 33 格第一数将由 33/29/29/30 变为 30/22/15/19，且"Join/Fault 三指标四数据集均优于全部基线"不再成立（Merlin 在 PEMS04/08 Fault 及中断阶段、ST-FiT 在 PEMS-BAY Join/Cold 领先）。v5 将 ST-FiT 与"未运行"方法并列为"未纳入"，Merlin 未在相关工作出现。
  2. 表2 PEMS04 w/o State Age 使用重复训练记录；原始记录 MAE/RMSE 优于 Full（−0.64%/−1.71%），用原始记录则 State Age 为 10/12 变差而非 11/12。
  3. SAHC（共享 GRIN 补全）与 Role-Scaled Refinement 在正文无消融；完整证据在历史 12 类消融与当前 5 项配对中（Refinement 10/12 变差，PEMS08 去除后 MAE/RMSE 改善）。
- **Completed**：上下文恢复；`paper/latex_neurips_20261020/ALIGNMENT_MATRIX_CN.md`（初版）；`notes/CONTEXT_FACTS_CN.md`（事实摘要、不一致与缺口、PCR）；`notes/METHOD_OUTLINE_CN.md`（方法章大纲、符号表、公式清单、证据映射）。
- **Modified but unverified**：无（本阶段只新增文档）。
- **Not started**：方法/摘要/结论写作；LaTeX 工程与 NeurIPS 样式下载；citation-check；模拟评审；修改。
- **Ruled out**：修改 v5；重跑或补跑任何实验；从历史目录挑选更有利数值；把 GRIN/DCRNN/Crossformer/PatchSTG/FiLM 写成原创。
- **Commands actually run**：git status/log/rev-parse；只读 Python 解析 JSON/CSV（selected_tables、four_dataset_results、fig2/3/4 值、before_model_filter 测试表、ablation_repeat 结果、manifest、run_report）；sed/grep 读源码与 YAML；TeX 工具探测。无训练/推理/test。
- **Next discriminating action**：用户确认检查点 1（尤其上面 1–3 号决策与 PCR），再进入阶段 2。
