# T-20261020-paper-writing

任务说明：[PROMPT_ARS_PAPER_CN.md](../../../../paper/writing_20261020/PROMPT_ARS_PAPER_CN.md)。用 ARS（academic-research-skills 3.23.1）补写方法、摘要、结论 → NeurIPS LaTeX → 模拟评审 → 修改。实验已冻结；不训练、不评估、不访问测试集、不改模型/配置/权重/结果；不 commit/push。

## 补充实验准备：候选方法与服务器提示词 — 2026-11-03

- **用户指令**：2.2 只有 GinAR、2.3 没有比较方法，每类给两个最合适的候选（按训练结果最终选择）；SAHC 与 OS 整体消融；实验在 Ubuntu 服务器上进行，最多同时 3 个，不终止他人进程。
- **Facts**：四个候选的官方仓库均已打开 README 核对：Merlin `ChengqingYu/Merlin`（本地 `references/Merlin/` 已有）、HD-TTS `marshka/hdtts`、STONE `PoorOtterBob/STONE-KDD-2024`、STOP `PoorOtterBob/STOP`。已有 `no_imputer`（`experiments/alfnet_ablation_adapter.py`）只替换补全器、保留插值与中心化；METR-LA、PEMS-BAY 的插值与中心化本就关闭，PEMS04、PEMS08 需重跑；"整体去除 OS 条件化"此前没有实现。ST-FiT 与 Merlin 的旧适配器已不在 `model/comparison_model/`。
- **Completed**：`SERVER_EXPERIMENT_PROMPT_CN.md`（下载清单、服务器端 Claude 的完整提示词：恢复上下文、服务器使用规则、四个方法的适配原则与注意事项、两项消融定义、冻结计划、验证选点后一次测试、结果全部保留由作者选择、产出与汇报）。
- **Not touched**：未训练、未评估；未提交。服务器需要同步 `survey/` 与本文件后才能使用提示词。
- **Next**：作者下载代码、同步仓库，在服务器启动新任务 `T-20261103-extension-experiments`。

## 第八轮：引言可读性与段落衔接 — 2026-11-03（等待作者审阅）

- **用户指令**：独立判断 Codex 对最新稿的写作审核；重点为引言的论证顺序与衔接，不扩充文献；保留第七轮的事实与分类修正；只改文字。
- **Completed**：引言按"具体情形 → 已有能力与较少显式刻画之处 → 设置边界与核心问题 → 核心认识与 OS 变量（四类传感器定义移到此处）→ 信息流与模块协同（借鉴来源合为一句）→ 实证概述 → 贡献"重写；2.1 按目标位置的训练与推理观测条件组织，2.2 改为两条设计轴，2.3 写明固定参数与窗口内缺失不是本文特有，各小节末尾的比较方法名单统一移到 4.1；结论按四项消融的实际干预改写。记录见 `notes/V6_CHANGELOG_CN.md` 第八轮。
- **Verified**：`make_tables.py --check` 一致；46 个引用键均在 bib；六个章节配平；`method.tex` 与表格文件只有换行符差异、无内容变化。
- **Not touched**：方法章、数值、表图、比较方法；未训练、未评估、未访问测试集；未提交。
- **Next**：作者审阅；上传 `sections/{abstract,introduction,related_work,experiments,conclusion}.tex` 与 `references.bib`（abstract、experiments 本轮无变化，若第七轮已上传则只需 introduction、related_work、conclusion）。

## 第七轮：按 Codex 审核理顺论证链 — 2026-11-03（等待作者审阅）

- **用户指令**：独立判断 Codex 对引言、相关工作与全文对应关系的审核；文献事实回查原文；保留架构、八个方法与已有结果；不训练、不评估、不访问测试集、不改数值；只做文字。
- **Facts（本轮回查）**：FRIGATE §4.2 为一小时历史预测一小时、Δ=12，§4.6 推理时丢弃 33.33% 快照且不重训（survey 中"输入长度待核"已更正）；SIML 式 (24) λ_o=λ_u=0.5；GMN、GRU-D、STONE 摘要与正文表述一致；图 3/4/6 代表基线来自 `paper/advisor_experiments_20261012/plan.json` 的旧三组规则（`scripts/advisor_support_20261012.py:19`），排除 ST-FiT/Merlin。
- **Completed**：Codex 十项意见全部接受（依据见 `notes/V6_CHANGELOG_CN.md` 第七轮）；正文摘要、第 1、2 节、4 节引言、4.1、4.2、4.5、第 5 节按此修改，重新生成 sections；新增 `survey/ARGUMENT_MAP_CN.md`；更新 `survey/RELATED_WORK_OUTLINE_CN.md` 第 0 节、矩阵与标准草案中的 FRIGATE 与 UIGNN 记录；`CHANGES_FOR_ADVISOR_CN.md` 第十节。
- **Verified**：`make_tables.py --check` 一致；46 个引用键均在 bib；六个章节 LaTeX 配平；残留措辞检索无问题。
- **Not touched**：方法章、数值、表图、比较方法集合；模型、配置、权重、结果；未训练、未评估、未访问测试集；未提交。
- **Next**：作者审阅后上传 `sections/{abstract,introduction,related_work,experiments,conclusion}.tex` 与 `references.bib`；与导师讨论 2.2 只有 GinAR、SAHC 无消融两项是否补实验。

## 论文优化阶段 2（调整）+ 阶段 4：引言与相关工作重写 — 2026-11-03（等待作者与导师审阅）

- **用户指令**：取舍事项暂缓；按导师意见写好引言、相关工作与创新点，尽量套用已有结果；对比方法以现有八个为基础，尽量少补充或更换；阶段 3 推迟，直接进行阶段 4。任务说明 `NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md` 第 2 节已按此调整。
- **Completed**：阶段 2 定稿写入 `survey/RELATED_WORK_OUTLINE_CN.md` 第 0 节（四个小节、论证顺序、每类沿用的对比方法与暂未比较的近邻）；`drafts/manuscript_v6_cn.md` 第 1、2 节重写，4.1 比较方法段改为按四个方向归类（八个方法与全部数值不变）；新增 24 条文献，编号按首次出现重排为 [1]–[46]；同步 `_scripts/md_to_tex.py` 的 CITE 映射与 `references.bib`；重新生成 `sections/*.tex`；记录 `notes/V6_CHANGELOG_CN.md` 第六轮、`notes/CITATION_LOG_CN.md`、`CHANGES_FOR_ADVISOR_CN.md` 第九节。
- **Verified**：`make_tables.py --check` 两表一致；46 个引用键均在 bib 中、无重复；三个章节 LaTeX 括号与 `$` 配平、无残留编号；`sections/experiments.tex` 只有比较方法一段变化。
- **Modified but unverified**：未编译 PDF。
- **Not touched**：方法、实验数值与表图、摘要、结论；模型、配置、权重、结果；未训练、未评估、未访问测试集；未提交。
- **Next**：作者上传 `sections/introduction.tex`、`sections/related_work.tex`、`sections/experiments.tex`、`references.bib` 覆盖 Overleaf 并编译；与导师讨论核心问题表述与四类划分；实验部分（含未比较近邻的说明）待对比方法与 ALFNet 结构确定后进行。

## 论文优化阶段 1 修订 + 相关工作提纲 + 阶段 2 标准草案 — 2026-11-03（等待作者确认）

- **Input**：Codex 核验 `survey/CODEX_LITERATURE_AUDIT_20261103_CN.md` 与修订要求 `survey/PROMPT_FOR_CLAUDE_SURVEY_REVISION_20261103_CN.md`（作者转来，要求自行判断）。逐条回到原始来源核对：7 处实质问题、补漏条目、DOI 与代码链接全部成立；FRIGATE "12→12" 只核到预测 12 步，输入长度待核。
- **Completed**：`survey/LITERATURE_MATRIX_CN.md` 改写为 v2（v1 未入 git，已覆盖；v1 检索式与差异保留在 v2 第 0、1.2 节）；新增 `survey/LITERATURE_METADATA_APPENDIX_CN.md`（88 篇唯一论文、89 个版本的完整元数据与核实路径，旧 28 项映射）、`survey/RELATED_WORK_OUTLINE_CN.md`（四类提纲）、`survey/BASELINE_CRITERIA_CN.md`（三种资格、S1–S12、信息账本、逐候选判断、需取舍事项）；原始记录 `survey/verification_20261103/`。
- **Facts**：HD-TTS 官方数据代码切窗前整序列 `ffill().bfill()`（静态阅读）；SIML 式 (24) λ_o=λ_u=0.5 监督未观测节点；SCPT §3.6 在 A∪D∪G（含测试道路训练时段）上做 DCT；ST-FiT/Merlin 2026-10-01 评估 `verification.json` passed，权重 2026-10-17 已删除；`paper/`、`artifacts/` 下未找到最终模型逐样本预测。
- **Not touched**：正文、LaTeX、实验表、模型、配置、权重、结果；Codex 两份文件；作者未跟踪的 `paper/comparison_ten_baselines_20261102/`；未训练、未评估、未访问测试集；未提交。
- **Next**：作者确认分层近邻、四类提纲、标准草案，并决定 `BASELINE_CRITERIA_CN.md` 第 5 节五项取舍；之后进入阶段 3（实验缺口与工作量）或先做 ST-FiT/Merlin 只读有效性审计。

## 论文优化阶段 1：类似工作调研 — 2026-11-02（等待作者确认后进入阶段 2）

- **Checked state**：`paper/final`，HEAD `10cb396`，开工时工作区干净。
- **Completed**：`paper/latex_neurips_20261020/survey/LITERATURE_MATRIX_CN.md`（75 篇：核心 18 张卡片 + 扩展 57 条；检索式、数据库、日期、纳入标准、逐篇核实路径、未列入线索、实验类型对照、已有结果状态、ST-FiT/Merlin 事实记录、初步挑选标准 S1–S9、实验缺口 G1–G13）。原始核实记录在 `survey/verification_20261102/`。
- **Facts**：元数据经 Semantic Scholar + Crossref（或 DBLP 键、arXiv 注释、会议页面）两路核对；DBLP 网页被人机验证拦截，arXiv 官方 API 当日 429。ST-FiT、Merlin 的测试格在 `paper/all_final_test_20261001/validation_selected_test_cells.csv`；`docs/ai/` 下未找到"结果无效"的技术说明。
- **Not touched**：正文、sections/*.tex、模型、配置、权重、结果；未训练、未评估、未访问测试集；未提交。
- **Commands**：WebSearch、WebFetch；Semantic Scholar、Crossref、OpenAlex 公开 API（scratchpad 脚本）；只读 grep/sed/Python 读取仓库文件。
- **Next**：作者确认最接近工作、挑选标准初稿与实验缺口清单，并决定 ST-FiT/Merlin 的处理，再进入阶段 2（`survey/BASELINE_CRITERIA_CN.md`）。

## 图 1 重绘支持与下一阶段交接 — 2026-10-29 至 11-02

- **Completed**：方法章定稿（作者确认）。图 1 设计方案 `paper/latex_neurips_20261020/imgs/figure_design/FIG1_DESIGN_CN.md` 与 Codex 提示词 `CODEX_PROMPT.md`；经作者授权操作 Codex 桌面应用生成版式草稿与素材（`figure_design/codex_out/`，Codex 只写入该目录）；复核 Codex 结果 `CODEX_OUT_REVIEW_CN.md`；复核作者 PPT 重绘的 `figures/alfnet.png`（2026-10-31 版），连线、符号、样式问题共 19 项，见 `ALFNET_PNG_REVIEW_CN.md` 与 `alfnet_review_annotated.png`。
- **Not touched**：作者自己的图片改动（`figures/alfnet.png`、`imgs/alfnet.png`、`imgs/Reference_Image/`、`paper/architecture_20261017/Reference_Image/` 的删除）未提交；`figure_design/` 未提交。
- **Next**：论文优化，按 [NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md](NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md) 执行（导师 2026-11-02 意见：对比不够、相关工作单薄、引言与 motivation 不够凝练；先调研类似工作，确定挑选标准套到已有结果，再列实验缺口，不急于加实验）。图 1 由作者自行修改。

## 作者第三批意见修改 — 2026-10-27（提交 git，不推送）

- **用户指令**：摘要阶段映射核对；方法章去掉“部分数据集启用”与图位置说明、rn/P 定义移到 3.2、SAHC 直接用 S_t、GRU 定义顺序、两阶段注意力切片记号、MoE 路由表述、Teacher Forcing、L_route 定义位置与 L_mean 说明、全章符号复核；4.1 补 Ready/Steady 与 Outage/Unseen 的区别。
- **Facts**：阶段映射正确（`v4_schedule.py::_phase_id`）；SAHC 的 [x, m, R, a, h] 与 [x, S, h] 只差拼接顺序。
- **Completed**：重写方法章（30 个编号公式）；摘要与 4.1、4.4 修改；更新 changelog 第五轮、FIGURE1_NOTES、CHANGES_FOR_ADVISOR、README。
- **Verified**：重建的 texlint（scratchpad，静态检查）0 问题；`make_tables.py --check` 一致；方法章无图位置说明、无“传播支持/教师强制/η”残留、无数据集特定表述。
- **Not touched**：用户在工作区删除的 `drafts/snapshot_20261020_before_r2/` 未纳入提交，留给用户处理。
- **Modified but unverified**：未编译 PDF。
- **Next**：作者上传 abstract/method/experiments 三个 tex 并编译；重绘图 1 时参考 `notes/FIGURE1_NOTES_CN.md`。

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
