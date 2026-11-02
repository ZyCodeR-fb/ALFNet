# 下一阶段任务说明：论文优化（相关工作调研 → 对比与实验缺口 → 引言与相关工作重写）

写于 2026-11-02。供新会话恢复上下文使用。

## 1. 起因：导师意见（2026-11-02 聊天记录要点）

- 主要问题是**对比不够**，容易被找漏洞；实验结果需要更全面地支撑论点。
- 现在多为 AI 辅助审稿：审稿时会先找类似工作，再看论文有没有比较，没有比较就要给出说法。要把目前类似的工作都找到，**确定一个挑选标准，套到已有的结果上**，先把这个漏洞堵住。
- **不着急加实验**，先把潜在相关论文找出来，不一定都要加；然后看哪些实验被忽略或不够，再决定补。
- 调研不足：类似方法的差异是什么、我们的创新点是什么、类似方法都做了哪些实验，目前了解得少。
- 只盯着 MoGERNN 不够；Related Work 太单薄；Introduction 写得浅，没有体现对方向的深入理解，motivation 和要解决的科学问题不够凝练。
- 导师在同步看论文，之后会一起凝练 Introduction 的核心问题。

## 2. 目标与分阶段计划（每阶段结束停下，等作者确认）

**阶段 1：系统调研类似工作（只读与检索，不改正文）**
- 检索范围至少覆盖：未观测位置的归纳时空预测与 kriging；动态传感器网络（传感器新增、失效、变化布局）；部分感知与长期部分感知预测；缺失数据下的时空预测与联合插补预测；冷启动/新节点预测；面向观测掩码或可用性的条件化建模；图专家混合与条件化图传播；空间分区/大规模交通 Transformer。近三年（2023–2026）的 NeurIPS/ICML/ICLR/KDD/AAAI/IJCAI/WWW/CIKM/ICDE/TKDE/TR-C/TITS 等为重点。
- 每篇文献核实题名、作者、年份、会议或期刊、DOI 或 arXiv 号；无法核实的不写入。引用必须真实，不编造 DOI、页码或作者。
- 产出文献矩阵：问题设置（是否归纳、训练与推理时观测是否变化、是否预测未来）、方法要点、数据集、比较基线、评价设置、做过的实验类型（缺失率扫描、未观测比例扫描、新增或失效传感器、效率与参数量、多种子、可视化、消融等）、代码是否公开、与 ALFNet 的差异。

**阶段 2：对比方法的挑选标准与“未比较”的说法**
- 先写出**与结果无关**的挑选标准（如：问题设置是否一致、是否预测未来、能否在统一输入预算与数据暴露协议下运行、是否有可核实的官方实现、发表时间与影响力），再对阶段 1 的候选逐一套用。
- 对照已有结果：哪些方法已在最终协议下有结果，哪些只有旧协议结果，哪些从未运行。
- 对每个未比较的方法给出可核实的理由；对按标准**应当比较却没有结果**的方法，列为“建议补充实验”，交作者与导师决定。
- 不得按 ALFNet 是否占优来设计或调整标准。

**阶段 3：实验缺口分析（只列清单，不运行）**
- 对照类似工作常做的实验和模拟评审已记录的缺口，列出“已有 / 缺少 / 可用已有结果补充 / 需新实验”的清单，按审稿风险排序，并估计每项工作量。

**阶段 4：重写引言与相关工作（需作者确认阶段 1–3 后再开始）**
- 引言：凝练核心科学问题与 motivation，体现对方向的理解；创新点与最接近工作的差异写清楚。
- 相关工作：按阶段 1 的分类扩充，逐类说明与 ALFNet 的差异。
- 在 `drafts/manuscript_v6_cn.md` 修改，再生成 `sections/*.tex`（见第 4 节流程）。

## 3. 已有材料（先读，不要重复劳动）

| 文件 | 内容 |
|---|---|
| `paper/baseline_selection_cn.md` | 2026-09-14 的对比方法选型说明：三条研究方向、候选与推荐名单（含 GWNET、HA、FCLSTM、STGODE、ST-FiT、UIGNN、STAEformer 等） |
| `paper/related_work_cn.md`、`paper/related_work_evidence_cn.md`、`paper/related_work_search_20260914.json` | 早期相关工作稿、证据与检索记录 |
| `paper/BASELINE_RECENCY_AUDIT_20260926.md`、`paper/SELECTION_AND_BASELINES_REVIEW_20260927.md`、`paper/BASELINE_FAIRNESS_20260928.md`、`paper/SLPF_STBP_INTEGRITY_20260926.md` | 基线时效、选择、公平性与适配完整性审计 |
| `paper/all_final_test_20261001/validation_final_provenance.json` | 最终协议下有结果的模型：UIGNN、GinAR、MoGERNN、STFiT、SLPF、STGCN、STID、PatchSTG、STBP、Merlin、ALFNet |
| `paper/pems08_wd_test_20261012/selected_tables.json`、`summary.json` | 论文表 1 的权威结果（8 个基线 + ALFNet） |
| `paper/latex_neurips_20261020/review_round1/EDITORIAL_DECISION_CN.md`、`REVISION_CHECK_CN.md` | 第 1 轮模拟评审：已记录的实验缺口 R2（多种子）、R3（未经选择的留出确认）、R4（基线等预算调参）、R7（Graph WaveNet、AGCRN）、R8（角色误标或延迟的稳健性）、R9（整体去除 OS 条件的对照）、R12（历史均值或持续性等简单基线） |
| `references/paper/MoGERNN.md`、`references/paper/SIML-MSIGN.md` | 作者下载的两篇同领域论文全文（写作风格参考） |
| `paper/latex_neurips_20261020/notes/V6_CHANGELOG_CN.md` | 各轮修改记录与依据 |

## 4. 约束（继续有效）

- 不训练、不评估、不访问测试集、不改模型代码、配置、权重或结果；所有数字只能来自 `paper/` 下已有的表格、JSON 或 CSV，并能追溯到具体文件。
- 需要新实验或新基线时，只记录为“建议，待导师决定”。
- **ST-FiT 与 Merlin**：作者在 2026-10-20 决定从正文移除，理由是当时训练结果无效；检查点 1 曾记录，若纳入二者，“Join 与 Fault 三项指标四个数据集均优于全部基线”的结论将不成立。调研中若按挑选标准它们应当被比较，**如实报告并交作者决定**，不要自行加回，也不要为排除它们而调整标准。
- 写作约定（见用户记忆 paper-writing-conventions）：缩写第一次出现写“中文（英文简称）”，之后只用简称；四类传感器 AAS/NAS/VS/FS，场景 Unseen/Steady/Join/Fault，阶段 Cold/Warm/Ready 与 Onset/Decay/Outage；“状态进度（state progress）”；方法章不提其他模型名称、不放文献编号，引言中必要时只说“借鉴其思想”；不写“最先进”“显著”这类无统计支持的词；局限只保留一两点并写成后续工作；不写“测试结果参与配置决策”。
- 编辑流程：`paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md` 是唯一编辑源头；改完运行 `python paper/latex_neurips_20261020/_scripts/md_to_tex.py` 与 `python paper/latex_neurips_20261020/_scripts/make_tables.py --check`；不再打包 zip，告诉作者需要上传覆盖 Overleaf 的文件清单。README、CHANGES 等工作文档只留在仓库，不上传 Overleaf。
- Git：可以在合适的节点提交，不推送；不要提交作者自己的未提交改动（图片文件、`figures/alfnet.png` 等），除非作者要求。
- 图 1 由作者在 PPT 中重画，复核意见见 `paper/latex_neurips_20261020/imgs/figure_design/ALFNET_PNG_REVIEW_CN.md`，本阶段不处理。

## 5. 产出位置

调研与分析文档写在 `paper/latex_neurips_20261020/survey/`（作者私用，不上传 Overleaf），建议：
- `LITERATURE_MATRIX_CN.md`（阶段 1，附检索式、数据库、日期与核实记录）；
- `BASELINE_CRITERIA_CN.md`（阶段 2）；
- `EXPERIMENT_GAPS_CN.md`（阶段 3）；
- 正文修改按第 4 节流程进行，并在 `notes/V6_CHANGELOG_CN.md` 追加记录。
