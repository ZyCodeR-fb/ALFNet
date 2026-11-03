# 文献矩阵 v2：动态观测支持下的归纳时空预测——类似工作调研（阶段 1 修订）

- 版本：v2，2026-11-03。v1 写于 2026-11-02，SHA256 为 `3A954F97…C572FC`（Codex 核验时记录，改写前复核一致）。v1 未纳入 git，已被本版覆盖；v1 的检索式全文保留在第 1.2 节，v1 与 v2 的差异逐条列于第 0 节。
- 范围：任务说明 `docs/ai/handoffs/T-20261020-paper-writing/NEXT_PAPER_OPTIMIZATION_BRIEF_CN.md` 阶段 1 的修订。只读与检索；未训练、未评估、未访问测试集，未修改正文、LaTeX、实验表、模型、配置、权重或结果，未提交。
- 修订依据：Codex 核验报告 `CODEX_LITERATURE_AUDIT_20261103_CN.md` 与作者转来的修订要求 `PROMPT_FOR_CLAUDE_SURVEY_REVISION_20261103_CN.md`。Codex 报告作为待核意见处理，每一条都回到原始来源重新核对，结论见第 0.2 节。
- 配套文件：
  - `LITERATURE_METADATA_APPENDIX_CN.md`：全部条目的完整题名、作者、出处、DOI、arXiv 版本、官方来源、核实路径与访问日期；旧调研 28 项的复核映射。
  - `RELATED_WORK_OUTLINE_CN.md`：相关工作四类提纲。
  - `BASELINE_CRITERIA_CN.md`：阶段 2 基线标准草案与逐候选判断。
  - `verification_20261102/`、`verification_20261103/`：两轮核实的原始记录。
- 工具：ARS（academic-research-skills 3.23.1）`deep-research` 的 `lit-review` 模式，在主会话内执行；本文件由 AI 辅助检索与整理，结论供作者核对。

---

## 0. 本版改动

### 0.1 实质更正

| 位置 | v1 | v2 | 依据 |
|---|---|---|---|
| C12 HD-TTS | "无交通数据集" | 附录 F / 表 6 有 METR-LA、PEMS-BAY 的时间块与时空块缺失实验；正式页码 PMLR 235:34846–34865；官方 LA 配置为 24→12；官方数据代码在切窗前对整条序列做 `ffill().bfill()` | arXiv v2 附录 F；PMLR 页面；`config/dataset/la.yaml`；`experiments/run_realworld.py` 第 168–178 行（存于 `verification_20261103/`） |
| E10 FUNS | "预印本，未正式发表" | ICDM Workshops 2022，740–747，DOI 10.1109/ICDMW58026.2022.00101；升级为核心卡片 C22 | Crossref；arXiv 摘要页；代码页 |
| C16 UniSTOK | 只按 v1 描述 | v1 与 v2 分列：v2（2026-05-09）题名、作者与方法（RSR、RBC）均已改变；掩码仿射调制只能以 v1 引用 | arXiv v1、v2 摘要页 |
| C05 STSM | 缺 DOI；D5 只写"输入 24 步" | DOI 10.48786/edbt.2024.42（DataCite）；训练期选择性遮蔽使用 OpenStreetMap 的 POI、建筑层数、公园面积与道路属性，记入 D5 | 官方 PDF 首页与 §4.1；DataCite |
| C02 SIML-MSIGN | "MSIGN 零步是，SIML 否"；D1 为"推断" | MSIGN 单任务、SIML 零步（Step-0）、SIML 少步（Step-5）分开；式 (24) 与表 1（λ_o＝λ_u＝0.5）表明训练同时监督未观测节点，D1 为"否"；"无代码"改为"截至 2026-11-03 未定位可核验的官方实现"；数据保密声明与使用公开基准分开记录 | 本地全文式 (24)、表 1、§4.1.2、§4.2 |
| §2.2 D1 | ALFNet 的目标节点训练期都无信号 | 按场景分开：Unseen、Join 的目标是保留节点；Steady、Fault 的目标是已知节点；保留节点上的失效没有评价 | 正文 4.1 节 |
| §0、§8.1 | "未见同时满足四点的工作" | 撤回这一合取式结论；改为分别说明问题覆盖、计算机制、评价设计三类差异，阶段评价不作为方法新颖性的证据 | Codex 报告第 3 节第 7 条；本版第 8.1 节 |
| C07、C13 等 | 摘要级证据下写"否" | 改为"未报告（摘要级）"或"待核" | Codex 报告；本版第 3 节 |
| C08 SLPF | 记录旧审计的 OpenReview ID `Fj08aPjVjD` | arXiv v3（2026-09-30）正文写 `Ff08aPjVjD`，与正文参考文献一致；旧值很可能是 PDF 文字提取误差 | arXiv v3 HTML |
| C04 ST-FiT | "只覆盖相当于 Ready 的条件" | 限定为源论文实验设置的对照；本项目适配版本已在全部场景上运行 | 本项目最终协议结果 |
| §1.5 等 | "75 篇（18+57）" | 编号条目、唯一论文、版本分开计数：88 篇唯一论文、89 个版本 | 附表第 2 节 |
| §9.1 | "E01–E57 均至少两路核对" | 撤回；S2 返回的 DBLP 键不另计；逐条路径见附表 | 附表第 1 节 |

### 0.2 对 Codex 报告的处理

| Codex 意见 | 处理 | 本轮核对结果 |
|---|---|---|
| 第 1 节 7 处实质问题 | 全部接受 | 每条都已回到原文或出版方记录核对，见 0.1 |
| FRIGATE 应为核心卡片，"原实验 12→12" | 接受 | 第七轮（2026-11-03）回读 §4.2：原文写明"用一小时历史预测一小时"、Δ＝12（5 分钟采样）。v2 初稿记为"输入长度待核"是因为当时没有读到 §4.2，已更正 |
| SCPT 使用目标道路两天历史，§3.6 涉及训练未见道路的信号 | 接受 | §3.2 的 3×3 划分中 D、G 为验证道路与测试道路的训练时段；§3.6 在 A∪D∪G 上做逐路 DCT。网页摘要工具一度给出相反结论，已按逐字引文纠正 |
| STOP 需核新增节点设置 | 接受并已核 | 约 75% 节点训练；测试时遮蔽 10%、新增 30%；推理不更新参数（附录 E.2、§4.7） |
| 补 GenST、GSTAE、STA-GANN、RADGCN、ST-TTC、RT-GCN、GMN/SGMN | 全部接受 | 元数据经 Crossref 或 arXiv 核实；摘要经 OpenAlex 核读（GMN 无摘要） |
| 补 6 个 NeurIPS DOI 与 5 个代码链接 | 全部接受 | 6 个 `10.52202` DOI 均经 Crossref 解析；代码仓库网页均返回 200 |
| NeurIPS 2026 poster 139842 作者待核 | 部分补全 | 会议页面两次读取一致：作者 Xile Wang、Mingqi Yang，形式为 Poster，OpenReview `1DWmoUiM00`；轨道与全文仍待核 |
| S1–S9 修订、增设 S10–S12、S8 具体参照 | 接受，并在 `BASELINE_CRITERIA_CN.md` 中具体化 | 个别细化见该文件 |
| G5 改为条件句 | 接受，并补查 | 在 `paper/`、`artifacts/` 下只找到 2026-09 机制分析的配对预测（`artifacts/analyses/observation_support_v1/paired_predictions.csv` 等），未找到最终模型的逐样本预测 |

---

## 1. 检索方法

### 1.1 数据库与工具

| 来源 | 用途 | 可用性 |
|---|---|---|
| WebSearch | 发现候选 | 两轮均可用 |
| Semantic Scholar Graph API | 题名匹配，取作者、出处、DOI、arXiv 号 | 可用，限流明显；它是聚合数据库，返回的 DBLP 键不另计来源 |
| Crossref REST API | 解析 DOI，核对题名、作者、出处、卷期页、日期 | 两轮均可用 |
| DataCite API | OpenProceedings 等 DataCite 注册的 DOI | 第二轮可用 |
| OpenAlex API | 摘要 | 可用 |
| arXiv 摘要页与 HTML 全文 | 版本历史、实验设置 | 可用；arXiv 官方 API 两日均 429 |
| 出版方或会议页面与 PDF | PMLR、ICLR/NeurIPS 论文集、NeurIPS 2026 会议页、OpenProceedings | 可用；PDF 以 `pdftotext` 本地提取；MDPI 返回 403 |
| GitHub 仓库网页 | 代码链接是否存在 | API 被限流，改查网页状态码 |
| 本地全文 | MoGERNN、SIML-MSIGN | 可用 |
| DBLP | — | 网页被人机验证拦截，未使用 |

### 1.2 检索式

第一轮（2026-11-02，WebSearch 标准模式），与 v1 相同：

1. `inductive traffic forecasting unobserved locations 2025 2026 graph neural network`
2. `spatio-temporal kriging inductive 2025 KDD AAAI ICLR unobserved nodes`
3. `traffic forecasting sensor failure new sensors dynamic sensor network 2025`
4. `spatio-temporal forecasting with missing values 2025 graph neural network KDD NeurIPS ICLR`
5. `partial sensing traffic forecasting long-term unsensed locations 2025 2026`
6. `cold-start traffic forecasting new nodes inductive spatio-temporal limited data 2025`
7. `mixture of experts spatio-temporal graph traffic forecasting 2024 2025 TESTAM routing experts`
8. `missingness-aware conditioning observation mask spatio-temporal forecasting irregular sensor availability 2025`
9. `continual traffic forecasting expanding sensor network new nodes 2024 2025 KDD ICLR`
10. `large-scale traffic forecasting transformer spatial partition efficient 2025 2026`
11. `joint imputation and forecasting traffic incomplete data diffusion 2025 2026`
12. `spatio-temporal extrapolation unobserved regions forecasting 2025 2026 arXiv kriging forecasting future`
13. `"inductive" spatio-temporal forecasting unseen nodes 2026 ICLR OR KDD OR AAAI OR WWW traffic`
14. `Transportation Research Part C 2025 2026 traffic state prediction unobserved locations sparse sensors graph neural network inductive`
15. `IEEE TITS 2025 traffic forecasting missing data incomplete observations inductive graph`
16. `spatio-temporal out-of-distribution new nodes added test STONE KDD 2024 spatial shift traffic`
17. `spatio-temporal graph prompting transfer learning forecasting kriging extrapolation data-scarce CIKM 2024`
18. `spatio-temporal foundation model zero-shot unseen cities nodes traffic OpenCity UniST 2024 2025`
19. `traffic forecasting missing data mixture of experts missing patterns 2025 robust spatio-temporal`
20. `NeurIPS 2025 spatiotemporal forecasting missing data graph sparse observations`
21. `kriging traffic 2025 TKDE OR TITS OR "Information Fusion" inductive unobserved sensors forecasting future time steps`
22. `STRAP spatio-temporal pattern retrieval out-of-distribution new nodes traffic ST-TTC test-time`
23. `Flex-MoE arbitrary modality combination mixture of experts missing modalities NeurIPS 2024`
24. `traffic forecasting robust to sensor failures node dropout graph neural network 2025 2026 sensor outage forecasting`
25. `inductive spatiotemporal forecasting newly installed sensors without retraining zero-shot 2026 arXiv graph`

另有按题名或编号的定向核实检索：UniSTOK、MAS-Mamba、Taming Local Effects、STGNP、STSM、XXLTraffic、TFMoE、Merlin 与 GinAR+、Robust Traffic Forecasting against Spatial Shift over Years、Graph Mixture of Experts。

第二轮（2026-11-03）：不新增开放检索，按 Codex 报告给出的线索逐条定向核实（FRIGATE、SCPT、STOP、GenST、GSTAE、STA-GANN、RADGCN、ST-TTC、RT-GCN、GMN/SGMN、NeurIPS 2026 poster 139842），并复核旧调研中未重核的 STGODE、STGformer、AGCRN。

说明：Codex 报告补出的 FRIGATE、SCPT 是第一轮检索式本应覆盖的方向（"partial sensing""new roads"），第一轮遗漏说明 WebSearch 召回有限；"检索未命中"不等于"不存在"。

### 1.3 纳入标准

| 维度 | 纳入 | 只记录线索 |
|---|---|---|
| 主题 | 第 2 节四类之一，与 ALFNet 的问题设置、计算机制或评价设计有可说明的关系 | 只共享宽泛关键词 |
| 时间 | 重点 2023–2026；更早的工作在其为直接思想来源或领域经典时纳入 | — |
| 出处 | 同行评审正式版本优先，区分主会与 workshop；高相关预印本保留并标注版本 | — |
| 可核实性 | 题名、作者、年份、出处可由官方来源确认；DOI 或 arXiv 号有则记录，ICLR/TMLR/ICML 等没有正式 DOI 的以官方永久页面为准 | 任何一项核实不了 |

### 1.4 证据等级（四项分开记录）

| 项 | 等级 |
|---|---|
| 元数据 | 附表"元数据核实路径"一列，按独立来源计数 |
| 设置 | F：读过原文对应段落（卡片中注明节、式、表）；A：摘要或出版记录级；S：只有检索片段 |
| 实验类型 | 同上；第 6 节中的"—"只表示所读范围内未见，不等于论文没有 |
| 代码 | 未检索 / 论文给链接 / 仓库页可访问 / 关键实现已检查（静态阅读）/ 本项目已适配运行（非源方法复现） |

### 1.5 数量

| 项 | 数量 |
|---|---:|
| 唯一论文 | 88 |
| 被引用版本（UniSTOK 分 v1、v2） | 89 |
| 核心卡片 | 22（C01–C22） |
| 扩展条目 | 66（E01–E67，E10 已并入 C22） |
| 检索到但未列入 | 见第 9.2 节 |

---

## 2. 分类框架与比较维度

### 2.1 四类（与相关工作提纲一致）

| 类别 | 研究问题 | 主要条目 |
|---|---|---|
| 一 未观测或新位置的归纳预测与部分感知 | 训练期没有某些位置的动态信号时，如何预测这些位置的未来 | C01 MoGERNN、C19 FRIGATE、C22 FUNS、C03 UIGNN、C04 ST-FiT、C20 SCPT、C05 STSM、C06 GenCast、C08 SLPF、E64 GenST；克里金类 C15 KITS、C16 UniSTOK、E01–E09、E61 作任务边界 |
| 二 不完整观测下的预测与观测条件化 | 输入有缺失时如何预测，模型能否依据缺失模式调整计算 | C12 HD-TTS、C11 Merlin、C09 GinAR、C10 GinAR+、C14 GCN-M、C13 MAS-Mamba、E58 GSTAE、E60 GMN、E59 RT-GCN、E19–E25；插补类 E13–E18 作来源 |
| 三 传感网络变化、持续学习与测试期适应 | 传感器增减或分布变化时，模型是零步使用、用目标支持期适应，还是在线更新 | C02 SIML-MSIGN、C07 SC-SAMG、C21 STOP、E40 STONE、E42 STRAP、E63 ST-TTC、E33–E39、C18 EvoXXLTraffic、E43 XXLTraffic、E65 |
| 四 条件化图传播与跨节点表示 | 空间传播与跨节点上下文能否随输入条件改变 | C17 TESTAM、E44 GMoE、E45 Flex-MoE、E46 FiLM、E24 GRU-D、C16 UniSTOK v1、E47–E57、E26 |

### 2.2 设置对照维度

| 代号 | 维度 | ALFNet |
|---|---|---|
| D1 | 目标节点在训练期是否没有动态信号与标签 | 按场景不同：Unseen、Join 的目标是保留节点（是）；Steady、Fault 的目标是已知节点（否）。训练时四类场景都在已知节点上模拟。保留节点池由固定文件给出、验证与测试按种子各分一半（`lib/runtime_context.py::split_unseen_pool`）；PEMS-BAY、PEMS04、PEMS08 分别为 81/325、77/307、42/170 个，约 25%；METR-LA 的节点文件不在 `data_v4/`，未核对 |
| D2 | 推理时同一节点在一个窗口内是否发生可读与不可读之间的转换 | 是：Join（保留节点由不可读到可读）、Fault（已知节点由可读到不可读）；保留节点上先接入后失效的情形没有评价 |
| D3 | 预测未来 | 是，12 步 |
| D4 | 推理时参数固定 | 是 |
| D5 | 信息预算：推理时读取什么、训练期使用什么额外信息 | 推理只读窗口内 12 步合法交通值、掩码、观测角色、状态进度、日历、拓扑与静态结构特征（坐标、度）；训练期另有角色均衡损失与已知节点上的场景模拟 |
| D6 | 观测状态是否显式进入计算 | 是（角色、状态进度、掩码进入补全、路由与输出修正） |

---

## 3. 核心卡片设置对照

"未报告"表示所读部分没有相关信息；"待核"表示需要读全文才能判断。

| 编号 | 工作 | 出处 | D1 | D2 | D3 | D4 | D5 要点 | D6 | 代码状态 |
|---|---|---|---|---|---|---|---|---|---|
| C01 | MoGERNN | TRC 2025 | 是（VS） | 部分：传感器类型按测试配置定义，所读 §5.3 未见窗口内转换 | 是 | 是 | 12→12，掩码 | 部分：失效传感器以掩码输入 | 本项目已适配运行 |
| C02 | SIML-MSIGN | TRC 2026 | 否：式 (24) 监督未观测节点 | 部分：掩码随任务变化，窗口内按节点固定 | 是 | MSIGN 与 SIML Step-0 是；Step-5 否 | 历史 48 步（可配置性待核）；无外部数据 | 是：掩码门控扩散 | 未定位 |
| C03 | UIGNN | ICDM 2023 | 是 | 否 | 是 | 是 | 源历史 24 步，本项目适配为 12 | 未报告 | 本项目已适配运行 |
| C04 | ST-FiT | AAAI 2025 | 是 | 源实验中否（推理时有完整历史） | 是 | 是 | 12→12 | 未报告 | 本项目已适配运行 |
| C05 | STSM | EDBT 2024 | 是（区域级） | 否 | 是 | 是 | 2 小时→2 小时；训练期用 OSM 的 POI、建筑、公园、道路属性 | 否 | 论文给链接 |
| C06 | GenCast | AAAI 2026 | 是（区域级） | 否 | 是 | 是 | 2 小时→2 小时；ERA5 天气、路网属性、POI | 未报告 | 仓库页可访问 |
| C07 | SC-SAMG | Sensors 2026 | 部分：训练期缺席，适应期用 1–7 天目标观测 | 未报告 | 是 | 否：一阶元学习适应 | 目标支持期 | 未报告 | 未检索到 |
| C08 | SLPF | TMLR 2025 | 否：训练期使用未感知位置的临时传感器真值 | 否 | 是（8 小时） | 是 | 1 小时→8 小时 | 未报告 | 本项目已适配运行（单阶段） |
| C09 | GinAR | KDD 2024 | 否：直推，缺失变量的未来值作为目标 | 否：整窗缺失 | 是 | 是 | 12→12 | 部分 | 本项目已适配运行 |
| C10 | GinAR+ | TKDE 2025 | 待核 | 待核 | 是 | 是 | 待核 | 待核 | 未定位独立实现 |
| C11 | Merlin | KDD 2025 | 否：直推 | 部分：点缺失，含随时间变化的缺失率 | 是 | 是 | 12→12；训练期教师使用完整观测 | 未报告 | 本项目已适配运行 |
| C12 | HD-TTS | ICML 2024 | 否：直推 | 是：时间块、时空块缺失 | 是 | 是 | 交通 24→12；切窗前整序列前向与后向填补 | 是：注意力以缺失模式为条件 | 关键实现已检查（静态） |
| C13 | MAS-Mamba | IET ITS 2026 | 未报告 | 是：传感器中断 | 是 | 是 | 待核 | 是：掩码与距上次有效读数的时间 | 未检索 |
| C14 | GCN-M | DMKD 2023 | 未报告 | 是 | 是 | 是 | 多尺度历史记忆需要窗外历史（旧审计） | 是 | 论文给链接 |
| C15 | KITS | AAAI 2025 | 是 | 否 | 否（克里金） | 是 | — | 未报告 | 论文给链接 |
| C16 | UniSTOK v1 / v2 | arXiv 2026 | 是 | 是：已观测传感器有块缺失 | 否（克里金） | 是 | — | v1 是：掩码调制生成仿射参数；v2 待核 | 未提供 |
| C17 | TESTAM | ICLR 2024 | 未报告 | 未报告 | 是 | 是 | 待核 | 未报告（按交通状况路由） | 论文给链接 |
| C18 | EvoXXLTraffic | arXiv 2026（基准） | 部分：逐年新增传感器 | 年度粒度 | 是 | 依方法而定 | 长时序 | — | 论文给链接 |
| C19 | FRIGATE | KDD 2023 | 是 | 部分：测试时随机丢弃快照 | 是（Δ＝12） | 是 | 12→12（§4.2）；Lipschitz 嵌入（拓扑）；窗口内邻居统计 | 未报告显式掩码 | 仓库页可访问 |
| C20 | SCPT | DMKD 2024 | 部分：主干不训练新道路，但读取其两天历史，周期分解使用其训练时段数据 | 否 | 是（12 步） | 是 | 两天目标历史；训练时段全道路 DCT 周期成分 | 未报告 | 仓库页可访问 |
| C21 | STOP | ICML 2025 | 部分：测试时新增 30% 节点（有输入历史） | 否 | 是 | 是 | 待核 | 未报告 | 仓库页可访问 |
| C22 | FUNS | ICDMW 2022 | 是 | 未报告 | 是 | 是 | 待核 | 未报告 | 仓库页可访问 |

---

## 4. 核心卡片

完整元数据见附表。每张卡片末尾分列四项证据。

### C01 MoGERNN

- **出处**：Zhou 等，TRC 174, 105080（2025），DOI 10.1016/j.trc.2025.105080；arXiv:2501.12281。
- **问题设置**（§3、§5.1）：部分观测路网上预测未观测与已观测位置的未来交通状态；定义 VS、FS、NAS、AAS 四类传感器，ALFNet 的四类观测角色名称即来自这里。§5.3 把按 §5.2 配置训练的模型直接用于加入 NAS 与 FS 的新配置，并与重训模型比较。12 步预测 12 步；训练与测试 7:3，验证集与训练集相同；滑窗步长 12；训练时随机遮蔽 25%。
- **方法要点**：图专家混合（多个聚合器 + 稀疏门控）为未观测节点构造表示，再进入图编码—解码器。
- **数据集与基线**：METR-LA、PEMS-BAY；OKriging+ED、KNN+ED、TGCN、GMSDR、IGNNK、INCREASE、FUNS、UIGNN。
- **实验类型**：15/30/60 分钟；结构变化适应性（对照重训）；VS 与 AAS 比例；消融；激活专家数与推理时间；专家激活随时间的可视化。§5.1–5.6 中未检索到多种子或标准差。
- **与 ALFNet 的差异**：传感器类型按配置定义，所读部分未见窗口内的部分历史；MoGERNN 依据节点输入特征稀疏选择专家，在编码前构造未观测节点表示，ALFGraph 在每一步、每个门按观测状态稠密路由，不读取信号值；ALFNet 另有共享补全与 FiLM 条件分支；评价协议不同（验证集即训练集、步长 12 对比 60/20/20 时间划分与不相交保留节点）。
- **证据**：元数据 CR+S2+LOC；设置与实验 F（本地全文 §3、§5）；代码：论文 §5.1 给链接，本项目已适配运行。

### C02 SIML-MSIGN

- **出处**：Zhao 等，TRC 185, 105564（2026），DOI 10.1016/j.trc.2026.105564。
- **问题设置**（§2.1）：固定物理路网上的部分到全局交通状态预测；每个窗口的观测掩码 $\mathcal{M}_o\in\{0,1\}^N$ 按节点给定，不同布局组织为不同任务。
- **训练监督**（式 (24)、表 1）：$\mathcal{L}=\lambda_o\mathcal{L}_o+\lambda_u\mathcal{L}_u$，λ_o＝λ_u＝0.5，即训练同时用未观测节点的未来真值监督。
- **三种使用方式**：MSIGN 单任务训练与跨掩码泛化（§4.1.2：在一个 65% 覆盖的掩码上训练，冻结参数，在 10 个新随机掩码上测试）；SIML 零步（§4.2，Step-0：学到的初始化直接用于元测试任务，不做任务内更新）；SIML 少步（Step-5，内循环 5 步）。未观测节点输入为常数占位（§4.1.2）。历史长度 48（表 1）。
- **数据集与基线**：METR-LA、PEMS-BAY；GCN、DCRNN、T-GCN、STGCN、ASTGCN、AGCRN、IGNNK、MoGERNN；元学习部分另有 STL、JT、MT-FT、Reptile。指标只有 MAE 与 MAPE。
- **实验类型**：观测比例 50%–80%；跨掩码泛化（含标准差）；MSIGN 消融；Step-0 与 Step-5；适应动态；SIML 统计与消融分析；训练耗时。
- **代码与数据**："Data availability"写明数据保密，但 §3.1 使用的是公开的 METR-LA、PEMS-BAY，二者之间的关系待作者澄清；截至 2026-11-03 未定位可核验的官方实现。
- **与 ALFNet 的差异**：训练监督覆盖未观测节点，不属于节点信号归纳；掩码在窗口内按节点固定；没有观测角色与状态进度；SIML 少步需要目标支持样本与梯度更新。历史 48 步属于可配置窗口还是结构必需，待核。
- **证据**：元数据 CR+S2+LOC；设置与实验 F（本地全文 §2.1、式 (24)、表 1、§4.1–4.2）；代码未定位。

### C03 UIGNN

- **出处**：Mei 等，IEEE ICDM 2023, 1223–1228，DOI 10.1109/ICDM58522.2023.00152；长版 arXiv:2309.06800v5。
- **设置**：目标位置完全没有历史，在已观测位置上训练，预测已观测与缺失位置的未来并给出不确定性；源历史 24 步，15/30/60 分钟。
- **数据集与基线**：METR-LA、PeMS-BAY、SeData；Kriging、KNN、ST-MVL、IGNNK 各加 DGCN 的两阶段方法。
- **实验类型**：RMSE、MAE、R²；不确定性校准；基于不确定性的传感器布设；多步长。
- **与 ALFNet 的差异**：只对应 Unseen；静态。源历史 24 步在本项目已适配为 12 步，说明窗口长度属于可配置项。
- **证据**：元数据 CR+S2+AX；设置与实验 F（arXiv HTML 实验部分）；代码：本项目已适配运行。

### C04 ST-FiT

- **出处**：Lei 等，AAAI 39(11), 12031–12039（2025），DOI 10.1609/aaai.v39i11.33310；arXiv:2412.10912v2。
- **设置**：多数节点没有时间训练数据（默认 10% 节点有数据）；源实验中这些节点推理时有 12 步输入历史；12→12；时间 70/20/10；不微调。
- **方法要点**：时间数据增强（合成节点与混合）与空间图拓扑学习。
- **数据集与基线**：PEMS03、PEMS04、PEMS08；HA、FC-LSTM、STGCN、STGODE、TransGTR。
- **实验类型**：训练节点比例 5%–100%；消融；超参数；3 个种子均值±标准差；每轮训练时间；3/6/12 步。
- **与 ALFNet 的差异**：源实验设置只对应"训练无信号、部署有完整历史"，即 Join 的 Ready 一类条件；这是源实验范围，不是方法能力边界，本项目适配版本已在四类场景上运行。ST-FiT 不以观测状态为条件。
- **证据**：元数据 CR+S2+AX；设置与实验 F（arXiv HTML）；代码：本项目已适配运行。

### C05 STSM

- **出处**：Su 等，EDBT 2024，起始页 488，DOI 10.48786/edbt.2024.42（DataCite）；arXiv:2401.10518。
- **设置**：目标区域完全没有观测，相邻区域有观测；交通数据用过去 2 小时预测未来 2 小时，空气质量 24 小时→24 小时。
- **方法与外部信息**（§4.1）：选择性遮蔽时按子图相似度挑选与目标区域相似的子图，子图表示由 OpenStreetMap 的 POI 计数（26 类，表 1）、建筑层数、公园面积与最近道路的 4 维属性（等级、限速、是否单行、车道数）构成；再做图对比学习。
- **数据集与基线**：PEMS-Bay、PEMS-07、PEMS-08、Melbourne、AirQ；GE-GAN、IGNNK、INCREASE。
- **实验类型**：未观测比例 0.2–0.5；传感器数量；消融；超参数；环形划分；时间模块替换；距离函数。
- **与 ALFNet 的差异**：未观测单位是连续区域；静态；训练期使用外部静态特征，按 D5 应与 GenCast 的外部数据同样登记。
- **证据**：元数据 DC+PUB+S2+AX；设置 F（官方 PDF 首页、§4.1、实验设置段）；实验类型 F（arXiv HTML）；代码：论文给链接。

### C06 GenCast

- **出处**：Su 等，AAAI 40(18), 15761–15769（2026），DOI 10.1609/aaai.v40i18.38607；arXiv:2508.08947v2。
- **设置**：大片连续区域没有传感器；训练时在已观测区域上遮蔽以模拟；未来 2 小时；使用 ERA5 天气、路网属性与 POI。
- **数据集与基线**：PEMS-Bay、PEMS07、PEMS08、METR-LA、Melbourne，另有 NREL 太阳能；GE-GAN、IGNNK、INCREASE、KITS、STSM。
- **实验类型**：未观测比例 0.2–0.8；水平、垂直、环形划分；消融；超参数；跨领域；配对 t 检验与 Wilcoxon 检验。
- **代码**：`suzy0223/GenCast` 跳转到 `suzy0223/Generalising-Traffic-Forecasting-to-Unobserved-Regions-GenCast`（网页 301 → 200），v1 中"二者关系未核实"已解决。
- **证据**：元数据 CR+S2+AX；设置与实验 F（arXiv HTML 前 10 万字符）；代码：仓库页可访问。

### C07 SC-SAMG

- **出处**：Wang 等，Sensors 26(15), 4995（2026），DOI 10.3390/s26154995。
- **设置**（摘要）：预训练的网络级模型需要预测训练时不存在的传感器位置；用 1–7 天目标观测作支持期，一阶元学习适应；15–60 分钟；"防泄漏的保留节点协议"；对照微调的 GRU、AGCRN、DCRNN。
- **与 ALFNet 的差异**：需要目标节点的支持期标签与参数适应；其 "support" 指适应用的目标样本。窗口内观测变化与观测状态条件化：摘要未报告。
- **证据**：元数据 CR+S2；设置 A（OpenAlex 摘要）；全文未读；代码未检索到。

### C08 SLPF

- **出处**：Liu 等，TMLR 2025；arXiv:2408.02689v3（2026-09-30），v3 正文所载 OpenReview `Ff08aPjVjD`。
- **设置**：永久传感器位置与未感知位置；训练期未感知位置上的临时移动传感器提供真值；推理期只有永久传感器；1 小时→8 小时。
- **数据集**：METR-LA、PEMS-BAY、PEMS03、PEMS04、PEMS08。
- **实验类型**：感知比例；消融；噪声稳健性；效率；位置选择；预测长度；参数敏感性。
- **本项目状态**：单阶段预测主干适配（`paper/SLPF_STBP_INTEGRITY_20260926.md`），不能代表完整三阶段方法。
- **证据**：元数据 AX+S2+mlanthology；设置与实验 F（arXiv HTML）；代码：本项目已适配运行。

### C09 GinAR 与 C10 GinAR+

- **出处**：GinAR，Yu 等，KDD 2024, 3989–4000，DOI 10.1145/3637528.3672055，arXiv:2405.11333v1；GinAR+，Yu 等，IEEE TKDE 37(8), 4635–4648（2025），DOI 10.1109/TKDE.2025.3569649。
- **GinAR 设置**：部分变量在整个输入窗口内完全缺失；评价时缺失变量固定；缺失变量的未来值也作为预测目标；12→12；缺失率 25%–90%。
- **GinAR 数据集与基线**：METR-LA、PEMS-BAY、PEMS04、PEMS08、China AQI；11 个基线（含 BiTGraph、MegaCRN、STID 与多个插补 + 预测流水线）。
- **GinAR 实验类型**：缺失率扫描；消融；插值注意力与插补方法对比；超参数；5 个种子；可视化。
- **GinAR+**（摘要）：以不同缺失率的数据构成正负对，用对比学习训练 GinAR，使一个模型适用于多种缺失率。
- **代码**：GinAR 论文链接 `ChengqingYu/GinAR` 跳转到 `GestaltCogTeam/GinAR`（301 → 200），v1 中"二者关系未核实"已解决；GinAR+ 未定位独立实现，不能以 GinAR 代码代替。
- **证据**：GinAR 元数据 CR+S2+AX，设置与实验 F；GinAR+ 元数据 CR+S2，设置 A。

### C11 Merlin

- **出处**：Yu 等，KDD 2025 V.2, 3633–3644，DOI 10.1145/3711896.3737046；arXiv:2506.12459v1。
- **设置**：缺失率不固定的预测；点级随机遮蔽；直推；12→12；教师在完整观测上训练，学生在不完整数据上训练（离线蒸馏 + 多视图对比）。
- **数据集与基线**：METR-LA、PEMS04、China AQI、Global Wind；单阶段与两阶段方法若干。
- **实验类型**：缺失率扫描；消融；超参数；随时间变化的缺失率；效率；5 个种子；跨主干。
- **本项目适配**：教师只用训练节点 K 内被合成掩码隐藏的自然观测，现有审计未发现使用 V/T 真值；保留了节点身份参数（另 5 个基线改为共享结构编码）；训练预算改为每阶段 100 轮（`paper/BASELINE_FAIRNESS_20260928.md`）。
- **证据**：元数据 CR+S2+AX；设置与实验 F；代码：论文给 Zenodo，GitHub 页可访问，本项目已适配运行。

### C12 HD-TTS

- **出处**：Marisca、Alippi、Bianchi，ICML 2024，PMLR 235:34846–34865；arXiv:2402.10634。
- **设置**：输入有缺失的图时序预测；点缺失、时间块缺失、时空块缺失；直推。
- **交通实验**（附录 F、表 6）：METR-LA、PEMS-BAY，时间块与时空块两种缺失模式，结果为均值±标准差；对比 GRU-D、GRU-I、GRIN-P、GRU、DCRNN、AGCRN、GWNet 与 T&S/TTS 各变体。
- **官方实现的信息预算**（静态阅读，未运行）：`config/dataset/la.yaml` 为 window 24、horizon 12、stride 1，验证 0.1、测试 0.2；`experiments/run_realworld.py` 第 168–178 行在构造滑窗之前，对整条已掩码序列执行 `ffill().bfill()`，剩余缺失补 0。因此输入中的填补值可能来自窗口之外，后向填补还可能用到预测起点之后的读数。只把 window 改成 12 并不能满足 12 步输入预算，适配时需要改为窗口内填补；模型构造接收 `n_nodes`（第 223 行），是否含节点特定参数待查模型代码。
- **其他实验类型**：消融；吞吐与显存；解码器注意力可视化。
- **与 ALFNet 的差异**："依缺失模式调整计算"与 ALFNet 接近，但为直推，没有节点归纳、观测角色与状态进度。
- **证据**：元数据 PUB+S2+AX；设置与实验 F（arXiv v2 附录 F、v3 正文）；代码：关键实现已检查（配置与数据代码）。

### C13 MAS-Mamba

- **出处**：Huang、Chang，IET Intelligent Transport Systems 20, e70304（2026），DOI 10.1049/itr2.70304。
- **方法**（摘要）：选择性状态空间模型的离散化步长以观测掩码和距上次有效读数的时间为条件；三分辨率分支；分裂共形预测区间；METR-LA、PEMS-BAY，受控 MCAR 与 NMAR；对照紧凑模型与 DCRNN、Graph WaveNet。
- **与 ALFNet 的关系**："距上次有效读数的时间"与 Fault 中的状态进度同类，是机制层面的近邻；保留节点与新增接入：摘要未报告。
- **证据**：元数据 CR+S2；设置 A；代码未检索。

### C14 GCN-M

- **出处**：Zuo 等，DMKD 37, 913–947（2023；2022-12-16 在线），DOI 10.1007/s10618-022-00903-7；arXiv:2212.06419。
- **设置**（摘要）：时间上随机或连续缺失、空间上单传感器或多传感器缺失的联合预测。
- **本项目状态**：多尺度历史记忆需要窗外历史，此前判定为与 12 步输入预算冲突（`docs/ai/PROJECT_CONTEXT.md`）；按新的 S4，这属于"结构必需的长记忆"，需要单列扩展预算，而不是直接排除。
- **证据**：元数据 CR+S2；设置 A + 旧文件；代码：论文给链接。

### C15 KITS

- **出处**：Xu 等，AAAI 39(12), 12945–12953（2025），DOI 10.1609/aaai.v39i12.33412；arXiv:2311.02565v2。
- **设置**：归纳克里金（同时刻估计）；指出训练图比推理图稀疏的"图差距"，在训练图中加入虚拟节点。
- **数据集与基线**：8 个数据集（含 METR-LA、PEMS-BAY、PEMS07）；Mean、OKriging、KNN、KCN、IGNNK、LSJSTN、INCREASE 等。
- **实验类型**：缺失比例 20%–80%；随机、由细到粗、区域三种模式；5 个种子；消融；超参数；迁移。
- **证据**：元数据 CR+S2+AX；设置与实验 F；代码：论文给链接。

### C16 UniSTOK（两个版本分别引用）

- **v1**（arXiv:2603.05301v1，2026-03-05）：题名 "UniSTOK: Uniform Inductive Spatio-Temporal Kriging"；作者 Lewei Xie、Haoyu Zhang、Juan Yuan、Liangjun You、Yulong Chen、Yifan Zhang。方法：原始观测与只在缺失处合成的拼图增强两路共享主干；缺失掩码调制模块把二值掩码编码为仿射参数 (α, β)；双通道注意力融合。数据集 METR-LA、PEMS-BAY、NREL-AL；随机、块、混合缺失；均值±标准差。
- **v2**（arXiv:2603.05301v2，2026-05-09）：题名 "Uniform Inductive Spatio-Temporal Kriging"；作者 Lewei Xie、Haoyu Zhang、Yulong Chen、Liangjun You、Zongxian Yang、Yifan Zhang；摘要中的方法为可靠性引导信号调节（RSR）与残差偏差校准（RBC），v2 摘要未提及掩码仿射调制与拼图增强。
- **用途**：讨论"由掩码生成仿射调制"时只能引用 v1；讨论该工作当前形态时引用 v2。两版都是预印本，同时刻估计，不作预测基线。
- **证据**：元数据 AX（v1 另有 S2）；v1 设置 F（arXiv HTML 前 10 万字符），v2 设置 A（摘要）；代码未提供。

### C17 TESTAM

- **出处**：Lee、Ko，ICLR 2024；arXiv:2403.02600。
- **方法**（摘要）：时间专家（恒等矩阵）、静态图专家、动态图专家三类空间建模；路由改写为带伪标签的分类问题，按交通状况选择空间建模方式。
- **数据集**：METR-LA、PEMS-BAY、EXPY-TKY。
- **与 ALFNet 的关系**：在恒等与图传播算子之间路由，是 ALFGraph 的机制近邻；其路由依据为交通状况，而 ALFGraph 依据观测状态。节点归纳与缺失：摘要未报告。
- **证据**：元数据 AX+S2；设置 A；代码：摘要页给链接。

### C18 EvoXXLTraffic

- **出处**：Yin 等，arXiv:2605.29768v2（2026）；前作 XXLTraffic 见 E43。
- **内容**：九个 PeMS 辖区、最长 27 年；逐年活跃传感器、年度流量矩阵与图快照；标出新增（冷启动）、停用与持续存在的传感器。三种协议：间隔预测、小时/日预测、传感器演化的流式任务。
- **结论**：新传感器的表现是主要瓶颈；假设每步新增节点比例很小的持续学习方法在大幅年度变化下失效。
- **证据**：元数据 AX+S2；设置 F（arXiv HTML 前 10 万字符）；代码：论文给链接。

### C19 FRIGATE

- **出处**：Gupta、Kodamana、Ranu，KDD 2023, 649–660，DOI 10.1145/3580305.3599357；arXiv:2306.08277v1。
- **问题设置**（摘要、实验设置）：路网上只有部分节点装有传感器，已装传感器的历史也可能缺失，路网连接可能变化。从全部节点中均匀随机选出"已见"节点（默认比例：Beijing 5%、Chengdu 50%、Harbin 30%），在未见节点上做推理与评价。§4.2 写明常规实验"用一小时历史预测一小时"，Δ＝12（5 分钟采样）。
- **训练监督**（式 (1)）：损失只在有真值的训练节点集合 $\mathcal{V}^{tr}_{t+k}$ 上计算。
- **输入**：Lipschitz 嵌入（到 16 个随机锚点的最短路径距离，只用拓扑）；节点邻域在窗口 $[t-\Delta,t]$ 内已观测读数的统计矩（算法 1 第 7 行）；传感器读数。统计矩的标准化是否用到训练期统计，待核。
- **数据集**：Beijing（28,465 节点）、Chengdu（3,193）、Harbin（6,235），由出租车 GPS 轨迹聚合为 5 分钟速度；时间 70/20/10。
- **基线**：DCRNN、STGCN、STNN、LocaleGN、Graph WaveNet（改造）、STGODE（改造）；改造时去掉逐节点嵌入以支持归纳设置。
- **实验类型**：MAE、sMAPE、RMSE 带 95% 置信区间（表 3）；已见节点比例 10%–90%（图 6）；消融（门控、Lipschitz 嵌入、入出聚合、统计矩，图 7a）；GNN 层数（图 7b）；冻结参数下随机删改最多 10% 的边（图 7c，§4.6）；推理时随机丢弃 33.33% 的历史快照、不重训（§4.6、表 4）；参数量（表 2b）。
- **与 ALFNet 的差异**：未见节点集合固定，不存在窗口内由不可读到可读的转换；不规则性是对所有节点随机丢弃快照，不区分角色；没有显式掩码或角色条件；数据集不是 PEMS/METR，节点规模大得多。它与 MoGERNN 共同构成"节点归纳 + 冻结参数应对网络变化"的直接近邻。
- **证据**：元数据 CR+AX；设置与实验 F（arXiv v1 HTML：式 (1)、算法 1、实验设置、§4.6、表 2–4、图 6–7，经网页工具逐字引文核对）；代码：仓库页可访问。

### C20 SCPT

- **出处**：Prabowo 等，DMKD 38, 913–937（2024；2023-09-27 在线），DOI 10.1007/s10618-023-00982-0；arXiv:2305.05237（v4）。
- **设置**（§3.2）：道路按约 70/10/20 分为训练、验证、测试三组，时间按 7/1/2 分三段，构成 A–I 九个集合；测试在集合 I（测试道路 × 测试时段）上进行，编码器与预测模型都未见过这些道路。推理时空间编码器读取新道路两天的数据（G∪H），不重训。
- **信息暴露**（§3.6）：周期成分在训练集合 A∪D∪G 上做逐路 DCT 得到，其中 D、G 分别是验证道路与测试道路的训练时段数据，即周期分解使用了主干训练未见道路的信号。
- **方法**：空间对比预训练编码器，与 Graph WaveNet 主干结合；预测 12 步。
- **数据集与基线**：METR-LA、PeMS-BAY、PeMS-D7(m)、PeMS-11k(s)；基线为 Graph WaveNet。
- **实验类型**：10 次不同初始化重复；消融；训练开销；空间划分随机性（36 种组合）；分预测步长。
- **与 ALFNet 的差异**：推理时需要目标道路两天历史，周期分解读取目标道路训练时段的信号，二者都超出 ALFNet 的 12 步窗口预算与训练信号边界；不是自动相容的基线，但它是"新增位置冷启动"与"输入预算"讨论的重要近邻。
- **证据**：元数据 CR+AX；设置 F（arXiv §3.2、§3.6，经逐字引文核对，曾出现网页工具转述错误）；实验类型 S（网页工具转述，未逐表核对）；代码：仓库页可访问。

### C21 STOP

- **出处**：Ma 等，ICML 2025，PMLR 267:42165–42192。
- **设置**（§4.1、附录 E.2）：时间偏移——用第一年数据训练，在后续各年测试；结构偏移——选约 75% 节点训练与验证，测试时随机遮蔽 10% 节点模拟消失，并加入 30% 新节点；PEMSD3-Stream 直接使用其自然扩张。推理时不更新参数；部分基线去掉节点嵌入或自适应图以适配（附录 E.1）。
- **方法**：节点只与上下文感知单元交换信息（集中式消息），配合消息扰动与分布鲁棒优化。
- **数据集与基线**：LargeST-SD/GBA/GLA/CA、PEMSD3-Stream、KnowAir；14 个基线，含 STGCN、GWNet、STID、STAEformer、D2STGNN、BigST、CaST、STONE、TrafficStream、PECPM、TFMoE。
- **实验类型**：分布外泛化；消息机制对照；超参数；消融；效率；新节点上的归纳性能（§4.7 表 4、附录表 7）。输入与预测长度在所读部分未定位。
- **与 ALFNet 的差异**：新节点在测试时有输入历史，不涉及零历史与窗口内转换；它代表"推理不更新参数的结构分布外"这一设置。
- **证据**：元数据 PUB（PMLR 页面与 PDF）；设置 F（PDF 文字提取，`verification_20261103/stop.txt`）；代码：仓库页可访问。

### C22 FUNS

- **出处**：Roth、Liebig，2022 IEEE ICDM Workshops (ICDMW), 740–747，DOI 10.1109/ICDMW58026.2022.00101；arXiv:2211.11596。注意是 workshop，不是 ICDM 主会。
- **设置**（摘要）：只在已观测位置的数据上优化模型，预测完全没有传感器的位置的未来状态。
- **方法**（据 MoGERNN §5.1 的转述）：MPNN 框架 + GRU，消息聚合采用 GATv2 变体。
- **数据集、基线与实验类型**：全文未读，待核。
- **与 ALFNet 的差异**：对应 Unseen 一类；是"未观测节点未来预测"最早的直接脉络之一，MoGERNN 已把它作为基线。
- **证据**：元数据 CR+AX；设置 A（OpenAlex 摘要）；代码：仓库页可访问（`roth-andreas/funs`）。

---

## 5. 扩展条目

完整题名、作者与核实路径见附表；本节只列用途。"关系"一列：**同协议候选**、**相关工作**、**机制来源**、**背景**。

### 5.1 类别一：归纳克里金、外推与状态估计（任务边界）

| 编号 | 工作 | 出处 | 要点 | 关系 | 代码 | 内容证据 |
|---|---|---|---|---|---|---|
| E01 | IGNNK | AAAI 2021 | 子图采样学习可迁移消息传递，未采样位置插值 | 相关工作；可改造为预测（MoGERNN 已有先例） | 未检索 | A |
| E02 | INCREASE | WWW 2023 | 异质空间关系的归纳克里金，新位置无需重训 | 相关工作；可改造为预测 | 未检索 | A |
| E03 | STGNP | KDD 2023 | 时空图神经过程外推，带不确定性 | 相关工作 | 检索结果给链接 | A |
| E04 | LSJSTN | TNNLS 2024 | 解耦长短期模式的非传感器位置推断 | 背景 | 未检索 | A |
| E05 | IAGCN | KBS 2024 | 归纳自适应图卷积的交通克里金 | 背景 | 未检索 | 出版记录 |
| E06 | AnchorGK | KDD 2026 | 按特征可用性构造锚点并分层 | 相关工作 | 未检索 | A |
| E07 | SATCN | arXiv | 空间聚合与时间卷积的实时克里金 | 背景 | 仓库页可访问（`Kaimaoge/SATCN`） | 出版记录 |
| E08 | 物理引导增量训练克里金 | arXiv | KITS 增量训练用于空气质量 | 背景 | 未检索 | S |
| E09 | 混合归纳—直推插补 | arXiv | 未采样位置的交通流插补 | 背景 | 未检索 | S |
| E11 | DGAE | TITS 2025 | 稀疏传感器下的网络级状态估计（同时刻） | 背景 | 未检索 | A |
| E12 | SUSTeR | SIGSPATIAL 2023 | 时空都稀疏的观测下的交通预测 | 背景 | 仓库页可访问（`ywoelker/SUSTeR`） | A |
| E61 | STA-GANN | CIKM 2025 | 归纳克里金：相位解耦、动态元数据图与对抗迁移 | 相关工作（克里金） | 未检索 | A |
| E62 | RADGCN | Comm. Transp. Res. 2026 | 在已观测节点上学习功能关系，经双边扩散核传播到全网，用于状态估计 | 背景（状态估计） | 未检索 | A |
| E64 | GenST | arXiv 2026（v2） | 把未观测节点的未来预测改写为条件生成；用微调的 LLM 从节点描述提取语义 | 相关工作；需额外语义与 LLM 资源，按 S4 单列 | 未检索 | A |

### 5.2 类别二：不完整观测下的预测与插补来源

| 编号 | 工作 | 出处 | 要点 | 关系 | 代码 | 内容证据 |
|---|---|---|---|---|---|---|
| E13 | GRIN | ICLR 2022 | 图循环插补 | 机制来源（SAHC） | 本地 `references/grin/` | 旧文件 S+R |
| E14 | PriSTI | ICDE 2023 | 条件扩散时空插补 | 相关工作 | 旧文件给链接 | 旧文件 A+R |
| E15 | SPIN | NeurIPS 2022 | 稀疏观测下的时空图缺失重建 | 相关工作 | 未检索 | S |
| E16 | ImputeFormer | KDD 2024 | 低秩归纳偏置的可泛化时空插补 | 相关工作；可作流水线插补器候选 | 未检索 | A |
| E17 | STAMImputer | IJCAI 2025 | 面向块缺失的注意力专家混合插补 | 相关工作 | 未检索 | A |
| E18 | FENCE | AAAI 2026 | 按节点缺失程度调节扩散引导尺度 | 背景 | 未检索 | A |
| E19 | BiTGraph | ICLR 2024 | 时间卷积与图运算显式处理缺失的预测 | 同协议候选（类别二） | 仓库页可访问（`chenxiaodanhit/BiTGraph`） | 出版记录 |
| E20 | S4M | ICLR 2025 | 缺失感知双流 S4 | 相关工作（非图） | 摘要页给链接 | A |
| E21 | GeoMAE | Neural Networks 2026 | 掩码表示学习的缺失值时空图预测 | 相关工作 | 未检索 | S |
| E22 | DCDM | TMC 2026 | 带物理先验的扩散框架，统一插补与预测 | 相关工作 | 未定位 | A |
| E23 | IBN | arXiv 2025 | 变量缺失下的可解释双向建模预测 | 背景 | 摘要页给链接 | A |
| E24 | GRU-D | Sci. Rep. 2018 | 掩码与距上次观测时间进入循环网络并做衰减 | 机制近邻（状态进度） | 未检索 | 出版记录 |
| E25 | 信息中断建模 | arXiv 2026 | MNAR 假设下的检测器与通信故障 | 背景 | 未检索 | S |
| E58 | GSTAE | TITS 2023 | 插补与预测作为并行任务、顺序训练的图时空自编码器，避免先插补再预测的误差累积 | 相关工作；与"共享补全"对照 | 未检索 | A |
| E59 | RT-GCN | Information Fusion 2024 | 高斯分布节点表示与方差注意力，应对噪声与缺失 | 相关工作 | 未检索 | A |
| E60 | GMN/SGMN | TRC 2020 | 图马尔可夫网络联合缺失状态推断与预测 | 相关工作（经典） | 旧文件给链接 | 题名与旧文件 |

### 5.3 类别三：冷启动迁移、持续学习、分布外与测试期适应

| 编号 | 工作 | 出处 | 要点 | 关系 | 代码 | 内容证据 |
|---|---|---|---|---|---|---|
| E26 | Taming Local Effects | NeurIPS 2023 | 全局共享与节点特定参数的取舍；节点嵌入可迁移到新节点集 | 相关工作；支撑"基线节点身份参数处理"的说明 | 检索结果给链接 | A |
| E27 | STGP | CIKM 2024 | 提示式时空图迁移，覆盖预测、克里金、外推 | 相关工作 | 未检索 | A |
| E28 | GPD | ICLR 2024 | 扩散生成网络参数做少样本学习 | 背景 | 未检索 | 出版记录 |
| E29 | TPB | CIKM 2023 | 跨城市少样本预测，模式库 | 背景 | 未检索 | A |
| E30 | UniST | KDD 2024 | 提示驱动的通用城市时空模型 | 背景 | 未检索 | A |
| E31 | OpenCity | ACM TIST 2025 | 交通基础模型，零样本 | 背景 | 仓库页可访问（`HKUDS/OpenCity`） | A |
| E32 | FlashST | ICML 2024 | 交通预测提示微调 | 背景 | 未检索 | 出版记录 |
| E33 | TrafficStream | IJCAI 2021 | 扩张路网上的流式预测与持续学习 | 相关工作 | 未检索 | A |
| E34 | PECPM | KDD 2023 | 扩张路网上的模式扩展与巩固 | 相关工作 | 未检索 | A |
| E35 | STKEC | TITS 2023 | 扩张图上的知识扩展与巩固 | 相关工作 | 未检索 | A |
| E36 | URCL | ICDE 2024 | 回放式持续学习 | 背景 | 未检索 | A |
| E37 | TFMoE | arXiv 2024 | 按同质组分配专家的持续预测 | 背景 | 未检索 | S |
| E38 | EAC | ICLR 2025 | 扩展与压缩提示池 | 相关工作 | 未检索 | A |
| E39 | STBP | ICLR 2026 | 固定主干 + 可扩展模式库 | 已在表 1（离线适配） | 本项目已适配运行 | A+R |
| E40 | STONE | KDD 2024 | 时空分布外；验证与测试期节点增加 5/10/15% 或减少 5% | 相关工作；需与 STOP 一起核对新增节点的输入历史与更新流程 | 未检索 | A+S |
| E41 | 逐年空间偏移稳健预测 | arXiv 2024 | 学习一组图生成器并按新环境组合 | 背景 | 未检索 | S |
| E42 | STRAP | NeurIPS 2025 | 时空模式库检索并以提示注入；流式设置，原文有"由预训练参数初始化后再微调"的融合步骤，测试期是否更新待全文核实 | 相关工作（边界对照） | 未检索 | S |
| E43 | XXLTraffic | SIGSPATIAL 2025 | 23 年、9 个区域、节点逐年扩张的数据集 | 背景（真实传感器演化数据） | 论文给链接 | A |
| E63 | ST-TTC | NeurIPS 2025 | 测试期计算：频谱域校准器与流式记忆队列，实时修正周期性偏差；是否使用已到达的真值待全文核实 | 相关工作（测试期适应的边界） | 未检索 | A |
| E65 | Wang & Yang | NeurIPS 2026（Poster） | 在冻结检查点上评价 8 种时空图网络对传感器丢失、跨城市、工作日与周末的敏感性；会议页面摘要称丢失敏感性与路由集中度相关 | 相关工作；与实验缺口 G9、G10 直接相关；全文未读，结论只按会议页面摘要转述 | 会议页面称发布了检查点与代码，未检索 | S（会议页面摘要） |

### 5.4 类别四：专家混合、条件化与图传播

| 编号 | 工作 | 出处 | 要点 | 关系 | 代码 | 内容证据 |
|---|---|---|---|---|---|---|
| E44 | GMoE | NeurIPS 2023 | 节点自适应选择不同跳数的聚合专家 | 机制来源（ALFGraph） | 未检索 | S |
| E45 | Flex-MoE | NeurIPS 2024 | 按可用模态组合组织专家，含缺失模态库 | 机制近邻（按可用性路由，跨领域） | 未检索 | S |
| E46 | FiLM | AAAI 2018 | 特征级线性调制 | 机制来源（ALFTemporal） | — | 出版记录 |
| E47 | MegaCRN | AAAI 2023 | 元图学习器 + 元节点库 | 同协议候选（条件化图，次要） | 未检索 | A |
| E48 | DGCRN | TKDD 2023 | 动态图卷积循环网络与公平比较基准 | 背景 | 未检索 | A |

### 5.5 空间依赖、空间分区与大规模主干

| 编号 | 工作 | 出处 | 要点 | 关系 | 代码 | 内容证据 |
|---|---|---|---|---|---|---|
| E49 | PatchSTG | KDD 2025 | KD-tree 不规则空间分块，块内与块间注意力 | 已在表 1；机制来源（ALFTemporal 分区） | 本项目已适配运行 | A+R |
| E50 | LargeST | NeurIPS 2023（数据集与基准） | 大规模交通预测基准 | 背景 | 未检索 | 出版记录 |
| E51 | BigST | PVLDB 2024 | 线性复杂度时空图网络 | 背景 | 未检索 | A |
| E52 | STAEformer | CIKM 2023 | 时空自适应嵌入 + Transformer | 同协议候选（通用主干，次要） | 旧文件给链接 | A+R |
| E53 | PDFormer | AAAI 2023 | 传播时延感知的动态长程 Transformer | 相关工作 | 未检索 | A |
| E54 | AdaFre | AAAI 2026 | 时变频率路径选择 | 相关工作 | 未检索 | A |
| E55 | STID | CIKM 2022 | 时空身份嵌入 + MLP | 已在表 1 | 本项目已适配运行 | 出版记录 |
| E56 | Graph WaveNet | IJCAI 2019 | 扩张时间卷积 + 自适应邻接 | 同协议候选（S9） | 本项目有适配实现与旧协议结果 | 出版记录 |
| E57 | AGCRN | NeurIPS 2020 | 节点自适应图卷积循环网络 | 同协议候选（S9） | 旧文件给链接 | 出版记录 |
| E66 | STGODE | KDD 2021 | 图常微分方程时空预测 | 本项目有旧协议结果（语义图改为拓扑相似度） | 本项目已适配 | 出版记录 |
| E67 | STGCN | IJCAI 2018 | 图卷积 + 时间卷积 | 已在表 1 | 本项目已适配运行 | 出版记录 |

---

## 6. 实验类型覆盖对照

符号：✓ 做了；部分；— 在所读范围内未见。ALFNet 一列依据 `drafts/manuscript_v6_cn.md` 第 4 节。各工作的读取范围见第 4 节卡片的"证据"一行。

| 实验类型 | ALFNet | MoGERNN | SIML | FRIGATE | UIGNN | ST-FiT | SCPT | STSM | GenCast | HD-TTS | Merlin | GinAR | KITS | STOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 多次运行或置信区间、显著性检验 | —（种子 32） | — | ✓（跨掩码标准差） | ✓（95% 置信区间） | — | ✓（3 种子） | ✓（10 次重复） | — | ✓（t 检验、Wilcoxon） | ✓ | ✓（5 种子） | ✓（5 种子） | ✓（5 种子） | — |
| T2 保留或观测比例扫描 | — | ✓ | ✓ | ✓ | — | ✓ | 部分（划分随机性） | ✓ | ✓ | — | — | — | ✓ | — |
| T3 网络级缺失率或缺失模式扫描 | 部分（单节点窗口内数量与布局） | — | 部分（跨掩码） | 部分（快照丢弃） | — | — | — | — | — | ✓ | ✓ | ✓ | ✓ | — |
| T4 区域型或空间成片缺失 | — | — | — | — | — | — | — | ✓ | ✓ | ✓（时空块） | — | — | ✓ | — |
| T5 效率 | 部分（自身参数量） | ✓ | ✓ | ✓（参数量） | — | ✓ | ✓ | — | — | ✓ | ✓ | — | — | ✓ |
| T6 分预测步长 | — | ✓ | — | — | ✓ | ✓ | ✓ | — | — | — | — | — | — | — |
| T7 与重训或微调对照 | — | ✓ | ✓ | — | — | ✓ | — | — | — | — | — | — | — | — |
| T8 简单或两阶段参照 | — | ✓ | — | — | ✓ | ✓ | — | — | — | ✓ | ✓ | ✓ | ✓ | — |
| T9 消融 | ✓（4 项） | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T10 超参数 | ✓ | ✓ | — | ✓ | — | ✓ | — | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| T11 机制可视化 | — | ✓ | ✓ | — | — | — | — | — | — | ✓ | — | ✓ | — | — |
| T12 跨领域或大规模数据 | — | — | — | ✓（最多 28k 节点） | — | — | ✓（11k 节点） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T13 不确定性 | — | — | — | — | ✓ | — | — | — | — | — | — | — | — | — |
| T14 拓扑扰动、噪声或状态信息误差的稳健性 | — | ✓（结构变化） | — | ✓（删改边、丢快照） | — | — | — | — | — | — | — | — | — | ✓（结构偏移） |

SCPT、STOP 两列中，SCPT 的实验类型为网页工具转述，STOP 只读了 PDF 文字提取中的相关段落；"—"的含义同上。

ALFNet 已有而上表少见的实验：按事件持续步数的阶段评价；固定预测目标与非干预输入的配对观测干预；固定可见数量下的时间布局比较；同参数规模的局部上下文替代消融；验证与测试两套机制结果的跨划分核验。

---

## 7. ST-FiT 与 Merlin

按作者要求，本节给出具体建议，不再作为选择题。

### 7.1 已核实的事实

1. 二者已在 2026-10-01 统一评估中训练并测试：计划 `configs/paper/all_final_test_20261001/plan.json`；`paper/all_final_test_20261001/verification.json` 记录 117 次评估、`passed: true`，即测试前通过了完整验证复现。测试格在 `paper/all_final_test_20261001/validation_selected_test_cells.csv`。
2. 工作区中未跟踪的 `paper/comparison_ten_baselines_20261102/`（README 写明为离线合并已保存结果，未训练、未推理）已给出加入二者后的四数据集对比表：LifeAvg 三项仍全部第一；33 格第一数由 33/29/29/30 变为 30/22/15/19；PEMS08 的 MAE 降幅下限由 7.86% 变为 7.27%。本轮未重算这些数字。
3. 二者的权重已在 2026-10-17 清理中删除（`archive/cleanup_20261017/plan.json` 的删除记录中，STFiT 出现 62 次，Merlin 出现 229 次）。若要在二者上补做图 3、图 4、图 6 一类分析，需要重新训练。
4. 作者 2026-10-20 以"当时训练结果无效"移除二者；`docs/ai/` 下未找到具体的技术说明。
5. 已记录的适配差异（`paper/BASELINE_FAIRNESS_20260928.md`）：Merlin 教师的训练视图、保留节点身份参数、训练预算改动；ST-FiT 的学习率调度改为恒定、评估含 Gumbel 随机采样（固定种子重放）。

### 7.2 建议

- **文献地位**：ST-FiT 归类别一（训练无目标信号、部署有历史的节点归纳）；Merlin 归类别二（缺失率不固定的预测，训练期教师蒸馏）。二者都应进入相关工作，阶段 4 写入。
- **同协议候选资格**：按 `BASELINE_CRITERIA_CN.md`，ST-FiT 满足 S1–S6；Merlin 满足 S1–S6，但教师训练视图须在信息账本中登记并在正文披露。二者都属于"应当比较"的范围。
- **已有结果复用资格**：**待审计**。"有结果文件"不证明有效，"有适配差异"也不证明无效。建议先做一次只读审计，逐项判断：（a）作者所说"无效"的具体原因；（b）Merlin 的节点身份参数政策与其他 5 个基线不一致，是否违反 S2 或需要统一；（c）Merlin 两阶段训练预算与其他模型的差异是否影响可比性；（d）ST-FiT 评估随机性的重放是否与报告一致；（e）2026-10-01 计划与当前最终 ALFNet 配置的对应关系（PEMS08 权重衰减已于 2026-10-12 更新）。
- **审计之后**：通过则以现有结果进入比较（改正文需作者另行授权）；不通过则把修复后重训列为建议实验。审计完成之前，正文不宜写"未比较"或给出不比较的理由。
- 本轮未恢复任何表格或数值，也没有因二者影响 ALFNet 的第一数而调整标准。

---

## 8. 综合

### 8.1 分层近邻（精读与论证优先序，不是性能排序，也不是必跑清单）

| 层 | 工作 | 进入该层的理由 | 关键边界 |
|---|---|---|---|
| 第一层：问题设定直接近邻（训练期无某些位置的信号，预测其未来，推理不更新参数） | C01 MoGERNN | 同一问题设定与传感器命名；图专家混合；推理不重训应对传感器增减 | 传感器类型按配置定义；路由依据节点特征 |
| | C19 FRIGATE | 部分感知、无历史节点、不规则观测、冻结参数应对拓扑变化 | 未见节点集合固定；无角色条件；非 PEMS 数据 |
| | C22 FUNS | 未观测节点未来预测的早期直接脉络 | 全文未读 |
| | C03 UIGNN | 无历史位置未来预测，带不确定性 | 只对应 Unseen |
| | C04 ST-FiT | 训练无目标信号、部署有历史 | 源实验只对应 Ready 一类条件 |
| 第二层：同一问题的变体（区域外推、额外信息或受限信号边界） | C02 SIML-MSIGN | 固定路网、变化掩码、掩码门控 | 训练监督未观测节点；Step-0 与 Step-5 分开；历史 48 步 |
| | C05 STSM、C06 GenCast | 无观测区域的未来预测 | 区域型；外部静态特征与天气 |
| | C20 SCPT | 新道路预测，不重训 | 两天目标历史；周期分解读取目标道路训练时段信号 |
| | C08 SLPF | 部分感知长期预测 | 训练期使用未感知位置真值 |
| 第三层：同节点集合上的缺失鲁棒预测与观测条件化 | C12 HD-TTS | 以缺失模式为条件组织计算；有交通实验 | 直推；官方数据管线的窗外填补 |
| | C11 Merlin、C09 GinAR、C10 GinAR+ | 缺失率变化下的端到端预测，交通数据相同 | 直推 |
| | C14 GCN-M、E58 GSTAE、E60 GMN、C13 MAS-Mamba | 联合插补与预测、状态推断、观测间隔条件化 | MAS-Mamba 仅摘要级；GCN-M 需长记忆 |
| 第四层：设置不同的边界对照（目标支持期或在线更新） | C07 SC-SAMG、SIML Step-5、E33–E39、E42 STRAP、E63 ST-TTC；C21 STOP、E40 STONE（推理不更新但新节点有历史） | 传感器网络变化下的适应方式 | 需要目标数据或更新参数，或不涉及零历史 |
| 机制边界（不是任务近邻） | C17 TESTAM、E44 GMoE、E45 Flex-MoE、C16 UniSTOK v1、E24 GRU-D、E46 FiLM、E49 PatchSTG | 与 ALFNet 的路由、调制、分区机制形式相近 | 用于说明机制差异，不作为任务基线 |

### 8.2 差异的三种性质

撤回 v1 中"未见同时满足四点的工作"的合取式结论。分三类说明：

- **问题覆盖**：已有工作分别覆盖了训练期无信号节点的未来预测（FUNS、UIGNN、MoGERNN、FRIGATE、ST-FiT）、推理不更新参数应对传感器增减或拓扑变化（MoGERNN、FRIGATE、STONE、STOP）、同一节点集合上窗口内缺失变化（HD-TTS、GinAR、Merlin、MAS-Mamba、GCN-M）。ALFNet 在同一协议中评价保留节点的接入与已知节点的失效；保留节点上先接入后失效的情形没有评价。这只说明问题覆盖的组合不同，不能单独证明方法新颖。
- **计算机制**：ALFNet 让观测状态（角色、状态进度、掩码）进入共享补全、每一步每个门的传播路由与输出修正，跨节点时间上下文经 FiLM 调制。机制近邻包括 MoGERNN、TESTAM、HD-TTS、MAS-Mamba、UniSTOK v1、GRU-D、Flex-MoE。机制差异需要由实验支撑（第 8.3 节 P2），不能只靠与文献的对比来主张。
- **评价设计**：四类场景、按事件持续步数的阶段、固定目标的配对干预与时间布局比较。在所读范围内未见按接入或失效后持续步数分阶段报告的工作；这是评价设计的特点，不作为方法有效性的证据。

### 8.3 实验缺口建议（修订；只列建议，不运行；阶段 3 再估工作量）

排序依据是"能排除哪一种替代解释、支撑哪一条主张"，不是文献中出现的频次。

| 序 | 缺口 | 对应的主张或替代解释 | 现状与依据 | 初判 |
|---|---|---|---|---|
| P1 | 比较的有效性与覆盖：先复核已有适配完整性（SLPF 单阶段、STBP 离线主干、各基线节点身份参数政策），完成 ST-FiT/Merlin 有效性审计，再按 S1–S12 决定复用、新增或修复 | "优于所列方法"的范围与可信度 | 见 `BASELINE_CRITERIA_CN.md`；不先增加模型数量 | 只读审计可先做；新增或重训需授权 |
| P2 | 观测状态的整体机制控制 | "以 OS 为条件组织计算"是收益来源，而不只是更多参数或跨节点上下文（模拟评审 R9） | 现有 4 项消融各改一处；历史 Mask-only 同时改变角色与状态进度输入以及角色相关修正倍率，不是单一入口的干净对照（`docs/ai/PROJECT_CONTEXT.md` 2026-09-22 条）。需要设计覆盖补全、路由、ALFTemporal 输入与末步修正全部入口的匹配信息控制；路由权重可视化只能作补充说明，不能替代 | 需新训练；先整理已有证据与限制 |
| P3 | 可重复性与选择稳健性 | 单次运行下的计数式结论（R2）；测试集曾参与历史配置决定（R3） | 运行种子同时决定节点划分（`lib/v4_seeds.py`：`node_split_seed == run_seed`），多种子会同时改变训练随机性与保留节点。建议分两层：固定划分换训练种子、固定种子换划分。节点或时间 bootstrap 只衡量评价样本的不确定性，且滑窗重叠、时间相关，需要块 bootstrap，不能替代重复训练。另需一次未经选择的留出确认 | 需新训练；风险最高 |
| P4 | 简单参照、流水线参照与自适应图参照 | 收益是否只是任务难度低或流水线已足够（R12）；领域标准基线缺失（R7） | 定义见 `BASELINE_CRITERIA_CN.md` S8、S9 | 简单参照不训练但需计算指标，仍需授权 |
| P5 | 观测预算与过程 | 结论对保留比例、成片失效、角色信息误差是否稳健（R8） | 保留比例固定约 25%；未评价空间成片失效与保留节点上的失效；观测角色与状态进度由协议给出 | 需新评估或新训练 |
| P6 | 适配与调参公平性 | 基线是否因调参资源不足而偏弱（R4） | 各基线训练预算不一致（`paper/BASELINE_FAIRNESS_20260928.md`） | 需新训练 |
| P7 | 分预测步长 | 收益是否集中在某些步长 | 正文只有 12 步平均；在 `paper/`、`artifacts/` 下未找到最终模型的逐样本预测。若其他位置保存了带步长维的预测，可离线汇总；否则需要对已有权重重新推理（属评估，需授权） | 视预测文件情况而定 |
| P8 | 效率 | 收益是否以显著更高的计算代价换来 | 只报自身参数量；参数量可由已有权重统计，耗时需查日志或新测 | 低到中 |
| P9 | 扩展数据 | 更大规模或真实传感器演化下是否成立 | 4 个交通数据集，最多 325 节点；EvoXXLTraffic、LargeST 可作候选 | 需新实验；优先级低 |

---

## 9. 核实记录

### 9.1 逐条路径

见 `LITERATURE_METADATA_APPENDIX_CN.md` 第 2 节。本轮新增核实的要点：

- Crossref 新解析 16 个 DOI（FUNS、FRIGATE、SCPT、GSTAE、STA-GANN、RADGCN、RT-GCN、GMN、6 个 NeurIPS `10.52202` DOI、ST-TTC 等），全部与 Codex 报告给出的题名、作者、出处一致。
- DataCite 解析 STSM 的 `10.48786/edbt.2024.42`，doi.org 跳转到官方 PDF，首页印有该 DOI 与页码 488。
- PMLR 页面核实 HD-TTS（235:34846–34865）与 STOP（267:42165–42192）。
- arXiv 页面核实 UniSTOK v1/v2、GenST v1/v2、SLPF v3（含 OpenReview ID）。
- GitHub 仓库网页状态：`roth-andreas/funs`、`idea-iitd/Frigate`、`cruiseresearchgroup/forecasting-on-new-roads`、`PoorOtterBob/STOP`、`Kaimaoge/SATCN`、`ywoelker/SUSTeR`、`chenxiaodanhit/BiTGraph`、`HKUDS/OpenCity`、`marshka/hdtts`、`GestaltCogTeam/GinAR`、`ChengqingYu/Merlin` 均为 200；`ChengqingYu/GinAR` 与 `suzy0223/GenCast` 为 301 跳转。仓库可访问不等于已运行复现。

### 9.2 检索到但未列入

| 线索 | 原因 |
|---|---|
| STGformer（arXiv:2410.00385） | Semantic Scholar 两次 429，正式会议未核实 |
| LT-GSSM（检索片段，期刊为 IJACSA） | 未核实 |
| CP-MoE（arXiv:2406.12923）、Bridge（arXiv:2605.19172）、FAST（arXiv:2604.13453）、UrbanDiT、UrbanFM、FLEXIBLE | 偏题或未核实 |
| Implicit-Explicit Diffusion、STCDM、LSCD、CSBI、CoSTI | 只见于检索摘要，纯插补，未核实 |
| Crossformer、DCRNN、GRU（Cho 等）、稀疏专家混合（Shazeer 等） | ALFNet 的设计来源，已在正文参考文献，不重复列入 |

v1 中列为未核实线索的 STA-GANN、RADGCN、ST-TTC、RT-GCN、GMN/SGMN 已核实并移入 E58–E63；NeurIPS 2026 poster 139842 以单一官方来源列入 E65。

### 9.3 对正文参考文献的元数据补充（只记录，阶段 4 再处理）

| 正文编号 | 文献 | 补充 |
|---|---|---|
| [3] | GCN-M | DMKD 37, 913–947；DOI 10.1007/s10618-022-00903-7 |
| [4] | GinAR | KDD 2024, 3989–4000；DOI 10.1145/3637528.3672055 |
| [6] | UIGNN | ICDM 2023, 1223–1228；DOI 10.1109/ICDM58522.2023.00152 |
| [7] | MoGERNN | TRC 174, 105080 |
| [8] | SLPF | TMLR 2025；OpenReview Ff08aPjVjD；作者 8 人见附表 |
| [9] | SIML-MSIGN | TRC 185, 105564；DOI 10.1016/j.trc.2026.105564 |
| [14] | PatchSTG | KDD 2025 V.1, 307–317；DOI 10.1145/3690624.3709177 |
| [17] | STBP | Aoyu Liu, Yaying Zhang；ICLR 2026 |
| [24] | SC-SAMG | Sensors 26(15), 4995；DOI 10.3390/s26154995 |

---

## 10. 局限

- 有记录的定向调研，不是 PRISMA 系统综述，没有双人独立筛选；第一轮遗漏了 FRIGATE、SCPT，说明 WebSearch 召回有限。
- 精读范围：C01、C02 读本地全文；C03–C06、C09、C11、C12、C15、C16 v1、C18、C19 读 arXiv HTML 的设置与实验部分（部分只读前 10 万字符）；C05 另读官方 PDF 的 §4.1；C20、C21 读到关键段落；其余为摘要或出版记录级。
- 部分内容经网页工具转述后再要求逐字引文核对（SCPT §3.6 一度转述错误），关键否定判断只在读到原文的条目上作出。
- 代码只核对了仓库是否存在与少量关键实现（HD-TTS），未运行任何源代码。
- NeurIPS 2026 等新近论文索引不完整；E65 只有会议页面一个来源。
