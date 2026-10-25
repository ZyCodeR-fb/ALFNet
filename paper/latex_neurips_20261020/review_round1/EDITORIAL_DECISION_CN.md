# Editorial Decision（模拟评审第 1 轮）

## Manuscript Information
- **Title**: ALFNet：动态观测支持下的归纳时空预测
- **Manuscript ID**: alfnet-cn-v6
- **Submission Date**: 2026-10-20（中文 LaTeX 工作稿；评审副本为 `manuscript.md`，与 `sections/*.tex` 文字一致）
- **Decision Date**: 2026-10-20
- **Review Round**: Round 1
- **Mode / contract**: `reviewer_full`，`reviewer/reviewer_full/v2`；未提供目标会议评审标准绑定（`criteria_binding_unavailable`），不作会议契合度认证

## Review Panel Provenance (#540/#740)

- **Typed artifact**: `panel-provenance.json`（`review_panel_provenance.py build` 生成，`validate` PASS）
- **Artifact SHA-256**: e167fbe916854cdf4d3a9446cca3ab4dc3b23a74bc2e6f294b8cc9d90083938d
- **Panel ID**: alfnet-cn-v6-round1
- **Normalized manifest SHA-256**: dbe0925347907dc1a3e72fe1bb3cef9a5676bf7d35c62d38617fa1ab5433b8ed
- **Execution topology SHA-256**: b192cb30d0ef1c0e98ea407ceeb3c0ad2ec27158050bb357d4f77a591dc590a1
- **Fresh-context scope**: `within_panel_attempt_only`
- **执行说明**：本机插件中的合约文件为 CRLF 换行，统一为 LF 后与规范哈希 e9712090… 一致；构建脚本在临时目录的 LF 副本上运行，插件文件未改动。Schema 6 carrier 未生成。

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | main-session-20261020 | true | claude | anthropic | null |
| R1 | methodology | model | main-session-20261020 | true | claude | anthropic | null |
| R2 | domain | model | main-session-20261020 | true | claude | anthropic | null |
| R3 | perspective | model | main-session-20261020 | true | claude | anthropic | null |
| DA | da | model | main-session-20261020 | true | claude | anthropic | null |

| Provenance axis | Status |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation | false |
| Blind to peer outputs | false |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: Not computed. Persona or role diversity proves only `role_separated`; do not relabel the panel as independent.
- **Correlated-error disclosure**: All model-executed review seats used one model family; role separation does not remove correlated-error risk.
- **补充说明**：五个席位与稿件作者是同一模型、同一会话，Phase 1 并非真正"未见论文"。本轮意见应视为结构化自查，而非独立同行评审。

---

## Decision *

### Major Revision

机械综合：`dimension_verdicts: [D1=block, D2=warn, D3=block, D4=warn, D5=warn, D6=warn]`；`fired_conditions: [F2, F3, F5]`；最高严重度 F2（任一强制维度为 block）→ `editorial_decision=major_revision`。`check_panel_synthesis.py` 复算 PASS。

---

## Blocking Issues * (0–3, immutable source order)

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| B1 | 所有比较只有单次运行，计数式结论缺少方差支撑（D1 block） | R1 | text: §4.1 "所有实验使用随机种子32，因此不将较小的单次训练差异解释为统计显著" | R2 |
| B2 | 核心主张"观测支持作为预测计算的条件"缺少能隔离该机制的证据（D3 block，DA C1） | DA | table: 表2 四项组件消融（测试集 LifeAvg） | R9 |

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|---|---|---|---|
| Journal-Fit Reviewer | ML 会议应用型时空学习领域主席 | Major Revision | 4 |
| Reviewer 1 | 实验方法学与评测协议 | Major Revision | 4–5 |
| Reviewer 2 | 交通时空预测与归纳克里金 | Minor Revision | 3–4 |
| Reviewer 3 | 传感网络运维与部署 | Minor Revision | 3–4 |
| Devil's Advocate | Fixed adversarial seat | N/A — findings only | N/A — per-finding only |

---

## Consensus Analysis *

### Points of Agreement (Consensus)

**[CONSENSUS-3]**（EIC、R1、R2 同意，R3 未涉及）：
1. 结论与摘要的措辞强于证据范围，需要限定——EIC-W4（标题强调归纳但零历史结论为负）、R1-W1（单次运行下的计数式表述）、R2-W1/W4（"优于全部比较方法"应限定为所列方法及其适配版本）。

**[CONSENSUS-2，含 DA 佐证]**：
2. 比较公平性与覆盖不足——R1-W3/W4（调参预算、节点身份参数处理）、R2-W1（缺少 Graph WaveNet、AGCRN），DA-M5（缺少简单基线）佐证。
3. 测试集参与配置保留——R1-W2，DA-M1 佐证。
4. 消融覆盖不完整——R1-W6，DA-M3 佐证。

### Points of Disagreement

**Disagreement 1：核心主张证据缺口的严重程度**
- **R1 view**：D3 评为 warn，认为部分结论措辞强于证据，属于可修改的措辞问题（R1-W6）。
- **DA view**：D3 评为 block，C1 认为直接针对支持条件的两项消融幅度小、单次运行，最大的收益来自跨节点上下文，替代解释未被排除。
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: C1 判为 VALIDATED，作为阻断问题，但属于可修复（repairable）。
- **Resolution Rationale**: 表 2 数值确实显示去除状态年龄使 MAE 在三个数据集上上升约 0.3%–0.9%、以静态混合替代路由上升约 0.7%–2.0%，且稿件中没有整体去除支持条件的对照；但稿件已在 4.4 节承认消融范围有限，主张可以通过改写为"组织方式与跨节点上下文共同支撑"来修复，不构成拒稿理由。

**Disagreement 2：缺少自适应图基线的严重程度**
- **R2 view**：Major，需要补比较或在全文限定结论范围。
- **EIC view**：未单独提出，关注点在贡献表述。
- **Disagreement type**: Perspective difference
- **Editor's Resolution**: 保留为必改项（R7），以文字限定为最低要求，补实验由作者与导师决定。

---

## Decision Rationale *

本轮触发的最高条件是 F2：方法严谨性（D1）与论证一致性（D3）两个强制维度均被评为 block，且均为可修复类。D1 的问题在于所有结论来自单一随机种子，而文中大量使用"四个数据集三项指标全部变差""12 格中 11 格"这类依赖逐格方向的计数式表述（R1-W1）；测试集参与配置保留（R1-W2、DA-M1）与基线调参预算不对等（R1-W3）进一步增加了领先幅度的不确定性。D3 的问题由 DA 的 C1 提出并经编辑判定成立：直接作用于支持条件的两项消融幅度较小，主要收益来自跨节点上下文与时间分支，稿件尚无隔离"以观测支持为条件"这一机制本身的证据。其余维度均为 warn：领域准确性方面缺少自适应图基线（R2-W1），跨领域方面未讨论角色与年龄在部署中的可得性（R3-W1），编辑方面贡献表述停留在"组织方式"且篇幅超长（EIC-W1、W2）。

之所以不是小修：两项阻断问题若只靠措辞修改，能够降低主张强度，但无法补足证据；作者需要明确选择"弱化主张"还是"补实验"。之所以不是拒稿：没有致命问题，稿件对边界结果的披露相当坦诚，评价协议有价值，核心实验在所列基线范围内一致支持 ALFNet 的总体领先。

---

## Required Revisions * (Must Fix)

"修改类型"一列：**文字**＝可文字修改；**实验**＝需新实验或新分析（只记录，不执行，待导师决定）。

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | 修改类型 |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 提炼可检验的设计原则并写明与 MoGERNN 的差异 | — | major | text: §1 贡献段 "观测支持应同时作为输入有效性的描述和预测计算的条件" | 4 | EIC | must_fix | section: §1 末段、§3.2 | 文字 |
| R2 | 单次运行：计数式表述改为"在本次运行中"，小差异不作方向性解读；多种子重复 | — | major | text: §4.1 "所有实验使用随机种子32，因此不将较小的单次训练差异解释为统计显著" | 5 | R1 | must_fix | section: §4.2–§4.4、§5 | 文字＋实验 |
| R3 | 列出参考测试结果的决策范围；未经选择的留出确认 | — | major | text: §4.1 "部分历史配置的选择曾参考测试结果" | 4 | R1（DA-M1 佐证） | must_fix | section: §4.1、§5 | 文字＋实验 |
| R4 | 报告基线超参数来源与训练预算；等预算调参 | — | major | text: §4.1 "各基线保留自身预测主干与目标函数" | 4 | R1 | must_fix | section: §4.1（或附录） | 文字＋实验 |
| R5 | 说明基线在保留节点上的节点身份参数处理方式 | — | major | absence: §4.1 比较方法 — expected 基线在保留节点上的节点身份参数处理方式; checked §2.3、§4.1、表1注释、结论局限段 | 4 | R1 | must_fix | sentence: §4.1 比较方法 | 文字 |
| R6 | 说明消融覆盖范围及共享补全、角色缩放修正为未消融的设计选择（或加入已有结果） | — | major | text: §4.4 "表2只覆盖生命周期信息、观测状态条件化传播与时间上下文四项干预" | 4 | R1（DA-M3 佐证） | must_fix | section: §3.3、§3.6、§4.4 | 文字（已有结果可加入） |
| R7 | 补 Graph WaveNet、AGCRN；在此之前将优越性表述限定为"所列八个比较方法" | — | major | text: §4.1 "Graph WaveNet 与 AGCRN 学习的空间关系在推理时不随输入窗口变化" | 4 | R2 | must_fix | sentence: 摘要、§1、§5 | 文字＋实验 |
| R8 | 说明角色与状态年龄在部署中的获取方式；评估标注延迟或误标的影响 | — | major | text: §3.1 "除掩码外，观测支持还包括两类显式描述" | 4 | R3 | must_fix | section: §3.1、§5 | 文字＋实验 |
| R9 | 核心主张改写为由组织方式与跨节点上下文共同支撑，写明支持条件本身的作用有限；整体去除支持条件的对照与多种子 | — | critical | table: 表2 四项组件消融（测试集 LifeAvg） | 4 | DA-C1 | must_fix | section: 摘要、§1、§4.4、§5 | 文字＋实验 |
| R10 | 调整标题、摘要与贡献对"归纳"的定位，写明优势集中于本地证据部分存在的条件 | — | major | text: 摘要 "在两个流量数据集上未能成立" | 4 | DA-M2（EIC-W4 佐证） | must_fix | sentence: 标题、摘要、§1 | 文字 |
| R11 | 结论中"时间布局与可见数量同样重要"按 4.8 节的划分依赖性改写 | — | major | text: §4.8 "布局优势具有样本和划分依赖性" | 4 | DA-M4 | must_fix | sentence: §5 第二段 | 文字 |
| R12 | 补充历史均值或持续性等简单基线；在此之前限定 Join/Fault 结论为相对所列方法 | — | major | absence: §4.1 比较方法 — expected 历史均值或持续性等简单基线; checked §4.1、表1、图2 | 3 | DA-M5 | must_fix | sentence: §4.3、§5 | 实验＋文字 |

### Required Item Details

**R1: 贡献的原则性表述**
- **Problem**: 贡献停留在"组织方式"，缺少可迁移的原则。
- **Source**: EIC-W1。
- **Requirement**: 在引言末段与 3.2 节写出 2–3 条可检验的设计原则，并指明对应实验；给出与 MoGERNN 的对照。
- **Acceptance criteria**: 引言与 3.2 节中每条原则都能对应到表 2 或图 3/4 的具体证据。

**R2: 单次运行的表述限定**
- **Problem**: 计数式结论依赖单次运行。
- **Source**: R1-W1。
- **Requirement**: 文字层面限定措辞；多种子重复记为待定实验。
- **Acceptance criteria**: 全文不再出现未加"本次运行"限定的计数式结论；差异小于约 1% 的格子不作方向性解读。

**R3: 测试集参与选择的范围**
- **Problem**: 披露未说明受影响的决策。
- **Source**: R1-W2，DA-M1 佐证。
- **Requirement**: 列出受影响的数据集与设置项。
- **Acceptance criteria**: 4.1 节或附录能让读者判断哪些配置受测试信息影响。

**R4: 基线调参预算**
- **Problem**: 未报告基线超参数来源与预算。
- **Source**: R1-W3。
- **Requirement**: 列出各基线超参数来源与训练预算。
- **Acceptance criteria**: 每个基线都有超参数来源说明。

**R5: 节点身份参数处理**
- **Problem**: 归纳设置下基线的逐节点参数处理方式缺失。
- **Source**: R1-W4。
- **Requirement**: 在 4.1 节说明统一替代方式及其影响。
- **Acceptance criteria**: 4.1 节明确写出处理方式，并在局限中说明。

**R6: 消融覆盖范围**
- **Problem**: 共享补全与角色缩放修正无消融。
- **Source**: R1-W6，DA-M3 佐证。
- **Requirement**: 在方法或 4.4 节说明这两项是未经消融的设计选择，或加入已有结果。
- **Acceptance criteria**: 读者能从正文判断哪些模块有消融支持、哪些没有。

**R7: 自适应图基线**
- **Problem**: 缺少领域标准基线。
- **Source**: R2-W1。
- **Requirement**: 最低限度在摘要、引言、结论限定结论范围。
- **Acceptance criteria**: 所有优越性表述都带"所列比较方法"的限定。

**R8: 条件信息的可得性**
- **Problem**: 未讨论角色与年龄的部署来源与误差。
- **Source**: R3-W1。
- **Requirement**: 在 3.1 节或局限中说明假设；稳健性实验记为待定。
- **Acceptance criteria**: 文中写明角色与年龄在评价中由协议给出，并讨论部署中可能的延迟与误标。

**R9: 核心主张的证据缺口**
- **Problem**: 支持条件本身的作用未被隔离。
- **Source**: DA-C1。
- **Requirement**: 改写摘要、引言、4.4 与结论中的核心主张，使其与表 2 的证据强度一致。
- **Acceptance criteria**: 主张不再暗示支持条件是收益的主要来源；4.4 节写明直接针对支持条件的消融幅度及其不确定性。

**R10: 归纳定位**
- **Problem**: 定位与零历史结果不一致。
- **Source**: DA-M2，EIC-W4 佐证。
- **Requirement**: 调整标题或摘要首句的定位。
- **Acceptance criteria**: 标题、摘要与贡献对优势条件的描述与 4.3 节一致。

**R11: 布局结论**
- **Problem**: 结论表述强于 4.8 节的跨划分证据。
- **Source**: DA-M4。
- **Requirement**: 按 4.8 节的限定改写结论。
- **Acceptance criteria**: 结论提到布局时同时说明其划分依赖性。

**R12: 简单基线**
- **Problem**: 缺少朴素基线。
- **Source**: DA-M5。
- **Requirement**: 实验记为待定；文字限定 Join/Fault 结论的比较范围。
- **Acceptance criteria**: 4.3 与结论中 Join/Fault 的结论带比较范围限定。

---

## Suggested Revisions (Should Fix)

| Transport ref | Revision Item | Severity | Evidence Anchor | Source Reviewer | Obligation class | 修改类型 |
|---|---|---|---|---|---|---|
| S1 | 压缩篇幅：超参数敏感性与部分案例移至附录，合并重复的边界说明 | minor | text: §4.7 "参数作用具有数据集依赖性" | EIC | should_fix | 文字 |
| S2 | 统一图 1 与公式的符号（候选式缺 x、P⁺Ỹ 写成加减、Y0→Ỹ） | minor | figure: 图1 Graph-GRU gates 与 Role-Scaled Graph Refinement 区域 | EIC | should_fix | 文字（作者重绘） |
| S3 | 标题或摘要首句突出"观测支持变化"（与 R10 同源） | minor | text: 标题 "动态观测支持下的归纳时空预测" | EIC | consider | 文字 |
| S4 | 写明 MAPE 分母处理规则及流量低值敏感性 | minor | text: §4.1 评价指标 "MAPE 以百分比表示" | R1 | should_fix | 文字 |
| S5 | 机制分析补 bootstrap 区间；在此之前改为描述性措辞 | minor | text: §4.8 "窗口胜率依次为54.7%、51.6%、50.0%和62.5%" | R1 | consider | 新分析（离线）＋文字 |
| S6 | 附录给出协议伪代码、事件长度分布、保留节点名单与代码可用性声明 | minor | absence: §3.1 与 §4.1 — expected 事件长度抽样分布、保留节点名单与代码可用性声明; checked §3.1、§4.1、结论 | R1 | should_fix | 文字 |
| S7 | 讨论近两年面向训练数据不足节点的归纳预测工作 | minor | absence: §2.2 — expected 2024–2025 年面向训练数据不足节点的归纳时空预测方法; checked §2.2、§4.1 比较方法、参考文献表 | R2 | consider | 文字／实验（与作者此前删除 ST-FiT 的决定相关，由作者决定） |
| S8 | 说明协议结果与常见直推设置不可直接比较 | minor | text: §4.1 "各条件内按日历与事件进展分层平衡汇总" | R2 | should_fix | 文字（＋可选实验） |
| S9 | 摘要与结论注明比较对象为共同协议下的适配版本 | minor | text: 摘要 "三项指标在四个数据集上均优于八个比较方法" | R2 | should_fix | 文字 |
| S10 | 报告训练与推理耗时、显存 | minor | absence: §4.1 训练与实现 — expected 训练与推理耗时及显存占用; checked §4.1、§4.7、第5节 | R3 | should_fix | 文字（如有既有记录） |
| S11 | 说明数据集特定设置的选择依据与推荐默认值 | minor | text: §3.8 "这些设置不可学习，也不在推理时由路由选择" | R3 | should_fix | 文字 |
| S12 | 结论中加一句对其他传感网络的适用范围说明 | minor | absence: 第5节 — expected 对其他传感网络场景的适用范围与迁移前提说明; checked 第1节、第5节 | R3 | consider | 文字 |

---

## Revision Roadmap *

### Source-traceability checklist

> 按来源顺序列出，不代表工作顺序。作者对每项选择 `will_address`、`wont_address` 或 `not_on_point`。

- [ ] R1 — obligation `must_fix`: 提炼设计原则并对照 MoGERNN（文字）
- [ ] R2 — obligation `must_fix`: 单次运行的表述限定（文字）；多种子（实验，待定）
- [ ] R3 — obligation `must_fix`: 测试参与选择的范围（文字）；留出确认（实验，待定）
- [ ] R4 — obligation `must_fix`: 基线超参数来源与预算（文字）；等预算调参（实验，待定）
- [ ] R5 — obligation `must_fix`: 基线节点身份参数处理（文字）
- [ ] R6 — obligation `must_fix`: 消融覆盖范围说明（文字；已有结果可加入）
- [ ] R7 — obligation `must_fix`: 限定为所列比较方法（文字）；补自适应图基线（实验，待定）
- [ ] R8 — obligation `must_fix`: 角色与年龄的部署假设（文字）；误标稳健性（实验，待定）
- [ ] R9 — obligation `must_fix`: 核心主张改写（文字）；整体去除支持条件的对照（实验，待定）
- [ ] R10 — obligation `must_fix`: 归纳定位调整（文字）
- [ ] R11 — obligation `must_fix`: 布局结论改写（文字）
- [ ] R12 — obligation `must_fix`: 限定 Join/Fault 比较范围（文字）；简单基线（实验，待定）
- [ ] S1 — obligation `should_fix`: 压缩篇幅（文字）
- [ ] S2 — obligation `should_fix`: 图 1 符号（作者重绘）
- [ ] S3 — obligation `consider`: 标题定位（文字）
- [ ] S4 — obligation `should_fix`: MAPE 分母说明（文字）
- [ ] S5 — obligation `consider`: 机制分析区间（新分析，待定）
- [ ] S6 — obligation `should_fix`: 协议与复现细节（文字，附录）
- [ ] S7 — obligation `consider`: 近期归纳预测工作（作者决定）
- [ ] S8 — obligation `should_fix`: 协议不可比说明（文字）
- [ ] S9 — obligation `should_fix`: 适配版本限定（文字）
- [ ] S10 — obligation `should_fix`: 计算代价（文字）
- [ ] S11 — obligation `should_fix`: 设置选择依据（文字）
- [ ] S12 — obligation `consider`: 适用范围（文字）

未生成 `revision-roadmap/1.0` 机器文件；阶段 5 的修改按本清单人工执行并逐项记录。

---

## 对应矩阵必查项核对（ALIGNMENT_MATRIX_CN.md）

| 矩阵检查规则 | 评审结果 | 相关条目 |
|---|---|---|
| 引言每个问题都有方法与证据 | P1 的核心主张证据不足；P7（共享补全）、P9（角色缩放修正）无正文证据；P4 证据依赖划分 | R9、R6、R11 |
| 每个方法模块能回答引言问题并有消融或干预支持 | 共享补全、角色缩放修正、角色编码无消融，需在正文说明 | R6 |
| 相关工作三个方向与对比模型一一对应 | 2.3 方向缺 Graph WaveNet、AGCRN 的比较；2.2 方向近期工作讨论不足 | R7、S7 |
| 贡献与结论与实际结果一致 | 摘要"优于八个比较方法"未注明适配版本；结论布局表述强于 4.8；标题强调归纳但零历史为负 | S9、R11、R10 |
| 术语前后统一 | 术语已统一；缩写检查提示 GRU 先用后定义、图 1 图注中的 WM 未定义、MAE/RMSE/MAPE 未给全称 | 见附件 |

---

## Journal-Supplied Deadline (Optional Transport)

- **Exact deadline from source letter**: NOT PROVIDED

---

## Closing

我们建议作者认真考虑审稿意见并提交实质修改后的稿件，修改稿将进入下一轮核验评审。需要新实验的条目只作记录，是否执行由作者与导师决定；本轮修改只处理文字层面的问题。

---

## Appendix: Full Reviewer Reports

- Phase 1 预承诺：`eic.phase1.md`、`methodology.phase1.md`、`domain.phase1.md`、`perspective.phase1.md`、`da.phase1.md`（`check_phase_conformance.py --phase1-only` 均 PASS；"触发条件少于 8 个词"为按空格计词造成的提示）
- Phase 2 评审卡：`eic.phase2.md`、`methodology.phase2.md`、`domain.phase2.md`、`perspective.phase2.md`、`da.phase2.md`（`check_phase_conformance.py` 均 PASS）
- 机械综合：`synthesis.md`（`check_panel_synthesis.py` PASS）
- 领域分析与配置卡：`FIELD_ANALYSIS_CN.md`

---

## Attachment: Acronym Check (advisory, #849)

`check_acronyms.py --input manuscript.md --lang en` 的完整输出见 `acronym_report.md`（提示性质，不产生审稿意见，无需回复）。其中 AAS/NAS/VS/FS 的"Defined again"来自作者要求的"术语后补英文缩写"，属于有意保留。
