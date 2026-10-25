# 检查点 1：事实摘要、不一致与缺口

日期 2026-10-20。只读核对，没有训练、推理或访问测试集；没有修改 v5、实验结果、模型或配置。"已验证"指本轮在文件或源码中逐项核对过；"依据文档"指沿用同一 HEAD 下已有核验文档、本轮未逐行复读。

## 1. 仓库与权威来源

| 项 | 事实 | 状态 |
|---|---|---|
| Git | 分支 `paper/final`，HEAD `f06367e`，与 `origin/paper/final` 0/0（未 fetch）。既有改动保留：tracked 删除 `paper/manuscript_draft_cn-v4.md`、`paper/manuscript_draft_cn.md`；未跟踪 `paper/architecture_20261019/claude_handoff_20261019/`、`nanobanana_v4_20261019/`、`paper/writing_20261020/` | 已验证 |
| 最终配置 | [selected_configuration_manifest.json](../../pems08_wd_test_20261012/selected_configuration_manifest.json)：METRLA_patch7、05_PEMSBAY_combo2、06_PEMS04_bridge_delta1、13_PEMS08_weight_decay_0.001；seed 32；披露"历史测试参与了配置保留" | 已验证 |
| 主表 / 阶段表 | [selected_tables.json](../../pems08_wd_test_20261012/selected_tables.json)；降幅 [summary.json](../../pems08_wd_test_20261012/summary.json) | 已验证（表1 108 值逐值一致） |
| 正文消融 | [four_dataset_results.json](../../pems08_wd_ablation_test_20261012/four_dataset_results.json)：Full＋5 项（含 w/o Graph Refinement） | 已验证（表2 60 值一致） |
| 全量 12 类消融 | [all_ablations_test_20261010/results_and_audit.json](../../all_ablations_test_20261010/results_and_audit.json)；PEMS08 行对应旧 WD=1e-4 Full | 已验证（计数见 §4） |
| 图 2–6 | [final_figures_20261017](../../final_figures_20261017/README.md)；正文用 validation（图2 用 test） | 已验证（图2 非第一格、图3 18 个比较点、图4 胜出计数、图6 胜率） |
| 参数量 | 四份 run_report.json `parameter_count`=1,037,529（文件在 `artifacts/experiments/…/seed32/`，非 `paper/`；best.pt 本地不存在） | 已验证；阶段 3 用脚本把来源写入 `tables/provenance.json` |
| 数据集规模 | manifest 时间范围推得 METR-LA 34,272、PEMS04 16,992、PEMS08 17,856（天数×288）；PEMS-BAY 181 天×288=52,128，与 v5 的 52,116 差 12，需另找来源（原始数据缺若干时刻） | 部分已验证；PEMS-BAY 待取证 |
| 保留节点 | manifest `unseen.count`：50/81/77/42（比例 0.2415/0.2492/0.2508/0.2471），验证与测试各半 | 已验证（manifest） |
| TeX 环境 | 本机无 latexmk/pdflatex/xelatex/lualatex/tectonic/bibtex/biber | 已验证 → Overleaf 编译 |
| 导师已知信息 | [advisor_report_cn_v5.md](../../advisor_report_cn_v5.md) 为当前版本（四数据集、八基线、四项消融、Refinement 已从正文移除）；`advisor_paper_talk_cn.md` 是 v4 时期两数据集讲稿，已过时 | 已验证 |

## 2. 方法实现事实（写方法时以此为准）

逐项与源码和四份最终配置对照；行号对应 HEAD `f06367e`。

- **输入**：`x_enc=[X_vis, R(4), a, M]` 共 7 通道，全 N 节点；生命周期上下文 6 维（`ALFNet.py:_v4_inputs`）。
- **角色与场景**（`lifecycle_roles.py`、`v4_lifecycle_loader.py`）：AAS=0、NAS=1、VS=2、FS=3。评价时 Unseen 与 Join 采样保留节点，Steady 与 Fault 采样已知节点；自然缺失只改变 M，不改变角色。
- **SAHC**：一次 `shared_input_imputer` 调用（`ALFNet.py:384`），两分支切同一张量前缀。flow 适配器以 forward hook 替换补全输出：`(~mask)&available` 处用窗口内线性插值覆盖，端点外取最近可见值，再减中心（`flow_support_transfer.py:9-31`；`flow_unified.py:10-46`）；PEMS04 `bridge_delta1`（插值、无中心、d=1），PEMS08 `triple_halfcenter`（插值、中心 0.5·last、d=1）。q=0 节点 `available=False`，保留 GRIN 输出，中心为 0。
- **ALFGraph**：编码与解码各 2 层（`e_layers=2`），`d_model=64`，`k_hop=1`；`moge_router_mode=soft`（四数据集），`top_k`/`noisy` 字段在 soft 下不起作用。路由 `softmax(S·W_gate)`（无偏置）；传播 `Σ_j w_j⊙(P_j U Θ_j)+b`（`lifecycle_moge.py`，`LifecycleRoutedGraphConv.forward`）。支持：identity、binary mean、weighted mean、forward/backward random walk；weighted mean 与 forward random walk 在非负行上为同一归一化形式，但 Θ 与路由权重独立。
- **FiLM**：`film_head = Linear(160→32)→ReLU→Linear(32→128)`，末层零初始化（`ALFNet.py:244-249`）；`hidden*(1+Δγ)+β` 广播到两层，在解码前作用一次（`ALFGraph.py:315-330`）。
- **解码**：`initial_input = x_enc[:,-1,:,:1]`（中心化后的 X̄_L）；`output = anchor + d·projection(h_top)`；下一步输入为未修正 output（训练时按掩码教师强制）；解码路由固定用 `S_L`（`ALFGraph.py:97-118, 331-344`）。速度 d=0.3，流量 d=1（适配器覆盖）。
- **恢复与修正**：流量适配器返回 `pred + center` 后，`ALFNet.forward` 再调用 `graph_refiner`（`ALFNet.py:456`）→ 先 Restore 后 Refinement。Refiner：`A+I` 行归一化 P⁺ 与转置 P⁻，`[Y,P⁺Y,P⁻Y]→Linear(3→32)→GELU→Linear(32→1)`（末层零初始化）→ `Y + 0.05·s(R_L)·tanh(·)`，`s=[0,1,1,0.25]`（四份 YAML `graph_diffusion_role_scale`）。
- **损失**：`focal_elements = |Δ|·(2σ(0.2·std·|Δ|)−1)`，Δ 为标准化误差，等价于 `(e/σ_data)·tanh(0.1e)`（e 为原单位绝对误差）；`optimize = 0.9·base + 0.1·L_role + aux`（`v4_runner.py:379-444`；β=0.1 来自 YAML）；aux 为 12 个路由模块 `0.01·(CV²(importance)+CV²(load))` 的均值，soft 下 load 恒定、该项为 0。
- **ALFTemporal**（四份 patch YAML）：`layers=1`、`tem_patchnum=1`、`recur_times=5`（P=32）、`spa_patchsize`=7/16/10/6（即 C）、`crossformer_router_factor=4`（4 latent）、`crossformer_dropout=0.1`、input/node/tod/dow 维度 64/32/32/32。
- **未用参数**：`state_encoder`（6→32→32）在 forward 中计算但输出被丢弃（`ALFNet.py:_state`，`forward` 中 `_, last_availability`），参数计入 1,037,529。
- **依据文档、本轮未逐行复读**：BiGRIL 内部两阶段估计与融合、时间卷积 9→64、结构编码 7→64→32 的特征构成、注意力头数 1、ToD/DoW 计算、补足槽位用邻接排序真实节点。阶段 2 写到对应段落前复读源码。

## 3. 关键发现：ST-FiT 与 Merlin（B1）

- 2026-10-01 的 117 项测试中包含 ST-FiT 与 Merlin，与现有八基线同协议、同划分、同种子；同日用户要求从比较表移除，原表备份于 [before_model_filter_20261001](../../final_four_tables_20261001/before_model_filter_20261001/)。之后基线没有重训，现有八基线的数值与该备份逐值相同。
- 用当前最终 ALFNet（`selected_tables.json`）与备份中的 ST-FiT、Merlin 测试值离线比对：

| 数据集 | 当前非第一格（八基线） | 计入 ST-FiT＋Merlin | 新增失守格 |
|---|---:|---:|---|
| METR-LA | 0 | 3 | Merlin：Steady MAE/MAPE、Outage MAE |
| PEMS-BAY | 4 | 11 | ST-FiT：Join MAE/MAPE；Cold 三项；Warm MAE/MAPE；Ready MAE/MAPE |
| PEMS04 | 4 | 18 | Merlin：Steady 三项；Fault 三项；Onset/Decay/Outage 各三项（Outage MAE 26.568 对 46.619） |
| PEMS08 | 3 | 14 | Merlin：Steady MAPE；Fault 三项；Onset MAPE；Decay、Outage 各三项；ST-FiT：Onset MAPE |

- LifeAvg 三项在四个数据集上仍为第一（ST-FiT/Merlin 的 LifeAvg 均高于 ALFNet）。
- 背景（依据 [BASELINE_FAIRNESS_20260928.md](../../BASELINE_FAIRNESS_20260928.md)）：Merlin 教师可访问训练时段已知节点被掩码隐藏的完整历史，是源方法所需的训练期特权信息；它保留逐节点 ID 嵌入，与其他五个基线改用结构编码的政策不同；训练为两阶段、各 100 轮。这些可以作为讨论理由的素材，但移除发生在测试之后，理由必须由用户与导师确认，并在论文中披露。

## 4. 消融证据全貌（附录依据）

当前配对五项（[four_dataset_results.json](../../pems08_wd_ablation_test_20261012/four_dataset_results.json)，测试 LifeAvg 变差格数/12）：State Age 11（PEMS04 用重复记录；用原始记录为 10）、Adaptive Routing 12、Spatial Context 12、ALFTemporal–FiLM 12、Graph Refinement 10（PEMS08 去除后 MAE −4.09%、RMSE −3.26%）。

历史 12 类（[results_and_audit.json](../../all_ablations_test_20261010/results_and_audit.json)，PEMS08 行对应旧 WD=1e-4 Full；括号内为仅计当前配置三数据集 METR-LA/PEMS-BAY/PEMS04 的变差格数/9）：

| 干预 | 变差/12 | 当前配置三数据集 | 备注 |
|---|---:|---:|---|
| w/o ALFRecon（学习式补全→均值填充） | 8 | 7/9 | PEMS04 MAE/RMSE 改善 |
| w/o Expert Lifecycle Context | 4 | 3/9 | 多数数据集 MAE 改善 |
| w/o FiLM Multiplicative（γ） | 5 | 4/9 | PEMS-BAY 三项改善 |
| w/o FiLM Additive（β） | 5 | 4/9 | PEMS-BAY 三项改善 |
| Pointwise Refinement | 10 | 9/9 | |
| Stop Graph→Reconstruction Gradient | 5 | 4/9 | |
| Stop Temporal→Reconstruction Gradient | 10 | 7/9 | |

PEMS04 w/o State Age（B2）：原始 39.3139/52.4360/23.9545，重复 39.7565/53.1362/24.9573，Full 39.5654/53.3461/23.2568（[ablation_repeat_test_20261012/results.json](../../ablation_repeat_test_20261012/results.json)）。

## 5. 不一致与缺口清单

| # | 位置 | 问题 | 严重度 | 处理 |
|---|---|---|---|---|
| I1 | 4.1 比较方法 / 2.2 | ST-FiT、Merlin 测试后移除（B1） | 高 | 待决定 |
| I2 | 表2 | PEMS04 w/o State Age 用重复记录（B2） | 中高 | 待决定 |
| I3 | 4.1 / 2.3 | Graph WaveNet、AGCRN、PDFormer、GCN-M 未比较，理由不统一（B3） | 中 | 文字补理由；GCN-M 理由已有审计依据 |
| I4 | 第 3 节 | SAHC、Refinement 无正文消融 | 中 | 方法如实写，附录给全表 |
| I5 | 4.1 协议 | 未写明 Unseen/Join 在保留节点、Steady/Fault 在已知节点评价 | 中 | 方法 3.1 与 4.1 补写 |
| I6 | 参考文献 | 方法需引用 DCRNN、GRU（Cho et al. 2014）、Crossformer、稀疏/图专家混合（Shazeer et al. 2017、GMoE）；这些不在 v5 与相关工作证据中 | 中 | 阶段 2 前用 ARS citation-check 查证，查证后再写入 |
| I7 | 2.2 | v5 删去了相关工作证据中要求引用的 SC-SAMG（support-conditioned 命名近邻） | 中 | 建议恢复 |
| I8 | 术语 | v5 模块名与 v2 名称表不一致（§6 术语表） | 低 | 统一 |
| I9 | 4.1 | "样本均值项"应为"有效目标元素均值项"；"模型辅助损失"应写明是路由平衡项；"无可见历史回退到学习式重建"须补"所有节点先统一执行 GRIN" | 低 | 文字修改 |
| I10 | 4.5 | PEMS08 部分中断两点 ALFNet 分别高 4.0%、6.1%，"接近"偏弱 | 低 | 改写 |
| I11 | 贡献句 | "逐步加入与中断条件下优势一致"应限定"主条件、所列基线" | 中 | 改写（受 B1 影响） |
| I12 | `experiments_draft_cn.md` 4.6 | PEMS08 q=6 Recent/Early 写 22.444/27.888，与 `fig4_values.csv` 和 v5 的 22.822/28.404 不符；该稿已过时 | 低 | 以 v5/CSV 为准，不改旧稿 |
| I13 | `experiments_appendix_cn.md` | A.1–A.9、A.12–A.14 多为历史配置，含 FCLSTM/STGODE 等已退出基线，图片为服务器绝对路径 | 高（对附录） | LaTeX 附录需从当前权威文件重建 |
| I14 | 符号 | 架构图与 v2 README 中 P、G、H、α、y₀/Y₀ 存在一符多义（见方法大纲符号表） | 低 | 方法统一，图注同步 |
| I15 | 参数量 | 1,037,529 含未用 `state_encoder`；"w/o Spatial Context 参数量不变"需当前配置 run_report 取证 | 低 | 阶段 3 脚本取证 |
| I16 | 数据集表 | PEMS-BAY 时间点数 52,116 需来源 | 低 | 取证或引用原数据集论文 |

## 6. 论文配置记录（ARS Phase 0 PCR，请确认）

| 项 | 设定 |
|---|---|
| 论文类型 | 会议论文（实证 ML 方法），IMRaD 变体：引言、相关工作、方法、实验、结论＋附录 |
| 领域 | 时空预测 / 交通预测（图神经网络） |
| 目标会议 | NeurIPS 样式 preprint 模式（导师阅读用）；正文 9 页上限，附录不限 |
| 语言 | 正文写作语言见方法大纲 §0 的选择；最终英文；沟通简体中文 |
| 引用格式 | natbib 数字编号（`plainnat` 或模板自带）；不使用 APA |
| 摘要 | 仅英文摘要（中文工作稿可附中文对照），不做 ARS 默认的繁体中文双语摘要 |
| 篇幅目标 | 方法约 1,500–1,800 英文词（约 2 页，含 1 张架构图、1 个小表）；摘要 ≤250 词；结论约 250–350 词 |
| 已有材料 | 引言、相关工作、实验、参考文献（v5）；架构图（暂用）；全部表图与 CSV |
| 文献调研 | 已完成，跳过 ARS Phase 1；只对方法新增引用做 citation-check |
| 跨模型验证 | 不开启（`ARS_CROSS_MODEL` 关闭） |
| 评审标准绑定 | 未提供 ReviewTargetContext：`criteria_binding_unavailable`，不声称符合特定会议评审标准 |
| 必备声明 | NeurIPS 检查表（paper checklist）、局限性、数据可用性、代码可用性；ARS 的 CRediT/资助/伦理声明按导师意见决定，preprint 版先留占位 |
