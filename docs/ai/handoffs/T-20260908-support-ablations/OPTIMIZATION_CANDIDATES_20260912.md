# 两速度数据集：诊断与候选验证设计

> **2026-09-14状态更新：下文保留为2026-09-12历史设计，不是当前待执行清单。** C-S/route_shrink和C-M/bounded_mae已在第一轮两数据集完整训练，C-E/freshness已在第二轮两数据集完整训练，均未通过各自冻结门槛。C-U只做过固定0.75的GRIN局部消息干预，无一致正响应，可学习版本未训练；不能说全部C-U已排除，也不建议无新证据直接补跑。最新四损失候选均正常完成，Focal最有价值但仍未守住Full RMSE；本次决定结束优化搜索，不开新训练。详见[四损失结果及停止决定](../../../../paper/optimization/mae_tail_results_20260914/report.md)。

## Material Passport

- Task：T-20260908-support-ablations；2026-09-12 接手后的新增工作。
- Origin Skill / Mode：academic-research-suite / experiment-agent plan。
- Design identity：`two_speed_diagnosis_v1_20260912`。这是设计文档身份，**不是可执行训练计划或训练授权**。
- Verification：已有验证产物只读重算；候选未实现、未训练，机制因果关系未验证。
- 范围：METR-LA、PEMS-BAY；保留 Full、METR-LA 容量 8、PEMS-BAY 容量 11。容量不是分区数。
- 证据：[重算 JSON](DIAGNOSTIC_EVIDENCE_20260912.json)、[可复现脚本](reanalyze_saved_validation.py)、[原问题清单](ISSUES_20260912.md)。

## 1. 接手核验与证据边界

分支 paper/final，HEAD `9346dc0fc1dfb6b4aebfffe25224f1337f7a94cb`；相对本地 origin/paper/final 为 0/0，未 fetch；暂存区为空。已审阅现有 tracked diff 和 untracked 清单，没有覆盖代码、配置、论文或历史结果。

原 CHECKPOINT 的 88 个文件重新核对：87 匹配，只有 ISSUES_20260912.md 哈希不匹配。原快照没有保存该文件旧正文，无法仅凭哈希判断修改内容；保留当前文件与原快照，不将其追认成原版本。关键代码、配置、主表、消融产物及已保存配对预测均匹配。

使用 `/home/xd/anaconda3/envs/ReaCH/bin/python`（3.11.13）执行标准库重算；不导入训练模块，不读取原始数据或 checkpoint 权重。环境声明差异仍未解决。本轮未运行 builder/preflight、模型 forward、GPU、训练或 test，也未检查进程存活。

主表仍为 seed32、validation、mixed-source；重算不是新的独立实验。原六消融与 trajectory/q-layout 均完成，不重启。旧 final_plan 仍只作为历史身份，不作为新运行权威。

## 2. 已核验诊断：不能用一个整体开关解释所有缺口

以下差异统一定义为 `100 × (Full / control − 1)`，正值表示 Full 更差；与旧表以 Full 作分母的“降低比例”不同。

| 数据集 / 条件 | 对照 | Full 相对误差增加：MAE / RMSE / MAPE (%) | 诊断含义 |
|---|---|---|---|
| PEMS-BAY / Join | Mask-only | −6.14 / −5.31 / −11.18 | 整体去状态会损失接入优势 |
| PEMS-BAY / Fault | Mask-only | +3.95 / +2.61 / +6.74 | 优先排查故障条件下的负迁移 |
| PEMS-BAY / Steady | Static | +5.57 / +4.72 / +3.53 | 动态混合未体现一致必要性 |
| PEMS-BAY / Unseen | Mask-only | +1.16 / +0.89 / +6.33 | 无历史泛化及相对误差都需处理 |
| PEMS-BAY / Unseen | Static | −1.15 / −0.50 / +4.70 | 静态化只能改善其中的 MAPE，不能单独修复 Unseen |
| METR-LA / Unseen | Mask-only | −5.17 / −2.82 / −0.67 | 内部控制不能解决对 MoGERNN 的 Unseen MAPE 缺口 |
| METR-LA / Unseen | Static | −10.73 / −9.13 / −4.54 | 全面收缩路由有伤害无历史预测的风险 |
| METR-LA / Warm | Mask-only | −1.58 / −0.61 / +6.02 | 早接入的相对误差和绝对误差必须同时检查 |

完整 33 单元和三个内部控制差异均保存在重算 JSON；不得用此表截取替代完整门槛。PEMS-BAY 的 Join 正收益与 METR-LA 人工 Join-6 的负收益不是矛盾：数据集、采样总体和汇总方式均不同。

### 已保存 METR-LA 配对预测的速度分箱

按目标的原速度数值分为 [0,20)、[20,40)、[40,60)、[60,∞)，仅作诊断，不修改指标、epsilon 或样本 mask。各 arm 和 known/held-out 单独统计，不把同一目标在 20 个 arm 中的重复当作独立样本。MAPE 分母保持 `max(abs(y), 1e-6)`。

held-out 各 arm 均为 64 窗口、23 节点、768 个预测步；其中 [0,20) 只有 **45 步、6 窗口、4 节点**。以下 contribution 为该箱 APE 总和除以全部 768 步，与总 MAPE 的单位同为百分点，可跨箱相加。

| held-out arm / 对照 | Full 总 MAPE − 对照 (百分点) | [0,20) 对总差异的贡献 | 该箱平均偏差 Full / 对照 |
|---|---:|---:|---:|
| Full-history / Mask-only | +1.668 | +1.888 | +21.930 / +18.298 |
| Join-6 / Mask-only | +1.759 | +2.079 | +23.808 / +19.681 |
| Outage / MoGERNN | +1.345 | +1.748 | +29.420 / +25.576 |
| q=0 / MoGERNN | +2.277 | +2.364 | +30.770 / +25.576 |
| q=3 Early / MoGERNN | +2.687 | +2.541 | +26.976 / +21.536 |

偏差是 prediction − target。低速度箱存在更强的高估，其他箱的收益部分抵消它；因此不能仅用全局有符号均值解释 MAPE。q=3 Early 在 ≥60 箱也贡献 +0.649 个百分点，低速度不是唯一问题。

这支持“低速度高估值得优先诊断”，**不能外推成 PEMS-BAY 或自然 Unseen 的归因**。自然完整历史/未来的筛选、4 个低速度节点和重叠窗口限制外推；不作显著性宣称。保存预测重算的逐窗口 MAPE 与原 CSV 最大差异为 4.07e-5 个百分点（浮点精度差异），低于脚本断言 1e-4；目标、mask、节点、origin、track 的配对一致性通过。

## 3. 候选卡：单机制修改，按证据触发

下列是可审议设计，不是已选新 Full。各候选独立从当前 Full 出发；首轮不叠加，不默认接入 GWNet，不改变跨节点时间上下文或 patch 布局。

### C-S：动态路由向静态先验有限收缩（首选机制候选）

- 假设：部分条件的状态路由方差带来负迁移，但 Join 仍需要动态信息。
- 唯一修改：ALFGraph 路由采用 `g=(1−a)g_static+a*g_state`，其中 a 为仅由当前可用历史 mask、role、age 得到的共享标量，范围 [0,1]；初始化偏向原动态路径。保留 BASE 支持、GRIN、时间分支、FiLM、细化和训练目标。
- 与旧 SUP 的区别：不替换算子，不引入多跳/PPR 配方，不做 hard/top-k；与 Static 控制区别是仍允许状态自适应。应披露新增参数和计算。
- 前置诊断：按条件记录实际图卷积输出及混合后的有效传播；原支持相似度只是辅助，不能以路由 entropy 单独断言有效多样性。若算子或专家输出等效、混合变化几乎不改变预测，降低本候选优先级。
- 支持信号：PEMS-BAY Steady/Fault 负收益减小，同时 Join 优势保留；METR-LA Unseen 不退化。若只改善总平均而破坏接入/无历史优势，拒绝。
- 限制：不能据 Static 的 MAPE 优势预计 Unseen 三指标同时修复；若局部路由干预无响应，改查 GRIN/FiLM/细化，不能直接把整个状态管线一起重写。

### C-U：GRIN 内观测发送者与重建发送者分开归一化（Unseen 备选）

- 假设：无本地历史时，把不可靠重建消息与真实观测消息混合会加重低速度高估。
- 唯一修改位置：共享 GRIN 的消息聚合。用当步可见 mask 将发送者分组，分别归一化；用共享、有界门控混合两组。保留重建组，空组回退必须明确定义，不能直接切断所有无观测路径。gate 只用合法可见历史与图结构，不用真实隐藏值、V/T 身份或节点专属 embedding。
- 与已有 sender_gated_i_a_a2_residual 的区别：已有开关在 GRIN 后添加 I/A/A² 残差；本候选修改 GRIN 内的两类消息，保留重建组且不添加多项式传播。不是原 OBS_MASK/POLY 配方搬运，不改训练场景概率或新增 synthetic-unseen loss。
- 前置诊断：自然验证 Unseen 按可见邻居支持、目标速度和预测偏差分层，并记录重建误差；用现有 K 节点合法隐藏位置衡量重建，不能读取训练 V/T 信号。若误差与发送者可靠性没有对应关系，不进入训练。
- 支持信号：PEMS-BAY Unseen 三指标改善且中高速度绝对误差不恶化；METR-LA Unseen MAPE 改善。若仅校正低值而损失 RMSE，拒绝。

### C-E：仅在时间分支增加实际观测新鲜度编码（早接入备选）

- 假设：生命周期 age 不等于距最近一次真实观测的时间；q 相同的 Early/Recent 差异需要更直接的历史描述。
- 唯一修改：从 12 步完整 mask 计算逐步距上次观测的间隔和窗口 q，通过零初始化小投影加入 ALFTemporal token。保持 cross-node attention、分区、补齐规则及 FiLM 不变；不增加第二预测头。
- 输入未增加外部信息，是 mask 的确定性表示；需先核查已有通路是否已经等价编码，防止重复功能。q=0 的缺省间隔固定为窗口长度加一，不依据未来观测。
- 支持信号：METR-LA Cold/Warm 和 q=3/6 Early 改善，Recent/full 与 Fault 不退化。若只在选定自然完整窗口有效，不能晋升。

### C-M：有界相对权重的 MAE 辅助项（条件触发，非旧 LF）

- 假设：低速度高估与原 role-balanced MSE 对大绝对误差的偏好有关；当前分箱只提供局部支持，尚未隔离目标函数因果作用。
- 唯一修改：保留原预测目标及角色平衡，添加小权重、上界受限的 MAE 项。候选具体形式：标准化残差 e，`L=L_base+0.05*role_mean(w(y)*abs(e))`，`w=clip(tau/max(abs(y),epsilon),1,3)`；tau 为训练时间 K 的有效速度第 20 百分位，epsilon 沿用数值保护。权重仅训练使用，验证汇总完全不变。
- 与 LF 的区别：不是 signed_log_huber，不加低值校准头，不用验证集确定 tau；明确保留 MSE 主项，不重复原 λ/scale 配方。0.05、上界 3 是拟冻结的工程选择，尚无最优性依据。
- 前置诊断：自然验证两数据集的低速度贡献、正负偏差与 horizon 分层；检查训练 K 的 tau 有效且分箱不是几乎空集。不能仅由 METR-LA 的 4 个节点触发两数据集损失修改。
- 支持信号：MAPE 改善同时守住 MAE/RMSE；若出现旧 LF 的误差交换则停止，不改 checkpoint 指标追分。

## 4. 拟议验证门槛（必须在新运行前冻结）

定义误差降低 `gain(C,R)=100*(R−C)/R`。以下阈值是预先提出的实际改进门槛，不是显著性阈值或既有用户决定。

| 门槛 | 拟议规则 |
|---|---|
| 身份与公平性 | 新计划绑定完整工作区源码哈希（HEAD 不够）、外层/嵌套 YAML、数据/图/layout/mask/origin/scaler 身份、环境、seed、预算、输出根和 collector；builder --check 及 exposure preflight 通过后才具备可运行条件 |
| 首轮资源上界 | 最多两个通过前置诊断的单机制候选 × 两数据集 × seed32 = 4 个新候选训练；不是四候选全开，也不自动重训 Full/六消融 |
| 预算控制 | 与各数据集当前 Full 的训练预算、batch64、teacher forcing、optimizer、early stopping 保持一致；epochs 实际可不同，不允许候选单独延长找好点 |
| checkpoint | 继续以 validation LifeAvg MAE 严格改善选 checkpoint，tie 取更早 epoch；MAPE 门槛仅对这个 checkpoint 判定，不挑另一 epoch 的 MAPE 拼表 |
| 通用筛选 | 两数据集各自 LifeAvg 三指标相对 Full 退化均 ≤0.5%；33 单元中任一退化 >2% 判失败；所有指定修复单元必须改善，不能跨数据集平均抵消 |
| C-S 特定门槛 | PEMS-BAY Steady/Fault 的 MAE、MAPE 各自相对 Full 改善 ≥2%；Join 三指标退化 ≤0.5%；METR-LA Unseen 三指标退化 ≤0.5% |
| C-U 特定门槛 | PEMS-BAY Unseen MAE/RMSE/MAPE 各改善 ≥2%；METR-LA Unseen MAPE 改善 ≥1%；两数据集 Unseen 其他指标不退化 |
| C-E 特定门槛 | METR-LA Cold/Warm MAPE 各改善 ≥2%，MAE/RMSE 不退化；q=3/6 Early 配对 MAPE 改善且 full/Recent 不退化 >0.5%；自然阶段结果优先于人工窗口 |
| C-M 特定门槛 | 两数据集 Unseen MAPE 各改善 ≥2%，LifeAvg MAPE 各改善 ≥1%；LifeAvg MAE/RMSE 和 Unseen MAE/RMSE 不退化 |
| Full 晋升 | 筛选通过不等于替换 Full。统一新 Full 需两数据集 LifeAvg 三项均不差于旧 Full，且不差于各自 Mask-only/static 的逐指标较优值；仍公开全部内部控制与未通过项 |
| 最终缺口目标 | METR-LA Unseen MAPE ≤19.689703；PEMS-BAY Unseen MAE≤4.715492、RMSE≤6.865100、MAPE≤12.869602。均为当前 MoGERNN 参照，不能把“缩小缺口”写成“已全面超过” |

独立确认建议另立身份：只确认最多一个胜出候选，seed42/52 配对当前 Full 与必要内部控制。当前未授权这些训练，不能把 33 个相关单元、12 个 horizon 或 64 个窗口视为独立 seeds。多种子确认报告逐 seed 和均值；时间分块/节点聚类不确定性分析另预注册，三个 seeds 不足以承诺强显著性。验证已被多轮使用，其选择偏差继续披露；test 保持封存，冻结方案后另需明确授权。

## 5. 下一步与停止条件

建议先推进 C-S 的有效路由/路径定位和 C-U 的自然 Unseen 支持分层，C-E/C-M 保留为被诊断触发的备选。只有汇总的 PEMS-BAY 证据不能提供逐预测分箱；完整自然验证预测是否已保存、可否直接复用，尚未核实。若缺失，应另立 **validation-only 诊断推理** 的具体范围与身份，不能借“分析”自动启动重新评估。

下一运行计划建议名称 `two_speed_mechanism_v1`，但本轮不生成可执行 plan、不登记训练入口。冻结前必须选定实际候选、诊断/训练分开的任务清单、种子、时限和最多运行数；新输出根不得复用旧 final/support_ablations 目录。若 Full/控制无法证明同口径复用，则先记录差异；任何桥接重训计入单独明确的预算，不自动追加。

若前置诊断不能区分机制、任何候选触发关键退化、或只有局部人工样本涨分，则保留 Full 并记录失败，不扩大网格、不反复改门槛、不融合 GWNet 救场。

## 6. 本轮完成与未做事项

- 完成：Git/diff 核验、88 文件哈希比较、已有消融逐条件差异重算、已保存 METR-LA 预测的分箱/偏差/贡献分析、四张候选卡与分阶段门槛。
- 本轮新增脚本只作离线结果汇总，其断言检查配对目标/mask/元数据和逐窗口 MAPE；未做独立模型复现或完整仓库测试。
- 未开始：自然 Unseen 分箱、GRIN/FiLM/细化的局部干预、实际图算子数值距离与有效路由测量、候选实现/训练、独立确认、test、提交或推送。
- 排除：整体去状态作为默认升级；以路由权重冗余直接推断预测冗余；原样重跑 LF/SUP；将本候选文档视为可执行运行授权。
