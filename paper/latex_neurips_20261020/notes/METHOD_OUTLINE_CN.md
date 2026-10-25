# 第 3 节方法：大纲、符号表、公式清单与证据映射

ARS `academic-paper` outline 模式（Phase 2 Structure）产物，只含大纲，不含正文。依据：源码与四份最终配置（HEAD `f06367e`）、[v2 README](../../architecture_20261019/v2_20261019/README_CN.md)、[第一阶段审计](../../architecture_20261019/STAGE1_IMPLEMENTATION_AUDIT_CN.md)、[nanobanana v4 README/DESIGN](../../architecture_20261019/nanobanana_v4_20261019/README_CN.md)。事实核对见 [CONTEXT_FACTS_CN.md](CONTEXT_FACTS_CN.md) §2。

`criteria_binding_unavailable`：未提供目标会议的评审标准绑定，本大纲不声称符合某会议评审标准。

## 0. 写作语言选择（检查点 1 已确认：中文 v6）

建议阶段 2 先写**中文 v6**：`paper/latex_neurips_20261020/drafts/manuscript_v6_cn.md`，内容为 v5 第 1、2、4 节（只做术语统一与本矩阵列出的文字修正，不改数值）＋新写第 3 节、摘要、结论。理由：你说明正文先用中文与导师定稿；方法公式和措辞在中文稿上更易逐条核对。阶段 3 再译为英文 LaTeX。如果希望直接写英文，阶段 2 改为直接产出 `sections/method.tex` 等英文文件。

## 1. 结构与篇幅

- 结构：会议论文 IMRaD 变体。第 3 节按"设置 → 总览 → 数据流顺序的四个模块 → 训练目标 → 数据集配置"组织，与架构图从左到右、从上到下的阅读顺序一致。
- 篇幅（英文词，正文 9 页约束下）：

| 小节 | 目标词数 | 图表 |
|---|---:|---|
| 3.1 问题设置与观测支持 | 320 | 小表：四场景的角色/年龄轨迹（可并入正文或附录） |
| 3.2 总览 | 150 | 图1 架构图（通栏） |
| 3.3 SAHC | 230 | — |
| 3.4 ALFGraph | 400 | — |
| 3.5 ALFTemporal 与 FiLM | 330 | — |
| 3.6 Role-Scaled Graph Refinement | 130 | — |
| 3.7 训练目标 | 200 | — |
| 3.8 数据集特定设置 | 80 | 引用 4.1 的设置表 |
| **合计** | **≈1,840** | |

篇幅风险：v5 第 4 节有 2 表 5 图，按 NeurIPS 9 页正文估计放不下。阶段 3 需要把部分实验内容（例如 4.7 超参数、4.8 案例的一部分）移到附录，届时另行确认。

## 2. 架构图 ↔ 小节对应

| 架构图区域（v4 图） | 小节 | 图中关键标注 |
|---|---|---|
| 左列 Inputs / Lifecycle Context（四场景轨迹，R、a、M） | 3.1 | Unseen/Steady/Join/Fault；`S_t=[R_t,a_t,M_t]` |
| 浅红 Support-Aware History Completion | 3.3 | BiGRIL 双向→保留观测→Interpolation*（q>0）→Reference centering*→X̄ |
| 绿色 ALFTemporal（嵌入→分区→跨分区→分区内→恢复顺序→Z→FiLM head） | 3.5 | 1 patch/节点，160 维，P×C，4 latent |
| 黄色 FiLM（Δγ→⊙，β→⊕） | 3.5 | 作用于两层编码末状态，一次 |
| 蓝色 ALFGraph（Encoder、cell 放大框、Residual AR Decoder） | 3.4 | 五专家 dense 路由；`y₀=X̄_L`；×12 步回送未修正 `y_k` |
| Restore ⊕（+αv） | 3.4.3 | 先于修正 |
| 浅橙 Role-Scaled Graph Refinement | 3.6 | `[Y,P⁺Y,P⁻Y]`→MLP→tanh→×0.05·s(R_L)→⊕ |
| 底部 Training only | 3.7 | `L=0.9L_mean+0.1L_role+L_route` |

图注采用 nanobanana v4 README 的英文图注；正文注释"架构图为暂用版，待替换为 PPT 版"。

## 3. 详细大纲

### 3.1 Problem Setting: Forecasting under Changing Observation Support（≈320 词）

- **目的**：给出 P1–P4 所需的形式化对象，使后文每个模块的输入都有定义。
- **服务的问题**：P1（支持作为条件）、P2（零历史）、P3（加入/中断进展）、P4（布局）。
- **内容**：
  - 3.1.1 拓扑已知、信号归纳：静态图 `𝒢=(𝒱,A)`，N 个节点；已知节点 `𝒱_K` 与保留节点 `𝒱_V`、`𝒱_T`；训练只用 `𝒱_K` 的动态信号，验证/测试分别加入 `𝒱_V`/`𝒱_T`；标准化统计量只来自训练时段的 `𝒱_K`。
  - 3.1.2 观测支持：掩码 `M_{t,n}` 为三者之交（自然有效、暴露范围、生命周期允许）；角色 `R_t`（AAS/NAS/VS/FS one-hot）、状态年龄 `a_t`；`S_t=[R_t,a_t,M_t]∈ℝ^{N×6}`。给出"引言用语 ↔ 角色缩写"的对应。
  - 3.1.3 场景作为角色轨迹：Unseen（VS，年龄 0）、Steady（AAS，年龄 1）、Join（VS→NAS，接入后年龄 1/12…m/12）、Fault（AAS→FS，中断后年龄 1/12…m/12）；自然缺失只改变 M，不改变角色；评价时 Unseen/Join 采样保留节点，Steady/Fault 采样已知节点（补 I5）。阶段划分与配额留在 4.1。
  - 3.1.4 任务：`f(X⊙M, S, A, calendar) → Ŷ∈ℝ^{H×N}`，L=H=12；可见数 `q_n=Σ_t M_{t,n}` 只用于描述，不是额外输入。
- **来源**：MoGERNN [7]（加入/失效设定的先例）、SIML-MSIGN [10]（变化布局）作为差异参照；无新引用。
- **过渡**：既然支持随窗口变化，下一节给出把 `S` 送入各计算环节的总体结构。

### 3.2 Overview of ALFNet（≈150 词＋图1）

- **目的**：一句话数据流＋明确"借鉴什么、组织什么"。
- **内容**：SAHC 生成两分支共享的 X̄；ALFTemporal 从跨节点历史生成条件特征 Z，经 FiLM 调制 ALFGraph 的编码末状态；ALFGraph 以观测状态路由图传播并残差自回归解码；恢复参考值后经角色缩放图修正得到唯一输出。ALFNet 全称与"Adaptive 指输入条件化而非测试时更新参数"。说明 GRIN/BiGRIL、DCRNN/GRU、图专家混合、PatchSTG 分区、Crossformer 两阶段注意力、FiLM 均为既有算子，本文贡献在于以观测支持组织它们（沿用 v5 引言口径）。
- **来源**：[1] GRIN、[7] MoGERNN、[15] PatchSTG、[17] FiLM；**新增待查证**：DCRNN、GRU、Crossformer、稀疏/图专家混合。
- **过渡**：按数据流先介绍共享补全。

### 3.3 Support-Aware History Completion（≈230 词）

- **目的**：说明两个分支读取的是同一份补全历史，以及流量数据的固定输入处理。
- **服务的问题**：P7、P2（q=0 节点的来源）、P8。
- **内容**：
  - 对全部 N 节点、完整 12 步运行一次共享 BiGRIL，保留观测（F2）；角色与年龄作为 GRIN 辅助输入，掩码单独使用；无独立重建损失，补全器只经下游预测梯度训练。
  - 可选窗口插值（F3）：只覆盖 q>0 节点的缺失位置；内部线性，两端取最近可见值；q=0 节点保留 GRIN 的图消息补全。明确"不是 q=0/q>0 两条互斥支路"。
  - 可选参考中心化（F4）：`X̄=X^fill−αv`，`S` 不变；v 为窗口内最后可见标准化读数，无观测为 0；输出端对应加回（3.4.3）。
  - 同一 X̄ 送入两分支。不使用未来或被遮蔽的真值。
- **证据**：正文无消融；附录给出 w/o ALFRecon 与重建梯度路径对照（见 §5）。
- **来源**：[1] GRIN。
- **过渡**：X̄ 与 `S` 首先进入图分支的条件化传播。

### 3.4 Observation-Conditioned Graph Forecasting（ALFGraph，≈400 词）

- **目的**：给出条件化传播、图 GRU、残差自回归解码与参考恢复。
- **服务的问题**：P5（传播随支持调整）、P3（逐步角色轨迹）。
- **内容**：
  - 3.4.1 Lifecycle-Conditioned Graph Propagation（F5–F6）：路由只读 6 维 `S_t`，`w=softmax(S_t W_g)`；五个支持 identity、binary mean、weighted mean、forward/backward random walk，各有独立 `Θ_j`；dense softmax，不做 top-k 截断或加噪；披露 weighted mean 与 forward random walk 在非负图上为同一归一化形式，参数独立，因此不宣称五种数学上独立的算子。
  - 3.4.2 Graph GRU cell（F7）：编码输入 `x_t=[X̄_t,S_t]`；r、u、c 三门各用一个条件化传播模块；编码与解码各 2 层，每层每门独立路由，共 12 个；递推步共享参数。
  - 3.4.3 Residual autoregressive decoder 与 Restore（F8–F9）：`y_0=X̄_L`（中心化后的末步，不一定是最后一次真实观测）；`y_k=y_{k−1}+d·Linear(h_k)`；回送未修正的 `y_k`；解码路由固定使用 `S_L`，不建模未来角色演变；12 步后加回 αv 得到恢复序列。训练期按掩码教师强制（前 40 轮线性衰减，评价关闭），细节放 3.7 或附录。
- **证据**：表2 w/o Adaptive Routing 12/12 变差。
- **来源**：[7] MoGERNN；**待查证**：DCRNN、GRU、Shazeer et al. 2017 / GMoE（路由平衡项来源）。
- **过渡**：编码末状态在进入解码器前由时间分支调制，下一节给出调制参数的来源。

### 3.5 Spatial-Context Temporal Conditioning（ALFTemporal）与 FiLM（≈330 词）

- **目的**：说明跨节点上下文如何生成条件、在哪里注入、只注入一次。
- **服务的问题**：P6、P4（整窗编码保留位置信息）。
- **内容**：
  - 3.5.1 Temporal patch embedding（F10）：`[X̄,S,ToD/288,DoW/7]` 共 9 通道，kernel=stride=12 卷积，每节点 1 个 64 维 patch；拼接末步 ToD、DoW 嵌入与结构嵌入（各 32 维），得 160 维；结构嵌入由共享 MLP（7→64→32）作用于坐标、加权度等静态属性，不使用逐节点 ID 表（与归纳设置一致）。拼接顺序 patch|ToD|DoW|Struct。
  - 3.5.2 Partitioned Spatial Context Encoding（F11）：节点按布局分为 P=32 个分区、每区 C 槽位（速度 KD-tree 于坐标，流量为图上 component-aware BFS）；不足槽位以邻接排序的真实节点补足，只作上下文、不产生输出；一个两阶段注意力块：同槽位跨分区注意力，再经 4 个可学习 latent token 在分区内汇聚与回传（C→4→C），各阶段含残差、LayerNorm 与 MLP，单头；恢复原节点顺序得 Z∈ℝ^{N×160}。ALFTemporal 不产生独立预测。
  - 3.5.3 FiLM（F12）：`(Δγ,β)=MLP_{160→32→128}(Z)`，末层零初始化，初始为恒等调制；`H_mod=(1+Δγ)⊙H^{enc}+β`，同一参数广播到两层，在编码完成、解码开始前作用一次；没有 S 到 FiLM head 的直连。
- **证据**：表2 w/o Spatial Context、w/o ALFTemporal–FiLM 均 12/12；附录 FiLM γ/β 单项各 5/12。
- **来源**：[15] PatchSTG、[17] FiLM；**待查证**：Crossformer（两阶段注意力与 router token）。
- **过渡**：解码与恢复后，输出端还有一个只读末步角色的有界修正。

### 3.6 Role-Scaled Graph Refinement（≈130 词）

- **目的**：完整交代最后一个前向模块及其有限作用。
- **服务的问题**：P9（服务 P3）。
- **内容**（F13）：P⁺、P⁻ 为 `A+I` 及其转置的行归一化；三路拼接后 MLP（3→32→1，GELU，末层零初始化），tanh 有界，乘 0.05 与固定角色倍率 `s=[0,1,1,0.25]`（AAS/NAS/VS/FS）后残差相加；AAS 节点自身不修正，但作为邻居参与他人修正；只读末步角色，不读年龄或掩码，无独立路由；修正结果不回送解码器。一句话说明：该模块的作用随数据集变化，完整配对结果见附录（PEMS08 去除后 MAE/RMSE 更低）。
- **证据**：附录当前五项配对 10/12；历史 pointwise 对照 10/12。
- **过渡**：至此前向完整，下面是训练目标。

### 3.7 Training Objective（≈200 词）

- **目的**：给出实际优化的目标，并声明它不是独立贡献。
- **服务的问题**：P10。
- **内容**：
  - F14 元素损失 `ℓ(e)=(e/σ_data)·tanh(0.1e)`，e 为原单位绝对误差；小误差近似二次、大误差趋近标准化 MAE；权重不截断梯度；四数据集用同一系数，但 e 的量纲不同，因此不是跨量纲的同等加权。
  - F15 `L_mean`：对有效目标元素求均值；`L_role`：按历史末步角色分组求均值，再对本批实际出现的角色等权平均。
  - F16 `L_route`：每个路由模块 `0.01·(CV²(importance)+CV²(load))`，按递推调用与模块取均值；dense 路由下 load 项为 0。
  - F17 `L=0.9L_mean+0.1L_role+L_route`。无重建损失、无额外预测头。
- **证据**：无当前配置目标消融；附录 A.12 为历史配置，标注清楚。
- **来源**：**待查证**：Shazeer et al. 2017（CV² 平衡项）。

### 3.8 Dataset-Specific Settings（≈80 词）

- **目的**：说明四数据集共享同一可学习主干，差异只在固定设置。
- **内容**：C、窗口插值、d、α 为固定配置，不可学习、不经路由；数值见 4.1 设置表（C=7/16/10/6；插值 关/关/开/开；d=0.3/0.3/1/1；α=0/0/0/0.5）。选择过程与测试集参与历史在 4.1 与附录披露。
- **服务的问题**：P8。

## 4. 符号表（拟定，解决一符多义）

| 符号 | 含义 / 维度 | 与 v2 README 或架构图的差异 |
|---|---|---|
| `𝒢=(𝒱,A)`，N | 静态图（加权邻接 `A∈ℝ^{N×N}`），节点数 | 图中 `G` 端口改写为 `𝒢`，避免与传播算子 `G_*` 混淆 |
| `𝒱_K, 𝒱_V, 𝒱_T` | 已知、验证保留、测试保留节点 | 新增 |
| L, H | 历史长度 12、预测长度 12（标量，斜体） | `H` 不再表示隐藏状态 |
| `X∈ℝ^{L×N}` | 标准化后的可见历史（单通道） | 同 |
| `M∈{0,1}^{L×N}` | 观测掩码 | 同 |
| `R_t∈{0,1}^{N×4}`，`a_t∈[0,1]^N` | 角色 one-hot（AAS/NAS/VS/FS）、状态年龄 | 同 |
| `S_t=[R_t,a_t,M_t]∈ℝ^{N×6}` | 逐步生命周期上下文 | 同 |
| `q_n=Σ_t M_{t,n}` | 可见计数，仅描述 | 同 |
| m | Join/Fault 事件持续步数 | 第一阶段审计用 `e`，与误差 `e` 冲突，改为 m |
| `X^rec, X^grin, X^fill, X̄` | 重建、保留观测后、插值后、中心化后的历史 | 同 |
| `v∈ℝ^N`，α | 最后可见标准化读数（无观测为 0），中心化强度 | v5 第 4.1 节用 α 表示损失系数 0.1，统一改为直接写 0.1 |
| `𝐏_j, Θ_j, w_{t,j}`，j=1…5 | 传播支持（粗体矩阵）、投影、路由权重 | 与分区数 P（斜体标量）用字体区分 |
| `W_g∈ℝ^{6×5}` | 路由矩阵（无偏置） | 新增 |
| `G_r, G_u, G_c` | 三门的条件化传播 | 同 |
| `h_t^{(l)}`，`H^{enc}`，`H^{mod}` | 第 l 层隐藏状态（64 维）、两层编码末状态、调制后状态 | 架构图写 `H`/`H_mod`；正文用上标区分 |
| `y_k`（k=0…H），d | 解码步输出（`y_0=X̄_L`）、解码残差倍率 | 同 |
| `Ỹ` | 恢复参考值后的预测 `Y^{dec}+αv` | 架构图与 v2 写 `Y₀`，与 `y_0` 仅大小写不同；正文改用 `Ỹ`，PPT 重绘时同步（检查点 1 已确认） |
| `𝐞_n, 𝐳_n^0, Z` | 64 维 patch 嵌入、160 维拼接、条件特征 `Z∈ℝ^{N×160}` | 同 |
| P, C | 分区数 32、每区槽位容量 | 同 |
| `Δγ, β∈ℝ^{N×64}` | FiLM 参数 | 同 |
| `𝐏^+, 𝐏^-`，`s(·)` | 修正传播矩阵、固定角色倍率 | 同 |
| e，`σ_data` | 原单位绝对误差、训练标准差 | v5 写 s，与角色倍率 s(·) 冲突，改为 `σ_data`；`σ(·)` 只表示 sigmoid |
| η=0.1 | 角色平衡项权重 | 代码名 `forecast_role_balanced_beta`；正文不用 β，避免与 FiLM 的 β 冲突 |

## 5. 公式清单与核对状态

"已核"＝本轮在源码或最终 YAML 中逐项核对；"依文档"＝依 v2 README / 第一阶段审计（同 HEAD），阶段 2 写到该式前复读源码。

| # | 公式（拟用形式） | 代码 / 配置依据 | 状态 |
|---|---|---|---|
| F1 | `S_t=[R_t,a_t,M_t]`；四场景角色/年龄轨迹 | `lifecycle_roles.py:12-17`；`v4_lifecycle_loader.py:196`（cohort）、`:319`（轨迹） | 角色顺序与 cohort 已核；年龄递增规则依文档 |
| F2 | `X^grin = M⊙X + (1−M)⊙X^rec`，`X^rec=BiGRIL(X⊙M, M, [R,a], A)` | `ALFNet.py:384`；`source_grin.py:198,312,416` | 共享一次调用已核；BiGRIL 内部依文档 |
| F3 | `X^fill_{t,n} = X^grin_{t,n}`，若 `M_{t,n}=0` 且 `q_n>0` 且插值开启则取窗口内线性插值（端点取最近可见值） | `flow_support_transfer.py:9-31`；`flow_unified.py:36-37` | 已核 |
| F4 | `X̄ = X^fill − α·v` | `flow_unified.py:10-22`；`flow_support_transfer.py:28-30` | 已核（PEMS08 α=0.5，其他 0） |
| F5 | `w_t = softmax(S_t W_g)`（逐节点，dense） | `lifecycle_moge.py:269`（`_soft_gating`）；四份 graph YAML `moge_router_mode: soft` | 已核 |
| F6 | `G(U,S_t) = Σ_{j=1}^{5} w_{t,j} ⊙ (𝐏_j U Θ_j) + b` | `lifecycle_moge.py:445`（`LifecycleRoutedGraphConv.forward`）、`:61`（`_candidate_supports`） | 已核 |
| F7 | `r=σ(G_r([x,h],S))`，`u=σ(G_u([x,h],S))`，`c=tanh(G_c([x,r⊙h],S))`，`h'=u⊙h+(1−u)⊙c` | `DCRNNCell.py:60`；`GRUCell.py:30` | 三门工厂已核；GRU 更新式依文档 |
| F8 | `y_0=X̄_L`；`y_k=y_{k−1}+d·Linear(h_k^{(2)})`，k=1…12 | `ALFGraph.py:64-118（输出式 :113）, 331` | 已核 |
| F9 | `Ỹ = [y_1,…,y_H] + α·v` | `flow_unified.py:46`；`flow_support_transfer.py:54` | 已核 |
| F10 | `𝐞_n = Conv_{k=s=12}([X̄,S,ToD/288,DoW/7])_n ∈ℝ^{64}`；`𝐳_n^0=[𝐞_n‖E_tod‖E_dow‖φ(𝐠_n)]∈ℝ^{160}` | `ALFTemporal.py:188-201`；`patchstg.py:289-324`；`structural_node_encoder.py:345-403` | 维度配置已核；卷积与结构特征依文档 |
| F11 | 跨分区注意力（同槽位，沿 P）＋分区内 latent 汇聚/回传（C→4→C） | `patchstg.py:57-117,128,160,265`；patch YAML `layers:1`、`crossformer_router_factor:4` | 配置已核；latent token 是否按分区独立需复读 `patchstg.py` |
| F12 | `(Δγ,β)=MLP_{160→32→128}(Z)`；`H^{mod}=(1+Δγ)⊙H^{enc}+β` | `ALFNet.py:244-249,403-407`；`ALFGraph.py:315-330` | 已核 |
| F13 | `V=[Ỹ,𝐏^+Ỹ,𝐏^-Ỹ]`；`D=Linear_{32→1}(GELU(Linear_{3→32}(V)))`；`Ŷ=Ỹ+0.05·s(R_L)⊙tanh(D)` | `ALFNet.py:26-91,456`；四份 YAML 角色倍率 | 已核 |
| F14 | `ℓ(e)=(e/σ_data)·tanh(0.1e)` | `focal_loss.py:9-16` | 已核（代码为标准化误差形式，代数等价） |
| F15 | `L_mean`＝有效元素均值；`L_role`＝按末步角色分组均值的等权平均 | `v4_runner.py:379-436`；`v4_losses.py:25-38` | 已核 |
| F16 | `L_route = mean_modules mean_calls 0.01·(CV²(imp)+CV²(load))` | `lifecycle_moge.py:334-347`（router `forward`）；`ALFGraph.py` `pop_router_loss` | 已核 |
| F17 | `L = 0.9·L_mean + 0.1·L_role + L_route` | `v4_runner.py:434-439`；YAML `forecast_role_balanced_beta: 0.1` | 已核 |

## 6. 证据映射（方法中的每个陈述 → 文件）

| 方法中的陈述 | 证据文件 | 状态 |
|---|---|---|
| 四数据集参数量均为 1,037,529（含未用的 state encoder） | 四份 run_report.json（经 manifest 定位） | 已核；阶段 3 写入 `tables/provenance.json` |
| C=7/16/10/6；插值、d、α 取值；LR、WD | [manifest](../../pems08_wd_test_20261012/selected_configuration_manifest.json)；patch YAML `spa_patchsize` | 已核 |
| P=32 | 四份 `configs/paper/v4_data_manifests/*_run_manifest.json` `layout.recur_times=5` | 已核 |
| 去除条件化路由使四数据集三项 LifeAvg 均变差 | [four_dataset_results.json](../../pems08_wd_ablation_test_20261012/four_dataset_results.json) | 已核 |
| 跨节点上下文与时间条件分支的作用 | 同上 | 已核 |
| Refinement 作用随数据集变化 | 同上（w/o Graph Refinement 行） | 已核 |
| SAHC、FiLM γ/β、ExpertLifecycle 的历史干预 | [all_ablations_test_20261010/results_and_audit.json](../../all_ablations_test_20261010/results_and_audit.json) | 已核；须注明 PEMS08 行为旧 WD |
| 评价 cohort：Unseen/Join 保留节点，Steady/Fault 已知节点 | `data_provider/v4_lifecycle_loader.py:196`（cohort 字典） | 已核（源码） |

## 7. 需要新增并查证的引用（阶段 2 前运行 citation-check）

| 用途 | 文献（待查证，不得凭记忆写 DOI/页码） | 出现位置 |
|---|---|---|
| 图 GRU / 扩散卷积循环单元 | DCRNN（Li et al., ICLR 2018） | 3.2、3.4.2 |
| GRU | Cho et al., 2014 | 3.4.2 |
| 两阶段注意力与 router token | Crossformer（Zhang & Yan, ICLR 2023） | 3.2、3.5.2 |
| 专家路由与 CV² 平衡项 | Shazeer et al., 2017；GMoE（Wang et al., NeurIPS 2023，代码注释中的来源） | 3.4.1、3.7 |
| 归纳 support 命名近邻（相关工作恢复） | SC-SAMG（Sensors 2026，已在相关工作证据中） | 2.2 |
| Merlin（若 B1 决定写入） | 引用信息需查证 | 2.1 / 4.1 |

## 8. 小节之间的过渡逻辑

3.1 定义"支持如何变化" → 3.2 给出"支持在哪些环节被使用"的全景 → 3.3 先得到两分支共享的历史 → 3.4 用支持状态决定传播方式并解码 → 3.5 解释解码前的调制从何而来 → 3.6 输出端的有界修正 → 3.7 如何训练 → 3.8 四数据集仅在固定设置上不同，引出第 4 节实验设置。
