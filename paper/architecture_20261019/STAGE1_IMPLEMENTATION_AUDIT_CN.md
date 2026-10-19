# ALFNet 架构图第一阶段：实现核验与设计建议

日期：2026-10-19。状态：第一阶段完成；命名与布局是建议，尚未定稿或开始绘图。

## 1. 范围、身份与证据边界

- 按 WORKFLOW → PROJECT_CONTEXT → HANDOFF 索引 → `T-20260908-support-ablations/HANDOFF.md` 恢复上下文，读取最终项目索引、研究总结、v5 正文、实验稿及最终配置清单。最新研究交接是 2026-10-17 已完成状态，不恢复历史实验。
- 当前分支 `paper/final`；HEAD `e02b036d665e0c82f33afd4195b7a6ea8cad1c18`。相对本地 `origin/paper/final` 为 0/0，未 fetch。开始时没有 tracked/staged 修改；已有未跟踪目录 `paper/architecture_20261017/` 全部保留。
- 最终配置依据：[selected_configuration_manifest.json](../pems08_wd_test_20261012/selected_configuration_manifest.json)。逐份阅读四个外层 YAML 及其八个专家 YAML，不能用 `configs/ALFNet/` 默认配置替代。
- Windows 工作副本的四个 YAML 原始文件 SHA256 与清单不同，但仅在内存中将 CRLF 统一为 LF 后，四份 SHA256 全部与清单吻合。没有改写文件或清单。
- 20 个相关源码文件与 `paper/final_figures_20261017/plan.json` 的源码指纹比较：统一为 LF 后 19 个一致；`experiments/v4_runner.py` 不一致。当前提交相对父提交的该文件差异是删除退役基线注册、对应注释和 GWNET 梯度裁剪分支；所读 ALFNet 的 `compute_losses` 路径仍如本文。未完整恢复冻结运行环境，不宣称整个旧运行计划通过哈希检查。
- 四份最终 checkpoint 在本地清单路径均不存在。因此本轮是源码、配置、冻结记录及旧图的静态核验，未加载权重、未运行模型 forward、未重新数值验证。
- 没有训练、消融、GPU 工作、原始数据/test 读取、结果修改、分支切换、commit 或 push。没有执行 Python；仓库记录的已验证环境是服务器 `/home/xd/anaconda3/envs/ReaCH/bin/python`，本地未验证，不擅自替换环境。
- 本轮只新增本说明。两份论文稿和历史绘图目录均不改动。

## 2. 冻结配置与公共维度

记批量为 B，历史长度 L=12，预测长度 H=12，节点数 N；信号通道为 1。输入在训练统计标准化空间中表示。

| 数据集 | N | 分区数 P | 分区容量 C | 插值 | 解码残差倍率 d | 中心化强度 c | 最终运行适配 |
|---|---:|---:|---:|---|---:|---:|---|
| METR-LA | 207 | 32 | 7 | 关闭 | 0.3 | 0 | 原生 ALFNet |
| PEMS-BAY | 325 | 32 | 16 | 关闭 | 0.3 | 0 | 原生 ALFNet |
| PEMS04 | 307 | 32 | 10 | 开启 | 1 | 0 | `flow_support_transfer`，`bridge_delta1` |
| PEMS08 | 170 | 32 | 6 | 开启 | 1 | 0.5 | `flow_unified`，`triple_halfcenter` |

分区数的依据是四份 `configs/paper/v4_data_manifests/*_run_manifest.json` 的 `recur_times=5`，`lib/runtime_context.py:306` 调用统一布局解析，`lib/graph_native_layout.py:110,144,161` 分出 2^5=32 块。模型最终使用 `len(runtime_context.layout_parts)`。这里没有读取 payload 重建布局。

速度数据采用已验证坐标的 KD-tree；流量数据采用图结构的 component-aware BFS。原分区最大节点数为 7/11/10/6；最终槽位容量为 7/16/10/6，PEMS-BAY 尤其不能写成 16 个分区。补足后的 token 数分别为 224/512/320/192。容量不足直接报错；不足容量的分区用邻接排序的真实节点索引补足，不是零节点或新的可训练节点。

共同设置：图隐藏宽度 64，编码和解码各 2 层；GRIN 双方向各 1 层、隐藏宽度 64；时间分支 1 个长度为 12 的 temporal patch、1 个交互块、1 个 attention head、4 个潜在交互 token。时间分支维度为 64+32+32+32=160。

图专家 YAML 中的 `decoder_residual_scale=0.3` 不是 flow 的最终运行值，适配器会覆盖它；专家 YAML 中的 `loss=mse` 也不是外层 ALFNet 的最终训练目标。`scripts/run_flow_ofat_20261006.py:103` 和 `scripts/final_figures_20261017.py:50` 等运行入口显式安装相应适配器。

## 3. 模块—输入—输出—作用—代码依据

下表代码路径相对仓库根目录；行号对应本次 HEAD。

| 模块 | 输入 | 输出 | 实际作用 | 代码依据 |
|---|---|---|---|---|
| 输入装配 | 可见历史 Xvis `[B,12,N,1]`；上下文 S `[B,12,N,6]`；A `[N,N]`；日历 `[B,12,3]` | 拼接输入 `[B,12,N,7]` | 全 N 节点轴；不可见信号在标准化空间置 0 | `args.py:152`；`data_provider/v4_lifecycle_loader.py:239`；`lib/v4_batch.py:70`；`model/ALFNet/ALFNet.py:321` |
| GRIN/BiGRIL 共享重建 | Xvis、M、A，辅助特征 `[R,a]` 共 5 维 | 保留观测的重建信号 `[B,12,N,1]`，并原样带回 S | 一次调用服务两个分支；双向历史递归、空间重建、融合 | `layers/source_grin.py:198,244,312,338,416`；`model/ALFNet/ALFNet.py:368` |
| 观测支持处理 | GRIN 输出、原始 Xvis/M、数据集开关 | 共享处理后历史 Xbar `[B,12,N,1]` | 可选缺失位置插值覆盖；可选减去最后可见参考值的 c 倍 | `experiments/flow_support_transfer.py:9,31,40`；`experiments/flow_unified.py:10,29` |
| 图编码器 | `[Xbar,S]`、静态 A；逐历史步 S_t | Henc `[2,B,N,64]` | 两层生命周期条件化图循环，递推 12 步 | `model/ALFGraph/ALFGraph.py:16,294` |
| 生命周期条件化图传播 | 门输入 U `[B,N,F]`；S_t `[B,N,6]`；5 个 supports | `[B,N,64]` | 五个带独立线性参数的传播结果，以逐节点 softmax 权重求和；用于 GRU 三个门 | `layers/lifecycle_moge.py:61,269,334,445`；`layers/DCRNNCell.py:60`；`layers/GRUCell.py:30` |
| 历史/日历嵌入 | `[Xbar,S]` 7 通道 + 2 个归一化日历通道 | `[B,1,N,64]` | kernel/stride=12 的时间卷积压缩整段历史 | `model/ALFTemporal/ALFTemporal.py:188,201`；`model/ALFTemporal/patchstg.py:230,289` |
| 结构节点编码 | 静态 A 与可用坐标产生 `[N,7]` 特征 | `[N,32]` | 所有节点共用 7→64→32 MLP；没有逐节点可训练 ID 表 | `layers/structural_node_encoder.py:345,352,403` |
| 分区空间上下文交互 | 历史 64 + ToD 32 + DoW 32 + 结构 32 | 重排 `[B,1,P,C,160]`，交互并还原后 Z `[B,N,160]` | 先跨分区自注意力，再在每分区内做潜在 token 汇聚与回传 | `model/ALFTemporal/ALFTemporal.py:15,27,105`；`model/ALFTemporal/patchstg.py:89,160,265` |
| FiLM head / 调制 | Z `[B,N,160]`；Henc | Δγ、β 各 `[B,N,64]`；Hmod `[2,B,N,64]` | 160→32→128 MLP；同一组参数广播调制两层编码末状态 | `model/ALFNet/ALFNet.py:244,403`；`model/ALFGraph/ALFGraph.py:322` |
| 残差自回归解码器 | Hmod；Xbar 的最后时刻；S_L；静态 supports | Ydec `[B,12,N,1]` | 两层图循环；前一输出作下一步 anchor；末步生命周期保持不变 | `model/ALFGraph/ALFGraph.py:64,331` |
| 参考值恢复 | Ydec 与 c×最后可见值 | Y0 `[B,12,N,1]` | 解码后恢复中心，随后进入图修正 | `experiments/flow_unified.py:46`；`experiments/flow_support_transfer.py:54` |
| 图残差修正 | Y0、A、末步角色 R_L | Yhat `[B,12,N,1]` | 拼接自身/正向/反向预测，MLP+tanh，固定角色倍率缩放后残差相加 | `model/ALFNet/ALFNet.py:26,72,456` |
| 训练目标 | Yhat、合法目标及 loss mask、R_L、router 权重 | 标量 loss | 误差幅度加权预测目标的均值/角色均衡混合，加 router 平衡项 | `experiments/focal_loss.py:9`；`experiments/v4_losses.py:25`；`experiments/v4_runner.py:379` |

统一前向：输入 → 一次共享 GRIN → 可选插值覆盖 → 可选中心化 → 同一历史分给 ALFTemporal 与 ALFGraph；ALFTemporal 的 Z 经 FiLM head 调制图编码末状态 → 图残差解码 → 参考值恢复 → 图残差修正 → 单一预测。

“并行分支”表示依赖关系：源码先计算时间分支得到 FiLM 参数，再执行图专家；不表示实际并发执行。两个分支共享输入张量和重建器，没有共享各自的主干权重。

## 4. Lifecycle Context：场景与逐时刻角色分层表达

`S_t=[R_t,a_t,M_t]`，维度 6；R 的固定代码通道顺序为 AAS/NAS/VS/FS，对应 0/1/2/3。S 是显式描述，不是一个统一的可学习状态编码器输出。

| 论文观测场景 | 目标节点的窗口内角色轨迹 | 年龄轨迹 | 观测掩码 |
|---|---|---|---|
| Unseen | VS 全窗口 | 0 | 全 0 |
| Steady | AAS 全窗口 | 1 | 仍受自然缺失约束 |
| Join | 前段 VS → 后段 NAS；e=12 时全 NAS | 前段 0，后段 1/12…e/12 | 前段 0，后段还需自然观测有效 |
| Fault | 前段 AAS → 后段 FS；e=12 时全 FS | 前段 1，后段 1/12…e/12 | 前段受自然缺失约束，后段 0 |

依据：`data_provider/lifecycle_roles.py:12`、`v4_lifecycle_loader.py:204,319`。e 对 Join 是接入时长，对 Fault 是中断时长；不等于两者的有效可见读数数目。自然缺失只改变 M，不把 AAS 自动变成 FS。M 是自然有效性、暴露范围和生命周期允许性三者的交集。

推荐绘图：左侧 Lifecycle Context 内，上方用四条短时间轨迹突出 **Unseen / Steady / Join / Fault**，下方只画一次 `R_t (role), a_t (state age), M_t (observed)`。R 的图例可用 Active / Newly joined / Unobserved / Failed，并在图注或符号表明确固定编码对应 AAS/NAS/VS/FS。不要将四个场景名称直接填进 one-hot 的四个通道。年龄应画成历史序列 `a∈R^(L×N)`，不是仅有末时刻的节点向量。

Unseen 是零历史观测条件；训练中在已知节点上模拟，验证/测试的该轨迹使用各自保留节点。因此也不能把 VS 角色直接等同“该节点从未参与训练”。上述呈现调整无需修改代码语义。

## 5. Visibility、共享重建、插值与中心化

### Visibility 的结论

没有独立、有效的 Visibility 预测分支，也没有额外输入通道。

- Observation Mask M 逐时刻参与 GRIN、ALFGraph 输入和 router、ALFTemporal 嵌入，确实有效。
- `ALFNet._state` 在 `ALFNet.py:300` 计算 `mean_t(M)`，与末角色/末年龄送入 `state_encoder`，但 `forward:452` 以 `_, last_availability` 丢弃该编码输出。不能把此 MLP 或可见比例画成有效预测模块。
- `flow_unified.reference_and_scale` 也计算可见比例；最终 `triple_halfcenter` 不走 `spatial` 或 `qdelta` 分支，该比例不控制最终中心或残差倍率。
- 当前 v5 无独立 Visibility 模块描述；实验稿“当前可见性之外”的“可见性”是自然语言，建议后续具体写成“逐时刻观测掩码所描述的可用性”，避免误读为新特征。附录已经说明 `state_encoder` 输出不参与预测，不需要删历史材料。

### 正确补全顺序

1. 对整个 batch、全部 N 节点、完整 12 步调用同一个 BiGRIL，不能按 q=0/q>0 将 GRIN 分成只对零历史执行的支路。
2. 每个方向的 GRIL 先由上一隐藏状态给出第一阶段估计，保留真观测；再用空间 decoder 产生第二阶段估计，再保留真观测并更新图 GRU。前后向是沿历史正序/逆序的两套递归，不是“前向 encoder → 后向 encoder”的串联。
3. 将对齐后的前后向表示与 M 通过融合 MLP 生成重建；包装器再次用 M 保留可见原值，得到 Xgrin。角色与年龄作为 GRIL 辅助输入 u，M 单独使用。
4. 仅当数据集启用插值时，对于至少有一次可见读数的节点，只覆盖缺失位置。内部缺口线性插值，窗口边界取最近可见端点。没有 spline。
5. q=0 节点保留 GRIN 重建，利用静态图上的空间消息及图循环状态；没有最终配置下单独的邻居均值中心补全。
6. 之后才做可选中心化。定义 v 为窗口内最后可见的标准化值，无观测时 v=0。Xbar=Xfilled−c·v，S 原样保留。PEMS08 c=0.5，其他 c=0。
7. 若训练启用 teacher forcing，合法 teacher 值同步减去相同中心；解码输出加回 c·v，随后进入 Graph Refinement。

因此“保留真实观测”指中心化前的补全步骤。中心化后的可见数值也会平移，图上不可再声称它与原始 Xvis 数值完全相同。参考值恢复也不是标准化逆变换；模型内部预测仍在标准化空间，原单位显示使用训练 scaler。

## 6. ALFGraph、FiLM 与图预测修正

### 图循环和五个传播专家

编码器和解码器都取 `configs.e_layers=2`，不是按 `d_layers=1` 画一个解码层。每层有三个独立图传播模块：reset r、update u、candidate c。共 2×2×3=12 个 router 实例，各自有权重，时间递推中复用；编码和解码不共享这些权重。

对每个门输入 U：

`w_t = softmax(S_t W_gate)`

`G(U,S_t) = Σ_(j=1..5) w_(t,j) ⊙ (P_j U Θ_j) + b`

五个真实槽位为 Identity、Mean、Weighted Mean、Forward Diffusion、Backward Diffusion。Mean 使用正边的二值邻接行归一化；Weighted Mean 使用加权邻接行归一化；Forward/Backward 分别使用 A/Aᵀ 的随机游走行归一化。

Weighted Mean 与 Forward Diffusion 是同一行归一化传播形式，在正常非负图行上重合；两份实现的 epsilon/浮点计算细节略有区别，不据此宣称不同传播几何。它们仍具有各自独立 Θ 和 routing 权重。保留五个专家槽位，在说明中披露矩阵重合，不能改画成四专家，也不能宣称五种数学上独立的算子。

最终全部采用 `moge_router_mode=soft`：无 top-k 截断、无噪声；YAML 中遗留的 top_k=1/noisy_gating=true 不激活对应路径。router 直接读 6 维 S，经线性矩阵乘再 softmax，不是读取信号/隐藏状态的多层 MLP。图传播的 U 才包含信号与隐藏状态。

GRU 更新为 `r=σ(G_r([x,h],S))`、`u=σ(G_u([x,h],S))`、`c=tanh(G_c([x,r⊙h],S))`、`h'=u⊙h+(1−u)⊙c`。只需在图内嵌入一个代表性 cell 放大框，注明各层/门独立参数；不要画一个共享 router 同时控制全网络。

编码使用 S_1…S_12；解码固定使用 S_12，未建模未来角色演变。

### FiLM 与解码

`(Δγ,β)=MLP(Z)`，MLP 为 160→32→128，最后投影零初始化。`Hmod=(1+Δγ)⊙Henc+β`，同一组 γ/β 广播到两层。

注入一次，位于编码完成与解码开始之间；不是每个预测步重新生成调制，也没有直接 `S→FiLM head` 的额外连接。S 经时间分支影响 Z 是间接路径。

初始 decoder 输入为共享处理后历史的最后时刻，不保证就是最后一次真实观测。每步 `y_k=anchor+d·Linear(h_k)`；推理用 y_k 作为下步输入。图修正发生于完整 12 步解码之后，其修正结果不反馈给自回归解码器。

### Graph Refinement 完整路径

令 P+、P− 为 A+I 与其转置分别行归一化的矩阵。最终 diffusion_steps=1、use_backward=true、add_self_loop=true：

`V = concat(Y0, P+Y0, P−Y0)`，维度 `[B,12,N,3]`；

`D = Linear_(32→1)(GELU(Linear_(3→32)(V)))`；

`Yhat = Y0 + 0.05 · s(R_L) · tanh(D)`。

角色倍率按 AAS/NAS/VS/FS 为 `[0,1,1,0.25]`，不是可学习的场景门。AAS 节点的最终修正量为零，但仍可作为邻居预测参与其他节点修正。最后投影零初始化。此模块只读末角色作为倍率，不直接读年龄或 mask，也没有这里专属的 lifecycle router。

## 7. ALFTemporal 的真实轴和顺序

1. 历史日历取时间戳的 minute/hour/day-of-week：ToD=12·hour+floor(minute/5)，范围 0…287；DoW 为 0…6。
2. `[Xbar,R,a,M,ToD/288,DoW/7]` 共 9 通道，经 12 步卷积得 `[B,1,N,64]`。因此每个节点只有 **1 个 temporal patch**，不能绘制多个时间窗口 token。
3. 拼接历史末时刻 ToD embedding 32 维、DoW embedding 32 维、结构 embedding 32 维，得到 `[B,1,N,160]`。日历既以归一化数值参与卷积，又以 embedding 参与拼接；这里是 concat，不是三项相加。
4. 结构特征为坐标 2 维、入/出/总加权度 3 维、log 总度 1 维、孤立标志 1 维。去除自环后计算度；连续特征按去均值/最大绝对偏差缩放。无坐标时两个坐标通道为零，不伪造 GPS。共享 MLP 7→64→32。
5. 按布局重排并补足，形成 `[B,1,P,C,160]`；传给 TSA 的轴为 `[B×Tpatch,C,P,160]`。
6. TSA 的 `time_attention` 在此适配中沿 **P 个空间分区** 计算；每个槽位索引各自跨分区交互。名称中的 time 不能直接沿用为图中的时间注意力。
7. 然后每个分区以 4 个可训练 latent tokens 为 query 汇聚 C 个槽位（sender）；再由 C 个槽位读取 latent buffer（receiver）。这是分区内经瓶颈 token 的交互，不是直接 C×C 的自注意力。各阶段有残差、LayerNorm 和 MLP。
8. 共一个 `WindowAttBlock`，不是多层交替循环。`recur_times=5` 是布局递归深度；`layers=1` 才控制交互块数。
9. `reo_parts_idx` 选回原节点位置，按 `ori_parts_idx` scatter 回原顺序；补足的上下文槽位不另输出预测。最后得到 Z `[B,N,160]`，无独立 forecast head。

绘图必须区分“图传播 lifecycle router 的 5 个专家权重”与“时间分支 attention 的 4 个 latent tokens”；二者没有共享参数或同一含义。

## 8. 仅训练期间使用的部分

最终目标名称是 `mae_focal_raw02`。设 e 是原单位绝对误差、σ 是训练标准差，元素损失为 `ell(e)=(e/σ)·tanh(0.1e)`；权重不 detach。

`L = 0.9 L_mean + 0.1 L_role + L_route`。

- L_mean 对合法、有限的预测目标元素求均值；不能严格称为“先逐样本平均”，因为代码对所有有效元素直接取均值。
- L_role 按历史末步角色划分有效目标，先各组求均值，再对当前 batch 中实际存在的角色等权平均；不是将整段 Join/Fault 轨迹当作固定 one-hot，也不是对空角色补零后总除以四。
- 每个 router 的辅助项为 `0.01·(CV²(importance)+CV²(load))`，先按递推调用求均值，再对 12 个传播模块求均值。dense soft 下所有槽位被使用，load 项为零，importance 项仍有作用。
- 没有独立 GRIN 重建损失、频谱损失、历史线性修正目标或额外监督头。补全器沿被下游使用的重建值接收预测梯度。
- Masked teacher forcing 仅训练使用，外部生成的 mask 受 `decoder_available_mask` 限制；前 40 轮线性衰减，评价关闭。图中只能用外围训练注记表示，不能把真实未来连入推理数据流。
- router 及其 softmax 在推理中仍有效；仅辅助 loss 的优化用途属于训练。实现可能在评价 forward 中也计算辅助标量，但它不参与生成预测。

## 9. 旧图的问题与可复用部分

已直接查看 `v1_20261018/alfnet_0.png`、`alfnet_1.png`，以及 `Reference_Image/1.png` 至 `7.png`。旧 SVG/生成脚本与 README 仅作为参考，不把旧验证声明当作本轮验证。

| 对象 | 问题 | 推荐处理 |
|---|---|---|
| alfnet_0 | 输入/输出用多时刻路网，未满足曲线要求；Observation Mask 重复出现 | 左侧只保留一组有断点的历史曲线和一次 context，输出只画 12 步曲线 |
| alfnet_0 | 上部已展开 FiLM、下部再次展开；时间分支只画泛化的堆叠块；GRU 与 refiner 内部不足 | 合并为单图，在原位置展开关键机制 |
| alfnet_0 | 结构节点 embedding 的箭头像先送入历史嵌入，再连主路径；未清晰表示真实 concat | 将历史卷积、末日历 embedding、结构 MLP 连接到同一 concat |
| alfnet_1 | 把 Unseen/Steady/Join/Fault 写成 Lifecycle role 的四个通道 | 场景轨迹与逐时刻角色分层表达 |
| alfnet_1 | q>0 走 interpolation、q=0 走 GRIN 的互斥图，违背实际执行顺序 | 全输入 GRIN → 可选覆盖 → 可选中心化 |
| alfnet_1 | 标注 linear/spline | 仅 linear，边界 nearest endpoint |
| alfnet_1 | 双向重建被画成 forward encoder→backward encoder 的串联 | 两条相反时间方向的并列递归，输出对齐后融合 |
| alfnet_1 | 时间支路的堆叠块、×L_t、alternating 暗示多层；未准确显示 4 latent tokens | 一个跨分区注意力阶段 + 一个分区内汇聚/回传阶段 |
| alfnet_1 | router 画 MLP，FiLM 参数标 N×d_t，易与 Z 的160维混淆 | router 写 Linear→Softmax；γ/β 标 N×64 |
| alfnet_1 | 紫色条件线直接接调制 head；refiner 只画图标 | S 接时间嵌入，Z 接 head；refiner 展示三路拼接、MLP、tanh、角色缩放和残差 |
| alfnet_1 | 上面完整总体结构，下面再次完整展开重建/图分支/时间分支，重复输入和输出 | 不保留上下两套完整架构；局部放大只出现一次代表性 cell |
| 两图 | 用节点向量标年龄，未突出随时间变化；“五专家”易被误读为五种不同矩阵 | 统一符号和轴，说明角色轨迹及矩阵重合 |
| 旧目录 README | 链接若干当前本地不存在的预览/PDF/风格稿；交付清单与本地文件不完全一致 | 后续以实际存在并重新核验的交付清单为准，保留旧文档 |

参考图初步取舍：5 的浅色圆角容器、正交连线与清晰层级最适合整体；4 的日历/特征矩阵/concat 表达适合时间支路；3 的路网节点与圈加/乘图元适合图传播；7 的嵌套区域可借鉴。1/2/6 主要是动机解释图，不适合作为本图的完整布局。参考中手写字体、密集装饰、多重边框不直接沿用；统一正式字体和细线。

## 10. 推荐命名表（待用户定稿）

ALFGraph、ALFTemporal 建议保留为已有模块标识，后面的英文是功能性副标题，不另造 OCGF/SCTC 等缩写，也不声称它们是这些副标题的首字母缩写。

| 英文全称/图中标签 | 缩写或标识 | 中文 | 功能定位与代码依据 | 与原方法的关系/命名意见 |
|---|---|---|---|---|
| Adaptive Lifecycle Forecasting Network | ALFNet | 自适应生命周期预测网络 | 整体组织；`model/ALFNet/ALFNet.py` | 可保留候选；Adaptive 指输入条件化，不暗示在线更新权重 |
| Support-Aware History Completion | SAHC | 观测支持感知历史补全 | GRIN、观测保留、可选覆盖/中心化、两分支共享；`source_grin.py` + 两个 flow adapter | 可作为组合模块名；必须注明 GRIN/BiGRIL 是借鉴基础，不称新提出的双向重建算法；support 在此指 observation support |
| Observation-Conditioned Graph Forecasting | ALFGraph | 观测条件化图预测 | 图编码、条件传播、残差解码；`ALFGraph.py` | 比只叫 graph reasoning 更完整；DCRNN/GRU 是沿用基础结构。外围 refiner 在代码中属于 ALFNet，图中若包在同一蓝色大区域需说明是功能分区 |
| Spatial-Context Temporal Conditioning | ALFTemporal | 空间上下文时间条件分支 | 整窗历史嵌入、空间交互、Z；`ALFTemporal.py`、`patchstg.py` | 不宣称时间自注意力或独立预测器；借鉴 PatchSTG 分区组织、Crossformer TSA 计算结构 |
| Lifecycle-Conditioned Graph Propagation | 不新增缩写 | 生命周期条件化图传播 | S→Linear→softmax→5 expert slots；`lifecycle_moge.py` | 清楚表达自身条件化方式；保留图专家混合与经典传播来源 |
| Partitioned Spatial Context Encoding | 不新增缩写 | 分区空间上下文编码 | 跨分区注意力 + 分区内 latent 汇聚/回传；`patchstg.py:89,160` | 比 PTCE 更准确地强调这里的注意力轴；时间信息已由整窗卷积编码。不将适配后的 TSA 换名声称全新算子 |
| FiLM Parameter Head | FiLM head | FiLM 参数生成头 | Z→Δγ,β；`ALFNet.py:244` | 比 TMH 更直接、少一个缩写；FiLM 保留经典全称 Feature-wise Linear Modulation |
| Role-Scaled Graph Refinement | 不新增缩写 | 角色缩放图修正 | 固定末角色倍率×图残差；`ALFNet.py:26` | 比 Lifecycle-Aware Graph Residual Refinement 更精确；不暗示年龄/mask 或新的 router 参与，建议不用 LGRR |

小型操作直接标注 History Embedding、Structural Node Embedding、Cross-partition Attention、Within-partition Latent Aggregation/Broadcast、Restore Reference。无需给每个 Norm、MLP、concat 起新名。

## 11. 两种单图布局方向

**优先方案 A：左侧共享输入与补全，右侧上下分支。** 左侧输入约占宽度 15%，浅红 SAHC 约 22%，其余留给上下分支和输出；这些是初始空间预算，第二阶段根据文字密度调整。

- 左列上方只画一次有实线观测和缺口的历史曲线，中间四种观测场景轨迹及 R/a/M，下方静态路网与邻接。日历由历史时间戳引出；坐标作为静态结构可选属性说明。
- SAHC 内顺序为 BiGRIL 双方向→观测保留→可选插值覆盖→可选参考中心化；右端仅一个 Filled History 分叉点。保留参考值 v 到 Restore Reference 的细连接。
- 上部绿色 ALFTemporal 依次画历史卷积/日历/结构 concat，P×C 分区，跨分区交互，再分区内 4 latent tokens 的汇聚/回传，恢复节点序，Z，FiLM head。
- 下部蓝色图路径为两层 Encoder→黄色 FiLM→两层 Residual AR Decoder→Restore Reference→浅橙 Graph Refinement→右端唯一 Forecast。FiLM 参数短线向下，避免贯穿整幅图。
- 一个代表性的图 GRU cell 与五专家混合放大框嵌在蓝色区域内部，下方靠近 Encoder/Decoder；它展示内部机制，不重复整条前向。
- 紫色条件流只分接实际使用处，注 `S_t / S_L / R_L`；静态图用底部细线分发，避免多次复制完整路网。
- 训练目标占图底一条紧凑区域，与推理主线分离。图例统一条件线、数据线、concat、逐元素运算。

**备选方案 B：共享补全居左中，时间分支居中上，图编码与解码分列下方，FiLM 居中桥接。** 更有利于放大图 cell，但补全向两分支的连线较长，静态图与上下文路由容易交叉。若 A 的文字在通栏宽度下仍可读，不选 B。

两方案均不使用 (a)(b)(c)(d) 和两套总架构。最终缩至论文双栏通栏时检查字体、边界与箭头，优先减说明句而不是缩字号。后续 SVG library 应提供可组合的真正矢量图元；本阶段未创建任何正式图或素材库。

## 12. 论文表述对齐与后续边界

- v5 目前含第1、2、4节，实验稿为第4节；没有第3节方法正文，不能声称图已与全部方法公式逐项一致。
- 引言的共享重建、图传播条件化、时间支路 FiLM、单一预测与实际主干一致；后续应把 FiLM 注入位置精确到编码末状态。
- “无可见历史回退到重建”可描述最终数值来源，但正式方法要补一句“所有输入先统一执行 GRIN，再按配置覆盖缺失位置”，避免推导出错误计算分支。
- 训练“样本均值项”建议改“有效目标元素均值项”，“模型辅助损失”明确为 router 平衡项。
- 第4节消融省略 Graph Refinement 不是删除模型结构的依据；图中保留。
- 实验稿 4.6 的 PEMS08 Recent/Early 数值仍是 22.444/27.888，而 v5 为 22.822/28.404；这是两稿现有文字差异，超出本轮架构图范围，仅记录，不修改或据此重评结果。
- 本轮未发现必须修改模型才能实现四类场景命名的需求。场景层与角色层区分即可满足准确表达。

英文图注工作稿（供第二阶段压缩，不是最终交付）：

> Overview of ALFNet under changing observation support. A shared GRIN-based completion module preserves observed values and supplies both branches with the same processed history, with optional visible-window interpolation and reference centering. ALFTemporal encodes the history and spatial partition context into node-wise conditioning features, which generate FiLM parameters for the graph encoder states. ALFGraph performs lifecycle-conditioned propagation and residual autoregressive decoding. Reference restoration, when enabled, precedes role-scaled graph refinement and the final 12-step forecast. Lifecycle roles vary within the history window; Unseen, Steady, Join, and Fault denote observation scenarios. Forecast and router-balancing objectives are used for training only.

## 13. 本阶段验证与交接状态

- Facts / Completed：4 外层配置、8 专家配置、指定前向模块及必要依赖、数据角色构造、损失、运行适配和旧图已核对；7 张参考图已查看；映射/命名/布局建议如上。
- Hypotheses：布局 A 的可读性是待排版验证的设计判断，不是已完成的视觉验收；没有新增科学效果主张。
- Modified but unverified：仅本说明新增；checkpoint/数值 forward、PPT 导入编辑性、未来图像的小尺寸可读性未验证。冻结 runner 的完整源码身份未恢复。
- Not started：第二阶段正式设计比较、第三阶段绘制、第四阶段 SVG/PNG 视觉验收与交付。按用户当前要求停在第一阶段。
- Ruled out：修改模型/配置/实验结果、恢复废弃机制、重训、测试集评估、直接发布旧 PNG、commit/push。
- 实际检查方式：PowerShell 文件读取/JSON 解析/SHA256，rg，git status/log/diff/rev-list，view_image，apply_patch；文档完成后检查路径和空白。没有把历史测试记录作为本轮测试结果。
- 收尾检查：`git diff --check` 通过；新增说明的 Markdown 文件链接无失效项、无行尾空白；tracked/staged diff 仍为空。工作区仅原有 `paper/architecture_20261017/` 与本轮新建 `paper/architecture_20261019/` 为未跟踪目录。没有执行模型 lint/typecheck/test，此次没有代码修改。
