# ALFNet 正式架构图与可编辑模块库

日期：2026-10-19。采用第二阶段推荐的方案 A；已完成整体布局、内部机制、排版和静态实现核验。只新增本目录文件，没有修改模型、配置、论文正文或实验结果。

## 交付文件

- [最终架构 PNG](alfnet_architecture.png)：6000 × 3960，白底。
- [SVG 模块库总览](module_library.svg)：19 个实际使用的模块，全部由可编辑文本、矩形、圆、折线和箭头组成。
- [独立 SVG 模块目录](module_library/)：每个模块单独保存，透明外部背景，适合分别插入或重组。
- [核验记录](VERIFICATION_CN.md)：检查项、修复内容、证据及核验边界。

`_source/` 保存可复现源稿与生成脚本；`qa/` 保存布局迭代、缩小预览及机器检查记录，均不是另外一套论文图。历史 `architecture_20261017/` 完整保留。

图幅 2000 × 1320 设计单位。按 180 mm 通栏宽度，实际文本字号约 7.1–7.9 pt。建议以通栏方式排版，图注放在正文中；不要强行缩为单栏。

## 英文图注

> **Architecture of ALFNet under changing observation support.** A shared GRIN/BiGRIL-based module reconstructs the full history, preserves observed values, and applies dataset-specific interpolation and reference centering before supplying both branches with the same processed history. ALFTemporal forms one temporal patch per node, performs cross-partition attention followed by within-partition aggregation and broadcast through four latent tokens, and restores the original node order. Its node-wise features generate FiLM parameters that modulate the terminal states of both graph-encoder layers. ALFGraph uses lifecycle-conditioned dense mixtures of five propagation experts in its recurrent cells and produces a 12-step forecast through a two-layer residual autoregressive decoder. Reference restoration precedes role-scaled graph refinement. Unseen, Steady, Join, and Fault denote observation scenarios; their time-varying roles, state ages, and observation masks constitute the lifecycle context. Purple dashed arrows indicate lifecycle conditioning, and named ports reference existing inputs. The footer describes training-only objectives. Batch dimensions are omitted; curves and matrix entries are schematic.

需要在方法正文或图注后紧接说明的细节：`P=32`，`C` 随数据集变化；WM 与 Fwd 的归一化传播形式重合，但具有独立的投影与路由参数。图中 `G` 端口指唯一静态图及其可用节点属性；`A` 仅指邻接矩阵。没有额外 Visibility 分支。

## 名称与缩写

| 英文全称 / 图中标识 | 缩写 | 中文 | 定位及来源 |
|---|---|---|---|
| Adaptive Lifecycle Forecasting Network | ALFNet | 自适应生命周期预测网络 | 完整组合模型；Adaptive 指输入条件化，并非测试时更新参数 |
| Support-Aware History Completion | SAHC，整图未额外重复缩写 | 观测支持感知历史补全 | GRIN 基础重建、保留观测、支持处理及两个分支共享的组织方式 |
| Observation-Conditioned Graph Forecasting | ALFGraph 为既有标识 | 观测条件化图预测 | 两层图编码及残差自回归解码；沿用 DCRNN/GRU 基础单元 |
| Spatial-Context Temporal Conditioning | ALFTemporal 为既有标识 | 空间上下文时间条件分支 | 历史嵌入、空间分区交互及条件特征；不另生成预测 |
| Lifecycle-Conditioned Graph Propagation | 不新增缩写 | 生命周期条件化图传播 | 6 维状态经线性映射与 dense softmax 组合五个专家 |
| Partitioned Spatial Context Encoding | 不新增缩写 | 分区空间上下文编码 | 分区组织借鉴 PatchSTG，两阶段注意力沿用并适配 Crossformer 计算结构 |
| FiLM Parameter Head / Feature-wise Linear Modulation | FiLM | FiLM 参数头 / 逐特征线性调制 | 经典 FiLM，作用于编码末状态；不声称重新提出该算子 |
| Role-Scaled Graph Refinement | 不新增缩写 | 角色缩放图修正 | 图预测残差经固定末角色倍率缩放；不引入另一个生命周期 router |

ALFGraph、ALFTemporal 是已有模块标识，不是其功能性副标题的首字母缩写。SAHC 是组合处理区域的名称，不应将 GRIN/BiGRIL 的双向重建算法作为本文新提出的机制。上表不修改任何代码类名、变量或配置字段。

## 符号与维度

图上用上标 `L` 表示历史末步，不表示乘方；以下写成下标 `L` 便于与代码说明对应。所有信号均在训练统计得到的标准化空间中。

| 符号 | 含义 / 实际维度 |
|---|---|
| B, L, N | 批量、历史长度 12、完整节点数；图中省略 B |
| X / Xvis | 可见交通历史 `[B,12,N,1]`；缺失输入由 M 标识，不把隐藏真值用于补全 |
| Rₜ | 角色 one-hot `[B,N,4]`，代码通道固定为 AAS/NAS/VS/FS |
| aₜ, Mₜ | 生命周期状态年龄、观测掩码，各 `[B,N,1]`；图上年龄条为 Join 的示意轨迹 |
| Sₜ=[Rₜ,aₜ,Mₜ] | 逐步条件 `[B,N,6]`；S 指完整 `[B,12,N,6]` |
| A, G | A 为静态邻接 `[N,N]`；G 指同一静态图及其可用属性，不是额外学习图 |
| q | 当前窗口可见计数 `Σₜ Mₜ`；只用于说明有无观测，不是新输入特征 |
| Xfilled, X̄ | 插值开关处理后的补全值，以及中心化后的共享历史；均 `[B,12,N,1]` |
| v, α | v 为最后一次可见的标准化读数 `[B,1,N,1]`，无观测则为 0；α 为固定中心化强度 |
| Conv (12) | 时间卷积 kernel=stride=12；9 个输入通道压成每节点唯一 64 维 patch |
| ToD / DoW | 历史末步日内时刻、星期 embedding，各 32 维；归一化日历还参与历史卷积 |
| P, C | 分区数 32、每分区容量；矩阵横轴 P、纵轴 C，图中仅画示意槽位 |
| Z | 原节点顺序的条件特征 `[B,N,160]`，160=64+32+32+32 |
| Δγ, β | FiLM 参数，各 `[B,N,64]`；共享给两层编码末状态 |
| H | 编码末状态 `[2,B,N,64]`；不是未来各步重新计算的 FiLM 条件 |
| L1 / L2 | 两个循环层；不是两个预测步或两个独立预测器 |
| U | 某个 GRU 门的特征输入 `[B,N,F]`，与 router 的 S 输入不同 |
| Pⱼ, Θⱼ, wⱼ | 第 j 个支持矩阵 `[N,N]`、投影 `[F,64]`、逐节点权重 `[B,N,1]`，j=1…5 |
| I / Mean / WM / Fwd / Bwd | Identity、Binary Mean、Weighted Mean、Forward Diffusion、Backward Diffusion；短标签不是新模块名 |
| r,u,c,h | reset/update gate、candidate、旧隐藏状态；h′ 为新隐藏状态，均 `[B,N,64]`；GRU 式中的乘积逐元素进行 |
| yₖ, Δyₖ, d | 解码步预测、`Linear(hₖ)` 投影增量、数据集固定残差倍率；各步预测为 `[B,N,1]` |
| X̄ᴸ / Sᴸ / Rᴸ | 共享历史末值、末步上下文、末步角色；后两者在预测窗口保持不变 |
| Y₀ | 解码后恢复参考值的预测 `[B,12,N,1]` |
| P⁺, P⁻ | 基于 A+I 与其转置分别行归一化的修正传播矩阵；不要与分区数 P 混同 |
| s(Rᴸ) | 固定角色倍率，按 AAS/NAS/VS/FS 为 `[0,1,1,0.25]` |
| Ŷ | 最终 12 步预测 `[B,12,N,1]`，图中只出现一个预测出口 |
| GR | BiGRIL 内的一次图循环重建步骤示意；三个小块表示展开示意，不是三层网络 |
| AR / MLP / FFN | Autoregressive / Multilayer Perceptron / Feed-Forward Network |

以 α 表示中心化强度，避免与 GRU 候选状态 c 混用；这是论文图符号调整，未改代码。

## 公式和数据处理顺序

1. **全节点共享重建**：所有输入均先运行一次共享 BiGRIL。正反两个方向分别递推，对齐每个历史位置的表示后合并；方向间不是串联。`Xgrin = M⊙X + (1−M)⊙Xrec`。M 还参与 BiGRIL 内部重建与融合。
2. **插值覆盖**：开关开启且 q>0 时，只用窗口内实际可见值覆盖缺失槽位；内部线性插值，边界取最近可见端点。q=0 保留基于图消息的 GRIN 补全。不是 q>0 和 q=0 分别跳过/执行 GRIN。
3. **中心化**：`X̄ = Xfilled − αv`，S 保持原样。两分支读取同一 X̄。观测值在中心化前保留；中心化后也会整体平移。
4. **时间嵌入**：`ToD=12·hour+floor(minute/5)`，`DoW∈{0,…,6}`；`[X̄,S,ToD/288,DoW/7]` 共 9 通道经 12 步卷积得到 64 维特征。拼接最后历史步的 ToD/DoW 各 32 维以及结构嵌入 32 维。
5. **结构嵌入**：节点属性为坐标 2 维、入/出/总加权度、log 总度、孤立标志，共 7 维，经共享 MLP `7→64→32`。没有坐标时坐标槽为 0；不存在逐节点可训练 ID 表。G 端口同时涵盖这一属性来源。
6. **分区交互**：重排/补足成 `[B,1,P,C,160]`，先同槽位跨 P 个分区注意力，再每分区经 4 个 latent tokens 汇聚 C 槽位并回传。只有 1 个交互块、1 个 attention head。布局补足用真实节点索引；交互后恢复原始 N 节点顺序。
7. **生命周期路由**：每个门分别计算 `w=softmax(S Wgate)`；`G(U,S)=Σⱼ wⱼ⊙(Pⱼ U Θⱼ)+b`。全部五槽均参与，未使用 noisy/top-k 路径。WM 与 Fwd 的归一化形式相同，仍保留各自独立参数。
8. **GRU**：`r=σ(Gᵣ([x,h],S))`、`u=σ(Gᵤ([x,h],S))`、`c=tanh(G꜀([x,r⊙h],S))`、`h′=u⊙h+(1−u)⊙c`。编码与解码各 2 层，每层 3 个独立 router，共 12 个；仅时间递推复用参数。
9. **FiLM**：`(Δγ,β)=MLP₁₆₀→₃₂→₁₂₈(Z)`；`Hmod=(1+Δγ)⊙H+β`，同一参数广播到两层，只在编码末状态交给解码器前注入一次。没有额外 S→FiLM head 的直连。
10. **解码**：推理时 `y₀=X̄_L`，`Δyₖ=Linear(hₖ)`，`yₖ=yₖ₋₁+dΔyₖ`，k=1…12。图内回环取未修正的解码输出；初值不一定是最后一次真实观测。输入 x 与隐藏 h 的拼接/更新均遵循真实 DCRNN cell。
11. **参考恢复与修正**：`Y₀=Ydec+αv`；`V=concat(Y₀,P⁺Y₀,P⁻Y₀)`；`D=Linear₃₂→₁(GELU(Linear₃→₃₂(V)))`；`Ŷ=Y₀+0.05·s(R_L)⊙tanh(D)`。恢复参考值不是逆标准化。Refiner 只用末角色控制倍率，不接 age/mask，也没有独立 router；修正结果不回送解码器。

## 四数据集开关

| 数据集 | N | P | C | 插值 | d | α | 布局 / 运行适配 |
|---|---:|---:|---:|---|---:|---:|---|
| METR-LA | 207 | 32 | 7 | 关闭 | 0.3 | 0 | 已验证坐标 KD-tree；原生路径 |
| PEMS-BAY | 325 | 32 | 16 | 关闭 | 0.3 | 0 | 已验证坐标 KD-tree；原生路径 |
| PEMS04 | 307 | 32 | 10 | 开启 | 1 | 0 | 图结构 BFS；`bridge_delta1` |
| PEMS08 | 170 | 32 | 6 | 开启 | 1 | 0.5 | 图结构 BFS；`triple_halfcenter` |

星号仅表示上述固定配置差异，不是可训练的开关或推理条件 router。四组最终配置来自 `selected_configuration_manifest.json`，未用默认 ALFNet YAML 替代。

## 观测场景、Visibility 和训练目标

| 场景 | 内部角色轨迹 | 年龄 |
|---|---|---|
| Unseen | VS | 0 |
| Steady | AAS | 1 |
| Join | VS→NAS | 前段 0；接入后从 1/12 递增 |
| Fault | AAS→FS | 前段 1；故障后从 1/12 递增 |

图上四条轨迹用不同角色色带显示。它们是观测场景示意，不是模型额外输入的场景 one-hot；事件可发生在窗口不同位置。天然缺测只影响 M，不自动改变为 FS。VS 不等价于“从未参与过训练的节点”。没有将已丢弃的 state_encoder 输出、可见比例或历史 Visibility 方案画成有效预测支路。

训练目标：`L=0.9Lmean+0.1Lrole+Lroute`。若 e 为原单位绝对误差、σdata 为训练标准差，元素项为 `ℓ(e)=(e/σdata)·tanh(0.1e)`。Lmean 对有效元素求均值；Lrole 先按历史末角色分别平均，再对本批次实际存在的角色等权平均。Lroute 包含实现中的 0.01 系数，并按递推调用、传播模块取均值；dense soft 下 load 项为零，importance 平衡项保留。Masked teacher forcing 只在训练启用，前 40 轮衰减；图中回环是推理关系。没有独立 GRIN 重建、频谱或第二预测头损失。

## 模块—代码依据

| 区域 | 代码位置（本次 HEAD 行号） |
|---|---|
| 角色/年龄/掩码与全 N 输入 | `data_provider/lifecycle_roles.py:12`；`data_provider/v4_lifecycle_loader.py:204,239,319`；`lib/v4_batch.py:70` |
| 唯一共享重建与分支分发 | `model/ALFNet/ALFNet.py:368,384,389`；`layers/source_grin.py:198,312,416` |
| 插值、参考统计和中心化 | `experiments/flow_support_transfer.py:9,31,40`；`experiments/flow_unified.py:10,29` |
| 编码 / 解码 / FiLM 注入 | `model/ALFGraph/ALFGraph.py:16,64,294,322`；`model/ALFNet/ALFNet.py:244,403` |
| Dense router 与五专家 | `layers/lifecycle_moge.py:61,269,334,445`；`layers/DCRNNCell.py:60`；`layers/GRUCell.py:30` |
| 时间嵌入与分区顺序 | `model/ALFTemporal/ALFTemporal.py:188,201`；`model/ALFTemporal/patchstg.py:89,160,265,289` |
| 结构嵌入与布局 | `layers/structural_node_encoder.py:345,352,403`；`lib/runtime_context.py:306`；`lib/graph_native_layout.py:110,144,161` |
| 图残差修正 | `model/ALFNet/ALFNet.py:26,55,72,456` |
| 预测和辅助损失 | `experiments/focal_loss.py:9`；`experiments/v4_losses.py:25`；`experiments/v4_runner.py:379` |

完整前期依据见 [第一阶段审计](../STAGE1_IMPLEMENTATION_AUDIT_CN.md)。本图是功能分区：FiLM head 和 refiner 在代码中属于 ALFNet，蓝色大区不是 Python 类边界。两分支图形并列表示依赖关系；源码并非并发执行两个专家，也不共享其主干权重。

## 图形与 PPT 使用说明

参考图 5 的浅色分区和正交线、图 4 的矩阵/特征条、图 3 的图节点及圈加/乘作为样式依据。白底，ALFGraph 蓝、ALFTemporal 绿、生命周期紫、共享补全浅红、FiLM 黄、修正浅橙。颜色不表示专家概率大小。

- 每个输入只出现一次；同名端口仅引用原信号。S、R、a、M 用紫色，静态图引用用灰色，数据/参数流用深灰实线。
- `[]` 和并排特征条表示拼接，圈加表示残差加法，圈内 ⊙ 表示逐元素乘。图内候选状态更新式为了紧凑省略部分 ⊙，语义见公式节。
- 整图只展开一次代表 cell；浅蓝无箭头引线表示“单元内部定义”，不是额外计算流。
- 素材库从排版后的整图提取；19 个独立 SVG 保留 `<text>`，无嵌入位图、`foreignObject`、外链或滤镜。Arial/Cambria Math 字体，使用者应保持字体可用。
- 本轮核验了 SVG 几何和浏览器渲染；**未在本机 PowerPoint 中实测“转换为形状”及拆分文本的结果**。不能把 SVG 可编辑性等同于已经验证所有 Office 版本原生拆分无差异。原 SVG 始终保留可编辑文本和基础图元。

## 可复现命令

仅依赖现有 Node.js 及本机 Chrome，未安装依赖，未运行 Python 或模型。

```powershell
node paper/architecture_20261019/v2_20261019/_source/build.mjs
node paper/architecture_20261019/v2_20261019/_source/render.mjs paper/architecture_20261019/v2_20261019/_source/alfnet_architecture.svg paper/architecture_20261019/v2_20261019/alfnet_architecture.png 3
node paper/architecture_20261019/v2_20261019/_source/render.mjs paper/architecture_20261019/v2_20261019/module_library.svg paper/architecture_20261019/v2_20261019/qa/module_library_preview.png 0.7 --audit-modules
```

渲染器使用独立临时 Chrome profile、无界面窗口和本地文件；沙箱中启动 Chrome 需要运行环境授权。本轮已获自动审批并完成，不涉及用户浏览器配置。`qa/` 中保留了先整体布局、后内部细节、最后字号/端点修复的过程。
