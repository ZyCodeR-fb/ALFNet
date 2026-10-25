# 给 Claude 的 ALFNet 架构图续绘任务 Prompt

请接手 ALFNet 正式论文模型架构图的设计、生成与核验任务。你需要先恢复项目上下文，再在已有成果上继续完善，最后使用你实际可调用的、经核实的最新 Nano Banana 图像模型制作架构图。

项目仓库：`A:/projects/ALFNet`。

## 一、我的目标与最新决定

我需要一张结构准确、视觉专业、机制清晰、便于用原生 PowerPoint 图形复现的完整 ALFNet 模型架构图。正式论文涉及 METR-LA、PEMS-BAY、PEMS04、PEMS08。

此前已经完成实现核验、布局设计、代码绘图和一次生图模型视觉重设计。现在有两个用途不同的核心文件：

1. **内容与计算连接参考**：
   `paper/architecture_20261019/v2_20261019/alfnet_architecture.png`
   由 `.mjs` 生成 SVG，再渲染为 PNG，6000×3960。已有静态实现与图形核验记录。其计算内容、顺序和连线是本轮的重要依据，但视觉表达偏框图，不要求照搬其外观；仍需对拟修改内容核对代码。
2. **选定的视觉底稿**：
   `paper/architecture_20261019/imagegen_v3/alfnet_imagegen_r1.png`
   由生图模型参考代码版生成，1639×960。整体比代码版美观，但存在连线和局部机制表达错误，**不是通过全部语义核验的正式成图**。

我的最新决定是：**暂时保留 R1，以 R1 为视觉基础继续优化，不采用 R2 及之后的图作为新底稿。** 后续多轮局部修图出现了越来越模糊、文字与符号退化、旧错误未完全修复或引入新错误的情况。保留这些历史文件，但不要继续在模糊版本上层层叠加。

风格参考目录：`paper/architecture_20261017/Reference_Image/`，包含 `1.png` 至 `7.png`。我认为 R1 对其中可复现的图形元素借鉴还不充分，一些模块只是贴标签的大方块，例如 Temporal Patch，让读者看不懂发生了什么。新图必须同时做到好看、机制清晰、连接准确。

## 二、先恢复上下文，辨别当前权威

遵循仓库根目录入口约定，依次阅读：

1. `AGENTS.md`；如存在适用于 Claude 的 `CLAUDE.md`，也读取。
2. `docs/ai/WORKFLOW.md`
3. `docs/ai/PROJECT_CONTEXT.md`
4. `docs/ai/HANDOFF.md`
5. 该索引指向的当前任务 HANDOFF。

然后读取：

- `docs/FINAL_PROJECT_INDEX.md`
- `docs/RESEARCH_LESSONS.md`
- `paper/manuscript_draft_cn-2-v5.md`
- `paper/experiments_draft_cn.md`
- `paper/pems08_wd_test_20261012/selected_configuration_manifest.json`
- `paper/architecture_20261019/STAGE1_IMPLEMENTATION_AUDIT_CN.md`
- `paper/architecture_20261019/STAGE2_LAYOUT_DESIGN_CN.md`
- `paper/architecture_20261019/v2_20261019/README_CN.md`
- `paper/architecture_20261019/v2_20261019/VERIFICATION_CN.md`
- `paper/architecture_20261019/imagegen_v3/GENERATION_PROMPT.txt`

查看两张核心图和全部七张参考图；不能只读文件名或前一位助手对图片的概述。旧图 `paper/architecture_20261017/` 只是历史参考，不是结构权威。

检查 Git 当前分支、HEAD、工作区和相关 diff，不切换分支、不覆盖用户改动。2026-10-19 本交接编写时分支为 `paper/final`，HEAD 为 `f06367e08e05e33f2364e7639dc614ab53c983f3`，当时 `git status --porcelain=v1` 为空。代码绘图核验记录中的 HEAD 是更早的 `e02b036d665e0c82f33afd4195b7a6ea8cad1c18`；不要把旧核验无条件宣称为当前 HEAD 的重新核验。

权威顺序：最新用户要求和最终配置清单 → 对应实际模型与运行适配代码 → 已核验的映射说明 → 论文草稿和历史设计。共享文档含大量已废弃候选和历史参数，不能只看默认 `configs/ALFNet/*.yaml` 就判断最终四个数据集的计算路径。服务器权重本地缺失，不妨碍本轮静态绘图任务，不需要为此下载权重或重新推理。

## 三、严格任务边界

- 不训练、不运行消融、不访问或重新评估测试集、不启动 GPU 模型实验。
- 不修改模型算法、网络结构、训练逻辑、最终超参数、权重或实验结果。
- 不恢复已删除、废弃或只存在于历史候选中的机制。
- 不为统一论文命名而重命名代码类、变量、配置字段。
- 不提交 Git commit，不 push，不切换分支。
- 仅在 `paper/` 下新建本轮独立绘图目录；保留所有历史图、脚本和说明。
- 发现论文与代码冲突时，列证据及建议；本轮不要擅自全局修改论文或历史文档。
- 不将经典 GRIN/BiGRIL、DCRNN/GRU、Crossformer 或 FiLM 改名后暗示为全新原创算子。
- 不编造已调用的模型、分辨率、核验结果或可编辑性。

## 四、实现核验：沿用已有成果，但对关键关系复核

优先使用第一阶段审计和 v2 README 的模块—代码映射，然后复读相关实际前向路径：

`model/ALFNet/`、`model/ALFGraph/`、`model/ALFTemporal/`、`layers/source_grin.py`、`layers/lifecycle_moge.py`、`layers/structural_node_encoder.py`、`layers/DCRNNCell.py`、`layers/GRUCell.py`，以及最终配置引用的必要模块。

数据与适配部分：`data_provider/v4_lifecycle_loader.py`、`data_provider/lifecycle_roles.py`、`lib/v4_batch.py`、`experiments/flow_support_transfer.py`、`experiments/flow_unified.py`；损失部分：`experiments/focal_loss.py`、`experiments/v4_losses.py`、`experiments/v4_runner.py`。空间布局相关见 `lib/runtime_context.py` 和 `lib/graph_native_layout.py`。

以下是已有静态核验结论，帮助定位，不代替源码证据：

### 1. 输入、场景与生命周期条件

- 历史长度与预测长度均为 12，输入交通通道为 1。省略 batch 后历史 `X∈R^(12×N×1)`。
- `S_t=[R_t,a_t,M_t]` 为 6 维：4 维 Lifecycle Role one-hot、1 维 State Age、1 维 Observation Mask。
- 论文问题场景用 **Unseen / Steady / Join / Fault**；内部角色通道仍是 **AAS / NAS / VS / FS**。
- Unseen 对应窗口内 VS；Steady 对应 AAS；Join 可以 VS→NAS；Fault 可以 AAS→FS。不要画成四种整窗固定、互斥的场景 one-hot 输入。
- 自然缺测影响 M，不自动把角色改成 FS。State Age 是角色状态年龄，不是独立 Visibility。
- 没有应当另画的有效 Visibility 输入分支。q 是当前窗口可见计数，由 M 统计；未参与预测的中间表示不能画成有效机制。

### 2. 唯一共享历史补全与支持处理

- 所有节点先统一执行一份共享 GRIN/BiGRIL；正反两个方向分别递推，按历史位置对齐后合并。不要把两个方向串联。
- 保留实际观测，填补不可用位置；可选窗口插值在配置开启且 q>0 时覆盖缺失位置，q=0 保留空间 GRIN 补全。
- **不能画成 q>0 走插值、q=0 才运行 GRIN 的互斥分支。**
- 可选中心化 `Xbar=Xfilled−αv`，v 是最后一次真实可见的标准化读数，无观测时为 0。
- 两个主分支共享同一处理后的 Xbar；不各运行一套 GRIN。
- 解码后先恢复 `+αv`，再 Graph Refinement。中心化不是逆标准化。

四数据集处理差异应放配套说明，主图只标可选处理：

| 数据集 | N | 空间分区数 P | 每分区容量 C | 窗口插值 | 解码残差倍率 d | 中心化 α |
|---|---:|---:|---:|---|---:|---:|
| METR-LA | 207 | 32 | 7 | 关 | 0.3 | 0 |
| PEMS-BAY | 325 | 32 | 16 | 关 | 0.3 | 0 |
| PEMS04 | 307 | 32 | 10 | 开 | 1 | 0 |
| PEMS08 | 170 | 32 | 6 | 开 | 1 | 0.5 |

### 3. ALFTemporal：重点提升视觉解释力

- `[Xbar,S,ToD/288,DoW/7]` 共 9 个通道，经 kernel=stride=12 的时间卷积形成每节点 **1 个 temporal patch、64 维**。
- 拼接末历史时刻 ToD 32、DoW 32、结构嵌入 32，得到 160 维；日历归一化输入和日历 embedding 是不同的实际用途。
- 结构嵌入来自共享结构 MLP `7→64→32`，输入为静态图属性及可用坐标；不是可训练 node-ID 表，无坐标时相应槽位为 0。
- 空间分区 **P=32**，容量 **C=7/16/10/6**；C 不是分区数。
- 重排/补足后的特征可表示为 `[B,1,P,C,160]`。补足使用真实节点索引，不要随意画成空白零节点。
- 先 **Cross-partition interaction**：同一槽位跨 P 个分区交互；再 **Within-partition interaction**：一个分区的 C 个槽位经 **4 个 latent tokens** gather / broadcast。
- 只有一个两阶段注意力块、一个 attention head，不要凭视觉效果增加多个时间 patch 或多个 Transformer 堆叠。
- 恢复原始节点顺序，得到 `Z∈R^(B×N×160)`。时间分支只有条件特征，没有独立 Forecast。

### 4. ALFGraph、Router 与 FiLM

- Graph Encoder 和 Residual AR Decoder 各 2 层，hidden=64。
- Encoder 使用逐步 S_t，Decoder 使用历史末 S_L，预测期间保持该上下文。
- 每个门的 router 从 6 维 S 计算线性映射与 softmax，五个专家 Dense Soft Routing，非 top-k；不要添加隐藏状态到 router 输入。
- 五专家：Identity、Binary Mean、Weighted Mean、Forward Diffusion、Backward Diffusion。
- Weighted Mean 与 Forward Diffusion 的归一化传播形式重合，但有独立投影/路由参数；不要为让图好看而伪造不同传播矩阵。
- `G(U,S)=Σ_j w_j⊙(P_j U Θ_j)+b`。
- GRU：`r=σ(G_r([x,h],S))`，`u=σ(G_u([x,h],S))`，`c=tanh(G_c([x,r⊙h],S))`，`h'=u⊙h+(1−u)⊙c`。候选计算不能漏掉 x。
- 各层、各门参数独立，时间递推复用。只画一个代表 cell 解释其内部，不能暗示全模型共用同一 router 权重。
- FiLM head `160→32→128`，输出 Δγ、β，各 `N×64`。
- `Hmod=(1+Δγ)⊙H+β`，调制**两层编码器末状态**，发生在交给 Decoder 前的一次状态变换。
- 参数线必须进入 FiLM；不能直接接 Decoder，也不能画成每个未来步重复 FiLM。没有额外 S→FiLM head 直接输入。
- 推理解码初值 `y_0=Xbar_L`，不是最后一次真实观测 v；`y_k=y_(k−1)+d·Linear(h_k)`，回环用尚未参考恢复/图修正的逐步预测。

### 5. Graph Refinement 与唯一输出

- 解码输出先恢复参考值，得到 Y_0，再进行修正。
- 正确路径：`[Y_0,P+Y_0,P−Y_0] → MLP(3→32→1, GELU) → tanh → ×0.05·s(R_L) → +Y_0 → Yhat`。
- P+、P− 分别由 A+I 及其转置行归一化得到。
- 固定角色倍率按 AAS/NAS/VS/FS 为 `[0,1,1,0.25]`。这里仅用末角色 R_L，不能添加完整 S 或另一个自适应 router。
- 原预测残差与修正量在加法节点相汇；只有这个结果通往最终 Forecast。
- 不得因部分展示的消融表没有 Refiner 行，就删除该模块。

### 6. 训练专用内容

- 训练目标 `L=0.9Lmean+0.1Lrole+Lroute`。Lroute 已包含实现中的 0.01 系数，不要重复相乘。
- 元素预测损失为回归型 `ℓ(e)=(e/σdata)·tanh(0.1e)`，e 为原单位绝对误差；不是简单 MSE，也不是分类 Focal。
- Lrole 对本批次存在的末角色先各自平均，再等权汇总。
- Router 辅助平衡损失和 masked teacher forcing 在独立 training-only 区说明；不能让真值进入推理路径。
- 没有额外 GRIN 重建损失、频谱损失或第二预测头。

## 五、R1 的具体问题：保留美感，修正结构

请以实际图片复核下列问题，给出修复后的连接清单，而不是把文字改动当作修图成功：

1. **FiLM 端点错误**：上方 Δγ、β 的连接落在 Residual AR Decoder 顶部，应进入左侧金色 FiLM，调制编码末状态。
2. **输出存在绕行歧义**：Restore 后横线穿过 Refiner 上部直接指向 Forecast，容易形成跳过修正的预测出口。删除该旁路，Forecast 只接最终残差和。
3. **Y_0 命名提前**：Decoder 内输出被标为 Y_0，但该符号应留给恢复参考值后的完整预测；区分逐步 y_k、解码序列和 Y_0。
4. **静态图线接入错误区域**：左侧 A/G 的长线落在 Reference centering 附近，可能误解为图直接参与中心化。用清楚的引用端口接 GRIN、结构 MLP、传播和修正；中心化仅按真实公式连接。
5. **q>0 放错位置**：Keep observations 的 Missing 行出现 q>0，容易限制通用补全。q>0 条件应附在可选插值覆盖；GRIN 对全部输入执行。
6. **时间嵌入连接欠清楚**：Conv 输出、单个 temporal patch、Concat、日历与结构 embedding 的合流关系需重新组织，不能只在附近摆放张量块。
7. **Patch 表达弱**：绿色长方体无法说明时间压缩过程，且多个小格可能被误读为多个时间 patch。
8. **分区交互表现不足**：大网格中的色带不充分说明哪些节点、槽位在交互；要用同槽位高亮、明确方向箭头及下一阶段的四 latent token 交互表达。
9. **GRU 候选式不完整**：R1 的候选式只呈现 r⊙h，遗漏当前输入 x。修正或用准确简图加配套公式，不能保留错误简式。
10. **循环层次和反馈不够明确**：立体块、层间箭头与回环要能区分“两个循环层”和“十二个预测步”；不能把第二层 hidden 当作预测反馈。
11. **参考恢复与修正残差绕线易混淆**：αv 恢复项和 Y_0 残差是不同信号；画清各自进入哪个加法节点，不用模糊长线混在一起。
12. **小字与公式风险**：R1 只是 1639×960 位图，复杂公式偏小。后续连续编辑还出现箭头反向、符号变形、字体发虚，不能靠不断修同一张低清图解决。

R2 及之后有些局部关系修对了，但整图质量退化，因此它们只能作为失败经验；不要把“后来的版本”自动当作更可靠的图。原有 `CORRECTION_PROMPT_R*.txt` 可按需查看，不必把全部旧提示词累积进新提示词。

## 六、视觉设计：更多借鉴参考图的有效图元

必须逐图分析 `Reference_Image/1.png` 至 `7.png`，提出简短的“参考图具体元素 → 用于 ALFNet 哪个机制 → 如何用 PPT 基础图元重建”表。不要只说借鉴配色或专业风格。

优先方向：

- 参考图 4：带时间/节点轴的薄特征张量、切片、时钟/日历、拼接特征条、明确的局部运算和 token 组。借鉴其表达手法，不复制 prompt / prototype / memory / node-ID 机制。
- 参考图 3：节点与边、薄矩阵、圈加/乘、局部运算连接。不要引入动态图、多尺度、蒸馏或下采样。
- 参考图 5：清晰容器、浅色区域、对齐、正交箭头和留白。不要复制“总体图加一整套重复内部细节”的分面布局。
- 参考图 1、2、6、7：有选择地借鉴观测曲线、节点身份保持、槽位高亮、连续嵌套流程，不能复制其其他模型计算。

至少让以下机制得到真正的图形表达，而不是统一替换为写着名称的方框：

1. **Temporal Patch**：画一小段带时间轴的 12 步输入切片，用括号圈住整个窗口，经 Conv(12) 压成每节点一个 token。标清“1 temporal patch / node”“64 features”。输出小格表示特征维，不能表示多个 patch。不要再画一整套重复交通曲线。
2. **Spatial Partition**：用节点特征小片或矩阵重排表现节点进入 P 个分区；标清 P 与 C 两轴。可仅显示少量分区并用省略号注明 P=32，不必真的画满 32 列。
3. **Cross-partition**：高亮各分区同一槽位，用短交互线/注意力连接展示跨分区交换，不能只靠框标题说明。
4. **Within-partition**：C 个节点槽位 → 四个 latent tokens → C 个槽位，明确 Gather 和 Broadcast 的两个方向；四个 token 要能数清。
5. **Restore node order**：同色节点/特征短条经索引重排回原序，突出“还原顺序”，而非新增网络层。
6. **Shared Completion**：双向递推、对齐融合、mask 选择与插值覆盖具有不同视觉语义；图形应解释保留观测与填补缺失。
7. **Dense Routing**：五个小 support / 图传播图形加五个权重条，汇入一次加权和；不要画成硬选择一个专家。
8. **FiLM**：两组参数分别进入逐元素缩放与平移，H 到 Hmod 的主路清晰；不是只放两块彩色张量。
9. **Residual Decoder / Refinement**：用明确的增量加法、回环和残差加法展现机制；不把不同的残差连接混为一条线。

这些是表达目标，不要求把每项都画成复杂插图。整图可读性优先，避免为了增加图元而拥挤。允许对弱区域重新排版，但必须保留 R1 的整体清晰度与视觉优点。

## 七、整体布局与风格约束

默认采用已经形成的单图方案，直接优化，无需重新等待我确认同一个总体方向：

`左侧一次性输入 → 一次性共享历史补全 → 右上 ALFTemporal / 右下 ALFGraph → 最右唯一 Forecast`。

时间分支通过 FiLM 参数向图分支注入条件。复杂机制在所属区域内展开；最多一个嵌入式代表 cell 定义框，以无箭头细引线说明对应关系，不能误当主路额外计算模块。

- 不分 (a)(b)(c)(d)，不再搭建一套 Overall Architecture 加一套相同展开图。
- 同一输入只详细画一次，同一模块只完整展开一次，最终输出只画一次。
- 不严格强求所有细节横排，优先上下分支、合理嵌套、短连线和足够留白。
- 白底；ALFGraph 浅蓝、ALFTemporal 浅绿、生命周期紫、补全浅红、FiLM 黄色、修正浅橙。
- 数据与参数用深灰/黑实线；生命周期条件统一紫色虚线；静态图引用用低对比灰线或命名端口；图例说明。
- 统一字体、字号、线宽、箭头、圆角、节点和张量样式。正式英文字体，避免手写、拼写红线、强渐变、发光、厚阴影。
- 使用矩形、圆、折线、薄张量侧面、矩阵、圆形运算符等 PPT 可复现元素。位图参考不能被描述为可直接提取的原生 PPT 对象。
- 未来输出明确为 12 步示意曲线，不使用实际测试数据。
- 缩至论文约 180 mm 通栏宽度后仍可阅读。不能通过无限增大画布、增加像素或缩小字体解决信息过载。

## 八、命名沿用已有体系，必要调整要说明理由

默认沿用 v2 README 的名称：

| 图中名称 | 含义/约束 |
|---|---|
| ALFNet — Adaptive Lifecycle Forecasting Network | 完整模型 |
| Support-Aware History Completion | GRIN 基础重建与支持处理的组合区域，不把 GRIN 称为新提出算法 |
| ALFTemporal — Spatial-Context Temporal Conditioning | 时间条件分支；ALFTemporal 是既有标识，不是副标题首字母缩写 |
| ALFGraph — Observation-Conditioned Graph Forecasting | 图预测分支；图中可用短标题 ALFGraph |
| Lifecycle-Conditioned Graph Propagation | 五专家 dense 路由传播，不新增无必要缩写 |
| Partitioned Spatial Context Encoding | 不将空间分区误叫多个 temporal patch |
| FiLM Parameter Head / FiLM | 经典参数生成与调制 |
| Role-Scaled Graph Refinement | 比笼统 Lifecycle-Aware 更准确地体现末角色固定倍率 |

图形功能区不一定等同代码类边界；FiLM head 和 refiner 实际位于 ALFNet，不因画在蓝色预测区就宣称它们属于 ALFGraph 类。

## 九、Nano Banana 生成与防退化策略

1. 先核实当前实际可调用的 Nano Banana 模型及版本，记录模型标识、调用方式和输出能力；“最新”以官方信息或实际工具列表为准，不凭记忆编造名称。若无法调用，应明确说明并完成独立的设计/提示词工作，不能假装已经使用该模型，也不要擅自换成其他模型冒充。
2. 先完成区域布局、信号接口和箭头清单，再细化模块，最后统一视觉。不要先生成大量漂亮小组件再硬拼一张大图。
3. 生图参考至少明确区分：代码 PNG 是内容/连线依据；R1 是视觉底稿；参考图是图元/风格依据。不要让模型同时复制三套互相竞争的架构。
4. 用精炼、经过消歧的生成提示词重新生成清晰版本；细节有误时优先从清晰 R1 或最近通过清晰度检查的源图重新生成/定点编辑，不沿着退化图反复重绘。
5. 有可靠局部编辑能力时才锁定局部修正；检查未编辑区域是否变化。连续修正无效或明显变糊时停止该链，调整布局/简化文字后从清晰源重新生成。
6. 优先模型原生高分辨率（如实际支持 4K）；保存原始输出，不把插值放大称为新增真实细节。分辨率能力不支持时如实说明。
7. 保留必要数学符号与机制，冗长公式可移到图注/配套说明；不能为减少文字而删除 FiLM 作用点、参考恢复、修正残差或四 token 等关键关系。
8. 生图模型绘制小字和精确箭头不稳定。对无法可靠生成的文字/端点明确标注待校正，不把错误版本包装成可直接投稿的最终图。需要改变制作方式时先说明；本任务指定的主要视觉生成应由 Nano Banana 完成。

## 十、执行、核验与交付

请按以下顺序推进，不只回复“可以”或停留在泛泛设计建议：

**第一步：恢复与核验。** 读取上下文与参考图，简要报告当前事实、R1 的错误、需要改进的视觉机制、拟采用的模型能力；避免无必要重复整套历史实验审计。

**第二步：明确设计。** 给出简明“参考图元→模块”映射、区域布局和关键连接清单。沿用已指定的单图方向，不必重新询问是否采用上下分支。

**第三步：整图生成。** 先完整布局，再机制，再排版；每一轮保存独立文件和使用的提示词。以清晰 R1 为基础，不继承模糊后续图。

**第四步：逐项核验与修复。** 每轮检查：

- 输入与共享历史是否各只出现一次；场景与角色是否混淆。
- GRIN、观测保留、插值、中心化、恢复参考值的顺序和开关。
- 单时间 patch、P/C 含义、跨分区→分区内、4 latent tokens、节点还原。
- Z 与 FiLM 参数维度、FiLM 到两层编码末状态的准确位置。
- 两层 encoder/decoder、五专家 dense router、正确 GRU 候选式。
- Decoder 初值和逐步预测反馈；不能从 hidden 或 refiner 输出取反馈。
- Restore→Refinement→唯一 Forecast；残差加法两个输入准确，无绕过修正出口。
- 每条箭头端点、方向和交叉；区分数据线、条件线、说明引线。
- 字体、标签、数学符号、拼写、遮挡、低分辨率退化和通栏缩小可读性。
- 训练目标与推理路径分离；没有凭空增加的机制。

采用通过/待修复/未验证的逐项记录，不以“看起来更漂亮”代替语义核验。代码版已有核验不等于新生图版自动通过。

**最终交付在新的独立目录内：**

1. 选定的正式 PNG（若仍有错误，应明确称设计稿并列出剩余问题，不能虚称最终核验通过）。
2. 实际模型标识、生成提示词、参考图清单、版本选择理由和核验记录。
3. 英文图注、中英文模块名称、符号及维度、必要代码依据。
4. 面向我后续 PPT 重建的简短图元指南，说明每个关键模块对应哪张参考图的哪些基础形状。

旧版可编辑素材位于 `paper/architecture_20261019/v2_20261019/module_library.svg` 和 `module_library/`（19 个独立 SVG），可以作为重建素材参考。它们不是新生图 PNG 的精确可编辑版本；PowerPoint 原生拆分在前轮未实测。不要把 PNG 外包一层 SVG 就交付为可编辑模块库。若另做 SVG，应真正包含可编辑图元和文本，并明确是额外制作。

我的核心要求：**以准确代码图为内容依据，以清晰的 R1 为视觉基础，更充分地利用参考图片中适合 PPT 的图形表达，把 Patch、分区交互、路由、调制和残差等机制画清楚；在一张不重复输入、不重复模块的大图中，同时实现美观、可读和科学准确。**
