# ALFNet Figure 1 生图交付与逐项核验

日期：2026-10-29。只应用用户指定的 `nature-figure` 技能（安装版 manifest 2.8.0），采用 AI-generated mechanism schematic 路线和内置生图工具。遵循 FIG1_DESIGN_CN.md 的 NeurIPS/PPT 视觉要求，没有应用 Nature 极简矢量风格，也没有调用其他技能或创建 PPTX。

## 交付状态：版式草稿与裁切素材，非全部核验通过的论文终稿

| 文件 | 实际像素 | 用途 |
|---|---|---|
| [fig1_draft_v1.png](fig1_draft_v1.png) | 1774×887，2:1 | 选定版式草稿：B 方案的一次定点修正 |
| [fig1_draft_v1a.png](fig1_draft_v1a.png) | 1774×887 | 首个独立方案；保留比较，含中心化与输出不确定性误绘 |
| [fig1_draft_v1b.png](fig1_draft_v1b.png) | 1774×887 | 第二个独立方案；保留比较，含三颗星及输出连接缺陷 |
| [assets_curves.png](assets_curves.png) | 2172×724 | 三个白底曲线素材区；右侧为同一坐标系的三条预测曲线 |
| [assets_modules.png](assets_modules.png) | 1341×1173 | 白底八类模块素材，四行两列 |
| assets_curves_v1a.png / assets_modules_v1a.png | 见文件 | 素材首版历史，未覆盖 |
| prompt_*.txt | UTF-8 文本 | 每次实际调用的完整提示词，逐字对应下方记录 |

**分辨率要求未满足**：用户要求整图宽度至少 3000 px。三次整图提示均请求原生 4096×2048，但内置工具实际均返回 1774×887。当前可调用接口没有尺寸参数，不能声称已经产出原生 4K。交付保留工具原始像素，没有人工放大或修改 PNG 内容。后续原生 PPT 重建可按目标尺寸重新导出。

**仍需人工重打/重画**：图中文字、数学上下标、少数计算连接、五专家矩阵和精确色号。尤其不要直接复用整图中的错误残差绕线或将热图示意当成最终算子定义。以下逐项列出，不以“文件已生成”代替语义通过。

## 工具、模型及参考记录

- 实际工具：内置 `image_gen.imagegen`（调用接口 `image_gen__imagegen`）。
- **模型名称/版本：工具没有返回，无法独立核实。** 本轮没有调用 OpenRouter、Nano Banana 或外部 API，也没有把技能文档中的默认模型名当成本次实际模型。
- 输出角色：供作者手工重建的内部设计草稿及示意素材。未核查直接投稿使用生成位图的资格；本轮目标不是直接提交位图。
- 三次主图生成、两次曲线素材生成、两次模块素材生成，共 7 次调用。
- 所有图形曲线和小矩阵均是概念示意，未读取实验数据、测试集或 checkpoint，未训练或推理模型。
- 用户指定的两类图只作为样式参考/内容参考；没有核实参考材料的第三方再分发许可，没有复制其文字或其他模型机制。
- 读取了设计稿、完整方法章、当前架构渲染及全部 1–7 参考图。实际传入生图工具的图片如下，不把“看过”写成“传入过”。

| 调用 | 实际输入图片及顺序 | 精确提示词 |
|---|---|---|
| 整图 A | `imgs/Reference_Image/3.png`；`4.png`；`imgs/figure_design/current_ALFNet_render.png` | [prompt_fig1_v1.txt](prompt_fig1_v1.txt) |
| 整图 B（从原始参考重新生成） | 同上，不输入 A | [prompt_fig1_v1b.txt](prompt_fig1_v1b.txt) |
| 选定整图定点修正 | `fig1_draft_v1b.png`；`assets_curves.png` | [prompt_fig1_final_refine.txt](prompt_fig1_final_refine.txt) |
| 曲线首版 | `imgs/Reference_Image/1.png`；`6.png`；`7.png` | [prompt_assets_curves.txt](prompt_assets_curves.txt) |
| 曲线定点修正 | `assets_curves_v1a.png` | [prompt_assets_curves_refine.txt](prompt_assets_curves_refine.txt) |
| 模块首版 | `imgs/Reference_Image/4.png`；`3.png`；`1.png` | [prompt_assets_modules.txt](prompt_assets_modules.txt) |
| 模块定点修正 | `assets_modules_v1a.png` | [prompt_assets_modules_refine.txt](prompt_assets_modules_refine.txt) |

上述 `imgs/` 均指 `paper/latex_neurips_20261020/imgs/`。下面附录完整收录 7 份实际提示词，独立 txt 便于复制。

## 设计稿第 5 节标签修正表：22 项逐条核验

核验对象为 **fig1_draft_v1.png**。状态“部分”或“重打/重画”不能视为通过；数学式以本表和 method.tex 为准。

| # | 位置 / 要求 | 状态 | 实际成图与 PPT 修正 |
|---|---|---|---|
| 1 | 可见历史 `X∈ℝ^{L×N}`；时间轴 `1…L` | 部分 | 画成 `X (L×N)`，轴有 1/L；缺少 `∈ℝ`。PPT 重打完整维度。 |
| 2 | `R_t: observation role`、`a_t: state progress`、`M_t: mask` | 符合主要标签 | 三项出现；四角色错误英文扩写已去除。S 的完整历史维度已修为 L×N×6。 |
| 3 | SAHC 方向 `1→L`、`L→1` | 符合 | 两方向标签和相反方向箭头存在，不沿用 t−11。 |
| 4 | `MLP fusion of ρ→_t,ρ←_t,m_t`；输出 X^c | 部分 | X^c 存在；ρ 上箭头/上下标不够可靠，需公式编辑器重打。融合和真实观测选择应按方法式(8)说明，不能仅据图中含混细线解释。 |
| 5 | `Window interpolation (ι)` | 部分 | 标题出现，但 ι 外观易与 l 混淆，PPT 重打希腊字母。 |
| 6 | `q_n>0: linear fill of missing slots`；`q_n=0: keep X^c` | 部分 | 两条说明出现；本图未清晰表现两端水平延拓，可用 curves 素材替换。开关 ι=1 下才覆盖缺失位置。 |
| 7 | `Shared history X̄` | 符合主要标签 | 已改为 Shared history X̄；中心化输入为 X^fill，αv 是减法侧输入。 |
| 8 | `F_n=[X̄,S,τ,δ]`；Temporal conv→e_n(64) | 部分 | 按方法采用更精确的 `[X̄,S,τ/288,δ/7]`，e_n(64) 存在；图标旁日历记号及 L×N 标注不正确，应改为长度 L 的 τ、δ 序列。F_n 本身是 L×9。 |
| 9 | `Time-of-day E^tod` / `Day-of-week E^dow` | 部分 | 正确英文出现，tod/dow 及索引需要重打；日历和结构嵌入已走卷积旁路。 |
| 10 | 空间分区按第 4 节 | 部分 | 路网色块、行分区/列槽位、P=32、斜线副本均出现；主图网格的分区色和路网色块对应不完全，建议使用 assets_modules 的 (a)。不画 KD-tree。 |
| 11 | Stage 1/2、T_{:,c}、T′_{p,:}、Q_p 4 tokens | 部分 | 纵向列高亮、横向行、四颗星及 Gather/Broadcast 存在；标题缩为“1./2.”，切片上下标不精确。PPT 重打完整两行。 |
| 12 | 两层编码末状态 h_L^(1),h_L^(2)∈ℝ^{N×64}；G 图标改 A 或删 | 部分 | 两层状态及 N×64 已出现；无旧 G 输入方框；编码区条件线上的 S_t 字符有变形，需重打并接到每个单元。 |
| 13 | FiLM `h_mod^(l)=(1+Δγ)⊙h_L^(l)+β` | 部分 | 公式主要内容与 l=1,2 出现，但下标需重打。参数连线仍为玫红实线，按线型规范应改深灰实线；条形参数保持玫红。 |
| 14 | Decoder `S_L`、`x̄_L`（下标） | 部分 / 缺项 | S_L 及 N×6 出现；解码初值 `y_0=x̄_L` 未清晰出现，必须补上。原始步预测回环需要明确，不从隐藏状态取反馈。 |
| 15 | Ỹ；[ỹ_k,P⁺ỹ_k,P⁻ỹ_k]；Linear–GELU–Linear；s(R_L)；+ỹ_k；Ŷ | 部分 / 连接须重画 | 主要文字出现且无 Y_0；逐步三输入不应标 H×N，应为每节点向量/逐步运算。右侧从 A/输出区域回到最终加法的线仍含混，应删去。让恢复后的 Ỹ 分叉：一条进入逐步三路传播，一条为原 ỹ_k 残差；最后加法只向 Ŷ 输出。R_L→倍率需补明确紫色虚线，倍率本身改紫色。 |
| 16 | 放大框 `Graph GRU cell with OS-routed mixture of graph experts (encoder & decoder)` | 符合主要文字 | OS-routed 已修正；缩小后仍建议重新输入标题，图中不可读细字不作为正式文字。 |
| 17 | P₁=I，P₂ binary，P₃ weighted，P₄ forward，P₅ backward | 部分 / 图标须重画 | 五名称出现；主图 P₁ 外观接近“1”，矩阵格数/对角并不准确。素材表也不能证明正确传播算子；按下方矩阵说明重新制作。 |
| 18 | 混合式含 P_j U Θ_j；w_t=softmax(S_t W_g)；删除旧 Dense mixture | 重打 / 缺项 | 删除了旧 Dense mixture；完整两公式未可靠呈现，部分 Θ 下标重复或模糊。用 PPT 公式编辑器录入本表下方标准式；U 输入与 w 权重分别接入加权和。 |
| 19 | 门控仅用 h̃ 候选；r/u/h̃/h′ 四式 | 部分 | 使用 h̃ 及 [x,r⊙h]，没有另用 c 作为候选；小字下标、括号难以逐字符确认，四式统一重打。 |
| 20 | Encoder:S_t；Decoder:S_L；12 modules (2 layers×3 gates×enc/dec) | 符合主要内容 | 标注存在，PPT 中使用真正下标及乘号；每层/每门独立参数，不能解释为一个共享 router。 |
| 21 | 删除可选处理星号 | 符合 | 标题无星号，以 ι、α 表达开关/系数。 |
| 22 | 图中删除 GRIN、BiGRIL、DCRNN | 符合（人工阅读） | 主图未发现这些名称，也未发现 KD-tree；不是程序 OCR 证明。本文档讨论来源/核验时不受图中文字禁词限制。 |

### 推荐直接在 PPT 录入的关键式

```text
X ∈ ℝ^(L×N)               Ŷ ∈ ℝ^(H×N)
S_t = [R_t, a_t, m_t] ∈ ℝ^(N×6)
F_n = [X̄_:,n, S_1:L,n, τ/288, δ/7] ∈ ℝ^(L×9)
z_n^0 = [e_n || E^tod_(τ_L) || E^dow_(δ_L) || φ(g_n)] ∈ ℝ^160
T ∈ ℝ^(P×C×160), P=32
Stage 1: cross-partition (T_:,c)
Stage 2: within-partition (T′_p,:)
Q_p: 4 latent tokens

h_mod^(l) = (1+Δγ)⊙h_L^(l)+β,  l=1,2
y_0 = x̄_L
ỹ_k = y_k + αv
ŷ_k = ỹ_k + 0.05 s(R_L)⊙tanh(D_k)
D_k = Linear_2(GELU(Linear_1([ỹ_k, P⁺ỹ_k, P⁻ỹ_k])))

w_t = softmax(S_t W_g)
G(U,S_t) = Σ_(j=1)^5 w_(t,j) ⊙ (P_j U Θ_j) + b
r = σ(G_r([x,h],S))
u = σ(G_u([x,h],S))
h̃ = tanh(G_c([x,r⊙h],S))
h′ = u⊙h + (1−u)⊙h̃
```

以上是可复制的线性记法，正式排版按 method.tex 的数学形式输入，而非把下划线字符直接印进图。

## 第 0–4 节和素材专项检查

| 项目 | 核验结果 / 处理建议 |
|---|---|
| 一个连续总体图、一次输入、唯一最终输出 | 布局符合；不重复绘制一整套总体路径。主图部分长线仍终止在面板边缘，PPT 中须接到实际张量/算子。 |
| 两方向补全→插值→中心化→共享输出 | 位置顺序符合；共享 Xbar 的两条外出箭头未真正延伸到上下分支输入，需补实线。 |
| 输入三泳道 | 自然缺失空心点、Join 绿三角、Fault 红叉存在；四场景在上下文另给轨迹。 |
| 插值 | 素材中虚线直线连接缺失点，首末水平延拓存在；素材的 L 刻度仍落在最后观测附近，PPT 中把 L 移到右侧窗口终点，使右端延拓留在窗口内。 |
| 输出曲线 | curves 素材已为一个轴上三条蓝色节点曲线；主图仍画成每条历史后分出多条虚线，建议直接用素材最右区替换。不要解释为置信带。 |
| 同一变量同色 | 主图交通蓝、S 紫、隐状态青、FiLM 玫红总体已区分；图倍率仍误用玫红、静态图节点仍有多色，需按配色表修正。像素色号/15%透明度未获精确保证；PPT 中输入指定 HEX。 |
| 标题、字体、方向 | 采用衬线风格；部分小标题仍加粗，卷积文字水平。需按最终 5.5 in 宽度重设字号，不能用当前位图证明 ≥14 pt。 |
| Panel 背景与立体张量 | 符合 PPT 式浅色面板和小立体张量的方向；不采用极简矢量限制。 |
| 底部图例 | 加法、Hadamard、矩阵乘法、实线/紫虚线/灰点线、四色角色和六类数据颜色均有示意；正式文字重打。 |
| (a) 分区素材 | 四个圆润半透明分区块，四行六槽位示意，行色对应图块，右下一个斜线副本；P=32 文字说明真实分区数，不是四分区模型。 |
| (b) latent tokens | 四颗青色五角星、Gather 向星、Broadcast 向槽位，方向可辨。箭头概括汇聚/回传，不是完整逐对注意力连接。 |
| (c) Router | Linear→Softmax、五根不同颜色权重柱齐全。图柱非实测值，不作定量权重说明。 |
| (d) 专家热图 | 五个 6×6 方阵存在，I 主对角清楚，P₃/P₄外观相同、标签说明独立投影。但 P₂–P₅ 未精确遵守请求的非零格坐标，P₅不等于正确转置支持。**这些只可用作形状模板，不能直接当正确算子图标粘贴。** |
| (e) FiLM head | 160→32→128 宽窄宽形状，Z 青、Δγ/β 玫红并列输出，64 标签存在。 |
| (f) 时间卷积 | 蓝交通/紫上下文/金日历→收缩梯形→青 e_n(64)，1 patch/node 标签清楚。输出中的多个小格表示64维特征，不是多个 patch。 |
| (g) 角色符号 | AAS 蓝实心圆、NAS 向上绿三角、VS 灰虚线空心圆、FS 红叉，四项符合。 |
| (h) 双层图 GRU | 两行三列示意、同层时间递推、同时间层间上行、每个单元紫虚线条件、两层青色末状态，符合主要机制；图中省略中间时刻。 |
| 素材白底与裁切 | 模块表已去掉大面板底色和框线；保留图标本身的色块/分区透明块。图名在上方，可裁切时排除。 |
| 检查方法 | 人工检查渲染及 PNG 尺寸；未运行 OCR、颜色误差阈值、原生 PPT 或 SVG 审计。用户请求位图生图素材，没有创建矢量或 PPT 文件。 |

### 专家矩阵重画的准确依据

采用 method.tex 式(18)–(19)：
`P1=I`、`P2=rn(1[A>0])`、`P3=rn(A)`、`P4=rn(A)`、`P5=rn(Aᵀ)`。
P3 与 P4 必须使用同样的格点分布与数值图案，允许专家身份颜色不同；Θ3、Θ4 和路由权重仍独立。P5 是**原始邻接先转置再行归一化**，不是把归一化后的 P3 直接转置。

便于 PPT 手画的六节点示例（不是实验数据）：
非零邻接位置为 (1,2),(1,4),(2,3),(3,4),(4,5),(5,6),(6,1)；
反向非零位置为 (2,1),(4,1),(3,2),(4,3),(5,4),(6,5),(1,6)。
其余格保持白色，再按每行归一化着色。现有素材尚未逐格符合这个示例。

## 科学关系消歧

- 设计稿“紫虚线 OS 条件”不应引入方法中不存在的 S→FiLM 参数头直连。S 经 ALFTemporal 间接影响调制；Δγ/β→FiLM 是实线参数流。修正倍率只读取 R_L。
- 输入三泳道中的 Steady 自然缺失不改变 AAS 角色；NAS/VS/FS 不能使用图像模型自行扩写的“always observed/variable sampling”等文字。
- 星号已删除，表示论文采用同一结构、由 ι 和 α 控制，不表示四数据集数值相同。
- 早期架构图的 P/C 绘法不适用于本次：本次固定**行 P、列 C**，因此跨分区竖向、分区内横向。
- 窗口插值只读真实可见值；无观测保留 X^c，所有节点都先执行补全。参考恢复在解码后、图修正前；最终修正不进入自回归反馈。
- 本轮按用户要求只做绘图/标签校对，不更改模型或文稿。阅读粘贴说明的首批工具命令误包含一次只读 `git status --short`，已向用户说明；之后未再调用任何 Git 命令。既有文件未编辑、移动或删除，交付仅在 codex_out 新建。内置工具自行写入其媒体缓存后，原始输出被复制到本目录。

## 完整实际提示词

### 1. 整图 A

```text
Use case: scientific-educational / infographic-diagram. Output role: layout-and-style draft for author reconstruction in PowerPoint, no experimental data.
Generate a NEW crisp full figure at native 4096 × 2048 pixels, 2:1 landscape. Requested final width MUST be at least 3000 pixels. Use high-resolution detailed typography.

REFERENCE ROLES: Image 1 is project style reference 3.png: serif academic typography, tensor slabs and circled operators. Image 2 is project style reference 4.png: little isometric cube tensors, stars, clock/calendar, pastel nested mechanisms; borrow graphics only, not that model. Image 3 is current_ALFNet_render.png: existing ALFNet spatial organization and mechanisms, but its old labels, colors and missing connections MUST be replaced by the instructions below. Do not copy its misspellings.

A clean, professional machine-learning model architecture diagram for a NeurIPS paper, landscape 2:1, white background, PowerPoint style like the reference figures: pastel rounded panels with thin coloured borders, small isometric 3D cube tensors, rounded-rectangle layer boxes, circled operators (⊕ ⊙), thin black orthogonal arrows, a legend bar at the bottom. Flat icons, no photorealism, no drop shadows, crisp vector look, consistent serif font, all text horizontal and legible.

Layout from left to right:
- Left column “Inputs”, three stacked panels: (1) “Visible traffic history X” with three thin swim-lanes of smooth blue curves on a shared time axis 1…L: lane 1 a full blue curve with one hollow circle (natural missing), lane 2 grey dotted line then a blue curve for the last steps starting at a green triangle (newly added sensor), lane 3 a blue curve that stops at a red cross followed by a grey dotted line (failed sensor); (2) purple panel “Lifecycle context S = [R, a, M]” showing a strip of four-colour role squares, a purple staircase bar chart labelled a_t, and a black-and-white checker row labelled M_t, plus four short trajectory bars labelled Unseen, Steady, Join, Fault; (3) orange panel “Static graph A” with a small node-edge road graph.
- Second column, coral-bordered panel “SAHC”, top to bottom: “Bidirectional graph recurrent completion” with two rows of tiny graph-GRU cells (“Forward 1→L”, “Backward L→1”) merging into an “MLP fusion” box, output a blue cube slab labelled X^c; then “Window interpolation (ι)” with one curve whose missing points (hollow circles) are joined by dashed straight segments; then “Reference centering” with a circled minus and a yellow box “αv”; output a blue cube slab labelled X̄ that splits into two arrows to the right.
- Top-right, green-bordered panel “ALFTemporal”, left to right: stacked inputs (blue X̄ cubes, purple S cubes, gold clock and calendar icons for τ, δ) entering a trapezoid “Temporal conv (kernel = stride = L)”, output a short teal bar e_n; a long concatenated vector bar coloured teal | gold | gold | orange labelled z⁰ (160); a “Spatial partitions” module: a light road graph with three translucent coloured blobs and, next to it, a coloured P×C slot grid with one hatched cell; a “Two-stage attention” module: stage 1 highlights one column of the grid (“cross-partition”), stage 2 highlights one row connected to four teal stars (“within-partition, 4 latent tokens”) with gather and broadcast arrows; a teal bar stack “Z (N×160)” into a small bottleneck box “FiLM head”, outputs two rose bars Δγ and β.
- Bottom-right, blue-bordered panel “ALFGraph”, left to right: “Graph encoder” (two stacked rows of graph-GRU cells unrolled over t = 1…L, dashed purple arrows labelled S_t entering from above, teal hidden-state bars h⁽¹⁾_L, h⁽²⁾_L); a yellow “FiLM” box with ⊙ and ⊕ receiving Δγ and β from the ALFTemporal panel (one long arrow down); “Residual AR decoder” unrolled over k = 1…H with a feedback arrow y_{k−1} and a dashed purple S_L arrow, plus a thin grey dotted arrow labelled “Teacher Forcing (train)”; a small “Restore +αv” box; a peach “Role-scaled graph refinement” stack (inputs ỹ_k, P⁺ỹ_k, P⁻ỹ_k → Linear–GELU–Linear → tanh → ×0.05 s(R_L) → ⊕); final output panel “Ŷ” showing a short solid history curve, a vertical dashed line at t = L and dashed forecast curves.
- Below the encoder, a dashed zoom-in box “Graph GRU cell with OS-routed mixture of graph experts”: S_t → “Linear → Softmax” router with a five-bar weight chart w_t → five small matrix heatmaps P₁…P₅ (each times Θ_j) → circled Σ → G; on the right four small gate boxes r, u, h̃, h′.
- Bottom legend bar: ⊕ addition, ⊙ Hadamard product, solid arrow = data flow, dashed purple = OS conditioning, dotted grey = training only, role glyphs (AAS ●, NAS ▲, VS ◌, FS ✗), and colour key: blue traffic values, purple lifecycle context, gold calendar, orange graph structure, teal hidden states, rose FiLM parameters.

Colour palette (use exactly): blue #2E75B6 / #BDD7EE; purple #7F5FA8 / #E4DFEC; gold #BF9000 / #FFF2CC; orange #C55A11 / #FBE5D6; teal #31859C / #DAEEF3; rose #C0508A / #F2D0E3; green triangle #70AD47; red cross #C00000; panel backgrounds very light with borders coral #E8857A, sage #93C47D, steel blue #6FA8DC.

Do not include the words GRIN, BiGRIL, DCRNN or KD-tree anywhere. Do not add modules that are not in the spec.

AUTHORITATIVE DESIGN AND METHOD CLARIFICATIONS (override older image labels and ambiguities):
- Color identifies DATA, not enclosing panel. Traffic X,X^c,Xbar,y,Ỹ,Ŷ always blue; lifecycle S and conditions purple; calendar gold; graph A and structure embeddings orange; hidden h,rho,e,T,Z,B and learned latent stars teal; FiLM parameters rose. Five expert identity colors and partition identity colors are local icon exceptions. ALL text Times New Roman-like serif, horizontal; bold only top-level panel titles.
- Three inputs drawn once; three swim lanes only, not four. Natural missing in steady does NOT change role. Join begins with green triangle, Fault ends at red cross. Lifecycle labels exactly "R_t: observation role", "a_t: state progress", "M_t: mask". The black-white mask is NOT purple-filled.
- SAHC runs all N nodes through both independent directional recurrent paths. Show purple S and orange A named ports into completion. MLP label "MLP fusion of ρ→_t, ρ←_t, m_t"; keep observed values to obtain X^c, all blue. Interpolation title "Window interpolation (ι)"; two exact short notes "q_n > 0: linear fill of missing slots" / "q_n = 0: keep X^c". "Shared history X̄". NO asterisks on any processing title. Alpha-v derives from last actually observed history, not from a hidden state or from filled values.
- Temporal F_n uses normalized calendar, label F_n = [X̄, S, τ/288, δ/7]; "Temporal conv (kernel = stride = L)" yields "e_n (64)" and EXACTLY ONE temporal patch per node. Other concatenation components are labeled "Time-of-day E^tod", "Day-of-week E^dow", and "φ(g)" from A through structural MLP 7→64→32. z⁰_n = e_n || E^tod || E^dow || φ(g), 160 = 64+32+32+32. NOT a time branch forecast.
- Partition grid MUST use ROWS = partitions p, COLUMNS = node slots c, P=32 only a label; draw representative rows with ellipses, one last hatched slot denotes actual-node duplicate context only, not zero padding or extra output. "T ∈ ℝ^{P×C×160}". Stage 1: VERTICAL column highlighted "cross-partition (T_{:,c})". Stage 2: HORIZONTAL row "within-partition (T′_{p,:})", Gather → EXACTLY FOUR teal stars "Q_p: 4 latent tokens" → Broadcast. Restore node order BEFORE "Z (N×160)".
- FiLM HEAD only takes Z, outputs Δγ and β (each N×64), no direct S→FiLM-head edge. Parameter connector is a solid data arrow and must land on yellow FiLM box between encoder and decoder, not on decoder. FiLM formula "h_mod^(l) = (1+Δγ)⊙h_L^(l)+β"; two encoder terminal states h_L^(1),h_L^(2), each N×64, both modulated once.
- Decoder starts y_0=xbar_L, uses S_L (subscript). Show TWO recurrent layers, feedback from unrefined y_k not hidden h or final Ŷ. Caption y_k = y_(k−1) + d Δy_k for inference. Teacher-forcing grey dotted arrow from tiny training-target label into decoder input ONLY. No new loss panel.
- After Restore +αv, label blue input to refinement "Ỹ" (Y with tilde, not Y_0). Refiner three blue traffic slabs "[ỹ_k, P⁺ỹ_k, P⁻ỹ_k]", orange A port; then "Linear–GELU–Linear" → "tanh" → purple multiplier "×0.05 s(R_L)" → circled plus with an ORIGINAL ỹ_k residual bypass. Final result alone enters Ŷ curve. NO bypass from Restore directly to final forecast. R_L only drives scale, no full S into refiner.
- Zoom-in has NO data-flow arrow from FiLM or decoder to itself; a fine no-arrow leader from encoder/decoder indicates same cell definition. Title "Graph GRU cell with OS-routed mixture of graph experts (encoder & decoder)". S_t→Linear→Softmax gives weights, feature U goes separately to every expert. Expert labels exactly "P₁ = I", "P₂ binary", "P₃ weighted", "P₄ forward", "P₅ backward". P₃ and P₄ show the SAME pattern but different identity tints, because operators coincide but learnable projections differ.
- Zoom formulas: "w_t = softmax(S_t W_g)"; "G(U,S_t) = Σ_j w_(t,j) ⊙ (P_j U Θ_j) + b". GRU only candidate symbol h̃, never c: "r = σ(G_r([x,h],S))"; "u = σ(G_u([x,h],S))"; "h̃ = tanh(G_c([x,r⊙h],S))"; "h′ = u⊙h + (1−u)⊙h̃". Footer "Encoder: S_t; Decoder: S_L; 12 modules (2 layers × 3 gates × enc/dec)".
- Bottom legend includes ⊗ matrix multiplication in addition to ⊕ and ⊙. Role glyph VS is dashed HOLLOW grey circle; AAS filled blue circle, NAS green triangle, FS red cross.
- Exactly one continuous architecture, no duplicated overview, no repeated inputs, no arbitrary extra modules. Keep color legend and all panels within canvas, no overlaps. Curves are conceptual examples. Spacious gutters for long FiLM connector. Prioritize structural accuracy and readable diagram over decoration. Do not include the prohibited names GRIN, BiGRIL, DCRNN or KD-tree anywhere.
```

### 2. 曲线首版

```text
Use case: infographic-diagram, clean scientific chart ASSETS for a NeurIPS architecture schematic, not experimental results. Produce ONE high-resolution PNG sheet at native 3600 × 1800 or larger, pure opaque white background. Exactly THREE separate equal-sized white chart tiles in one horizontal row, with generous white space between and all around them, no border joining tiles and no text whatsoever except tiny optional 1,L,H axis ticks. No title, no labels, no legends, no panel letters.

Reference image 1 (1.png) panel (a): smooth curves with tinted areas, thin clean axes. Reference image 2 (6.png) panel (a): consistent thin solid/dashed line treatment. Reference image 3 (7.png) decoder outputs: a few distinct smooth waves. Borrow only chart styling, not mechanisms or source data.

Three separate small chart tiles on a pure white background, flat vector style, matching reference figure 1 (a): smooth spline lines 2 px thick with a light 15% tinted area underneath, tiny filled circles at observed points, hollow circles at missing points, grey dotted segments where data is missing, no titles, only small axis ticks. Tile 1: three stacked swim-lanes as described for “Visible traffic history”. Tile 2: one interpolation curve, missing points joined by dashed straight segments, horizontal dashed extensions before the first and after the last observed point. Tile 3: forecast tile, a short solid blue history curve with a light blue fill, a vertical dashed line, then three slightly offset dashed forecast curves in blue shades.

EXACT DETAILS:
Left tile: THREE vertically stacked small swim-lanes sharing horizontal alignment and the time range 1 to L. All observed curves BLUE #2E75B6 with very light #BDD7EE (15% opacity) area to each local baseline, never fill missing intervals. Lane 1 blue smooth curve entire time range, small filled blue observations and exactly one hollow white-centered circle for natural missing. Lane 2 grey dotted guide at early times, then one green #70AD47 triangle at the arrival point and blue observed curve for the rest of time. Lane 3 blue observed curve early, then one red #C00000 cross at failure time and ONLY grey dotted guide later; NO blue continuation after failure. A grey dotted guide is only a missingness cue, not known ground truth. Nothing is written beside the lanes.
Middle tile: ONE interpolation curve with blue observed endpoints and small solid blue observed circles, white-centered hollow blue circles ONLY at interpolated sample times. Blue dashed STRAIGHT LINE segments between flanking observed anchors across two internal gaps. Before FIRST observed point and after LAST observed point, BLUE HORIZONTAL dashed extrapolation to window boundaries, exactly constant value of nearest observed endpoint. Distinguish grey missingness from blue interpolation. No green or orange interpolated values; all traffic stays blue. Optional low-opacity blue fill ONLY under observed solid sections. No smooth spline over an interpolated gap.
Right tile: Three vertically slightly offset blue-shaded node traces, each with short SOLID history at left and DASHED forecasts at right. All three histories end at same vertical grey dashed separator at L; all three future lines start exactly at their own history endpoint, from L to L+H. NO forecast confidence band and no uncertainty interpretation: only three deterministic illustrative node curves. Low-opacity blue area may appear under history only. No circles on forecast curves, no mean forecast line, no text except tiny ticks.

Style: flat, crisp vector-like PPT curve drawing, line 2–2.25 pt visual weight, axis thin grey/black, no photographic elements, no gradients beyond solid translucent fill, pure white wide crop margins. All curves fabricated illustrative diagrams, not measurements. Keep all three complete assets within canvas.
```

### 3. 模块首版

```text
Use case: scientific-educational. Asset sheet for cropping into a manually rebuilt NeurIPS/PPT architecture figure. Generate one high-resolution 3200 × 2800 PNG (native resolution if supported), opaque pure white canvas. EIGHT independent small modules in a neat 4-row × 2-column grid, separated by very generous blank white gutters. No enclosing full-figure architecture, no lines crossing between tiles, no decorative sheet title. Tiny tile letters (a)–(h) and short functional labels allowed. One consistent Times New Roman serif font. Crisp vector-like outlines, pale rounded boxes, thin isometric cube side faces, flat icons. No drop shadows, photographs or unrequested mechanisms.

References: image 1 reference 4.png for stars, triangles, tensor bars and weight-chart vocabulary ONLY; image 2 reference 3.png for matrices and graph symbols ONLY; image 3 reference 1.png panel(c) for partition blobs over a simple road graph ONLY.

Palette: traffic blue #2E75B6/#BDD7EE; S purple #7F5FA8/#E4DFEC; calendar gold #BF9000/#FFF2CC; static graph orange #C55A11/#FBE5D6; hidden features and latent tokens teal #31859C/#DAEEF3; FiLM rose #C0508A/#F2D0E3; NAS green #70AD47; FS red #C00000. Local partition identities and five expert identities may have distinct pastel tints. All neutral operators white with black outline.

(a) TOP LEFT: SPATIAL PARTITIONS. Left half: a connected light-grey road graph of ~12 little circular nodes with 3–4 translucent rounded irregular blobs behind node clusters (sage, lavender, peach, pale blue). Node outlines match their partition blob. Arrow to right half: illustrative P×C SLOT GRID, draw 4 rows ×6 columns with row identity colors matching blobs and small matching color dots at each row start. ROWS=partitions p, COLUMNS=slots c. Label left axis p, top axis c, corner "P = 32", under grid "T ∈ ℝ^{P×C×160}". Exactly ONE diagonally hatched LAST cell bottom-right denotes a copied real node for context, not empty zero padding. Row fills should make partition identities clear. No KD-tree graphics or named partitioning algorithm.

(b) TOP RIGHT: 4 LATENT TOKENS. Top row of SIX small teal feature slots, central row of EXACTLY FOUR solid teal FIVE-POINTED STARS, bottom row of SIX output slots. Downward arrows from upper slots into stars labeled "Gather"; then separate downward arrows from stars to output slots labeled "Broadcast". Readable short caption "Q_p: 4 latent tokens". No fifth star, no triangles instead of stars, no directionless double arrows.

(c) SECOND ROW LEFT: ROUTER. Purple input S_t → white rounded box "Linear → Softmax" → FIVE slim weight bars, all heights unequal; one identity pastel per expert in order pale blue / peach / sage / lavender / gold. Prefix/label w_t in purple. Five bars collectively one distribution; no axis numbers needed, do not invent a sixth. Keep all 5 present (dense routing).

(d) SECOND ROW RIGHT: FIVE EXPERT MATRICES in a horizontal strip, each matrix EXACTLY 6×6 equal squares, a short label above:
"P₁ = I", "P₂ binary", "P₃ weighted", "P₄ forward", "P₅ backward".
P1: only main diagonal cells filled, other cells white.
For P2–P4 use SAME nonzero adjacency support at row,column (1,2),(1,4),(2,3),(3,4),(4,5),(5,6),(6,1); zeros remain white. P2 binary-derived row normalization: row1 entries equal tint, all one-neighbor rows full tint. P3 weighted: row1 (1,2) lighter and (1,4) darker, other listed entries strong tint. P4 same support and relative intensities as P3, but different expert identity hue. P5 uses transposed support, row-normalized; keep sparse transpose pattern and independent tint. All FIVE matrices must have distinct pastel identity hue matching tile(c) bars; P3/P4 SAME numerical pattern, different identity hue is allowed. Little footer "P₃ = P₄; separate projections". Do not show values or claim empirical adjacency; symbolic example only.

(e) THIRD ROW LEFT: FiLM HEAD. Teal input vertical vector Z, then three connected trapezoidal/short-layer shapes visibly wide–narrow–wide, widths correspond "160 → 32 → 128". Arrow splits into TWO PARALLEL vertical output feature bars, dark rose "Δγ", light rose "β"; both labeled "64". NOT a serial Δγ→β chain. Gold background of tiny head enclosure permitted, data colors unchanged.

(f) THIRD ROW RIGHT: TEMPORAL CONV. Blue traffic slab + purple context slab + little gold calendar strip enter a WIDE-mouth-to-NARROW-mouth trapezoid, literal horizontal label inside "Temporal conv". Below "kernel = stride = L". Small bracket labels temporal extent "L" at input; output arrow to EXACTLY ONE teal patch vector, caption "e_n (64)" and "1 patch / node". Do not paint output traffic blue; it is learned feature teal.

(g) BOTTOM LEFT: FOUR ROLE GLYPHS, large separated and individually crop-able, each with its exact short label beneath:
AAS solid BLUE circle; NAS solid GREEN UPWARD-POINTING TRIANGLE ▲; VS GREY DASHED-OUTLINE HOLLOW circle with pure white center; FS RED cross ✗. No solid grey VS, no green circle, no red plus.

(h) BOTTOM RIGHT: TWO-LAYER GRAPH GRU unrolled over time. Two HORIZONTAL rows × THREE time columns of white rounded Graph-GRU cells. Bottom row Layer 1, top row Layer 2. Time labels 1, …, L above columns. Each row has left-to-right solid black recurrence arrows. Each Layer1 cell has a solid upward arrow to its Layer2 cell at SAME time (not diagonal). Purple S_t conditioning bus above each row has dashed purple DOWNWARD arrows entering EACH of the six cell boxes from above; route around interlayer arrows without crossing text. Hidden-state output bars at the right of EACH row are TEAL. Text inside cells simply "Graph GRU", not long formulas. Use small graph-node glyph on cell corners only if space; 2 layers are actual layers, columns are times. No third recurrent row, no bidirectional time arrows.

All eight tiles fully inside canvas with generous white crop margins. Pure white background, no extra frames around sheet, no watermark. These are conceptual diagram assets; do not add confidence intervals or numerical data. Prohibited labels anywhere: GRIN, BiGRIL, DCRNN, KD-tree.
```

### 4. 整图 B

```text
SECOND INDEPENDENT LAYOUT VARIANT. Please generate the image at 4096×2048 native pixels (at least 3000 pixels wide). The previous call returned only 1774×887 despite the requested resolution. Use the largest supported native image output. This is a fresh redraw from the three original references, not an edit of the previous generated figure.

CRITICAL LAYOUT REQUIREMENTS, enforce these before any fine details:
1. All traffic history dimensions MUST be L×N, not N×L×1 or 12×N×1. Final forecast H×N, no extra channel axis.
2. In SAHC, a solid vertical pipeline: bidirectional completion → X^c → interpolation → Xfill → circled MINUS → Xbar. The gold αv chip is a SIDE INPUT to the minus circle, never a serial step AFTER the minus. Label its wire αv and derive v from observed X and M. The minus output alone is Xbar.
3. Xbar forks at a visible junction to BOTH Temporal input and graph encoder; the upper arrow must visibly end at the Temporal blue Xbar tensor. Input S to the encoder must be purple, A orange. Do not place A next to a purple slab.
4. Only Xbar, S, normalized τ/δ enter temporal convolution. Calendar embeddings E^tod/E^dow and structure φ(g) BYPASS the convolution and enter concat together with e_n. Show 4 separately connected sources into the concatenated bar. This is essential.
5. Node partition grid row colors match graph blobs: each representative row one partition color, not multi-color rows. P=32, C columns; ONE hatched duplicate slot, not several.
6. Head: Z, e_n, hidden states are TEAL, never traffic blue. Fused history X^c and Xbar remain BLUE. Delta-gamma and beta ROSE. Use color by variable everywhere.
7. Encoder = TWO ROWS × THREE TIME COLUMNS of recurrent cells, bottom layer1 and top layer2; left-to-right time arrows in each row, bottom-to-top layer arrows in each column. TWO teal terminal states on right. Decoder show two rows too, y_(k−1) feedback from its blue step prediction. Parameter connector lands on the gold FiLM BETWEEN them.
8. NO confidence band, no uncertainty, no mean-forecast legend at output. Exactly three blue dashed deterministic illustrative node forecasts after one separator L and short solid history before it. Same blue semantic color, only slight lightness differences.
9. Zoom inset fits ONLY under Graph encoder / FiLM / decoder, NEVER under or across SAHC. Fine dashed no-arrow callout from encoder, not a causal line from SAHC. All other long wires use reserved gutters.
10. Restored blue Ỹ (capital tilde-Y), then graph correction, then ONE output. Residual arrow uses ỹ_k into final plus. No Y_0 labels, no alpha reference mixed with prediction residual.
11. Text horizontal. Only top-level titles bold. No numbered subpanel headers needed. Use measured whitespace rather than cluttered thick borders. Every arrow has an obvious endpoint.
12. Graph expert matrices P3/P4 have same pattern. Keep all five learnable expert transforms AND a separate feature U input; weights are not data passed serially through matrices.

Use case: scientific-educational / infographic-diagram. Output role: layout-and-style draft for author reconstruction in PowerPoint, no experimental data.
Generate a NEW crisp full figure at native 4096 × 2048 pixels, 2:1 landscape. Requested final width MUST be at least 3000 pixels. Use high-resolution detailed typography.

REFERENCE ROLES: Image 1 is project style reference 3.png: serif academic typography, tensor slabs and circled operators. Image 2 is project style reference 4.png: little isometric cube tensors, stars, clock/calendar, pastel nested mechanisms; borrow graphics only, not that model. Image 3 is current_ALFNet_render.png: existing ALFNet spatial organization and mechanisms, but its old labels, colors and missing connections MUST be replaced by the instructions below. Do not copy its misspellings.

A clean, professional machine-learning model architecture diagram for a NeurIPS paper, landscape 2:1, white background, PowerPoint style like the reference figures: pastel rounded panels with thin coloured borders, small isometric 3D cube tensors, rounded-rectangle layer boxes, circled operators (⊕ ⊙), thin black orthogonal arrows, a legend bar at the bottom. Flat icons, no photorealism, no drop shadows, crisp vector look, consistent serif font, all text horizontal and legible.

Layout from left to right:
- Left column “Inputs”, three stacked panels: (1) “Visible traffic history X” with three thin swim-lanes of smooth blue curves on a shared time axis 1…L: lane 1 a full blue curve with one hollow circle (natural missing), lane 2 grey dotted line then a blue curve for the last steps starting at a green triangle (newly added sensor), lane 3 a blue curve that stops at a red cross followed by a grey dotted line (failed sensor); (2) purple panel “Lifecycle context S = [R, a, M]” showing a strip of four-colour role squares, a purple staircase bar chart labelled a_t, and a black-and-white checker row labelled M_t, plus four short trajectory bars labelled Unseen, Steady, Join, Fault; (3) orange panel “Static graph A” with a small node-edge road graph.
- Second column, coral-bordered panel “SAHC”, top to bottom: “Bidirectional graph recurrent completion” with two rows of tiny graph-GRU cells (“Forward 1→L”, “Backward L→1”) merging into an “MLP fusion” box, output a blue cube slab labelled X^c; then “Window interpolation (ι)” with one curve whose missing points (hollow circles) are joined by dashed straight segments; then “Reference centering” with a circled minus and a yellow box “αv”; output a blue cube slab labelled X̄ that splits into two arrows to the right.
- Top-right, green-bordered panel “ALFTemporal”, left to right: stacked inputs (blue X̄ cubes, purple S cubes, gold clock and calendar icons for τ, δ) entering a trapezoid “Temporal conv (kernel = stride = L)”, output a short teal bar e_n; a long concatenated vector bar coloured teal | gold | gold | orange labelled z⁰ (160); a “Spatial partitions” module: a light road graph with three translucent coloured blobs and, next to it, a coloured P×C slot grid with one hatched cell; a “Two-stage attention” module: stage 1 highlights one column of the grid (“cross-partition”), stage 2 highlights one row connected to four teal stars (“within-partition, 4 latent tokens”) with gather and broadcast arrows; a teal bar stack “Z (N×160)” into a small bottleneck box “FiLM head”, outputs two rose bars Δγ and β.
- Bottom-right, blue-bordered panel “ALFGraph”, left to right: “Graph encoder” (two stacked rows of graph-GRU cells unrolled over t = 1…L, dashed purple arrows labelled S_t entering from above, teal hidden-state bars h⁽¹⁾_L, h⁽²⁾_L); a yellow “FiLM” box with ⊙ and ⊕ receiving Δγ and β from the ALFTemporal panel (one long arrow down); “Residual AR decoder” unrolled over k = 1…H with a feedback arrow y_{k−1} and a dashed purple S_L arrow, plus a thin grey dotted arrow labelled “Teacher Forcing (train)”; a small “Restore +αv” box; a peach “Role-scaled graph refinement” stack (inputs ỹ_k, P⁺ỹ_k, P⁻ỹ_k → Linear–GELU–Linear → tanh → ×0.05 s(R_L) → ⊕); final output panel “Ŷ” showing a short solid history curve, a vertical dashed line at t = L and dashed forecast curves.
- Below the encoder, a dashed zoom-in box “Graph GRU cell with OS-routed mixture of graph experts”: S_t → “Linear → Softmax” router with a five-bar weight chart w_t → five small matrix heatmaps P₁…P₅ (each times Θ_j) → circled Σ → G; on the right four small gate boxes r, u, h̃, h′.
- Bottom legend bar: ⊕ addition, ⊙ Hadamard product, solid arrow = data flow, dashed purple = OS conditioning, dotted grey = training only, role glyphs (AAS ●, NAS ▲, VS ◌, FS ✗), and colour key: blue traffic values, purple lifecycle context, gold calendar, orange graph structure, teal hidden states, rose FiLM parameters.

Colour palette (use exactly): blue #2E75B6 / #BDD7EE; purple #7F5FA8 / #E4DFEC; gold #BF9000 / #FFF2CC; orange #C55A11 / #FBE5D6; teal #31859C / #DAEEF3; rose #C0508A / #F2D0E3; green triangle #70AD47; red cross #C00000; panel backgrounds very light with borders coral #E8857A, sage #93C47D, steel blue #6FA8DC.

Do not include the words GRIN, BiGRIL, DCRNN or KD-tree anywhere. Do not add modules that are not in the spec.

AUTHORITATIVE DESIGN AND METHOD CLARIFICATIONS (override older image labels and ambiguities):
- Color identifies DATA, not enclosing panel. Traffic X,X^c,Xbar,y,Ỹ,Ŷ always blue; lifecycle S and conditions purple; calendar gold; graph A and structure embeddings orange; hidden h,rho,e,T,Z,B and learned latent stars teal; FiLM parameters rose. Five expert identity colors and partition identity colors are local icon exceptions. ALL text Times New Roman-like serif, horizontal; bold only top-level panel titles.
- Three inputs drawn once; three swim lanes only, not four. Natural missing in steady does NOT change role. Join begins with green triangle, Fault ends at red cross. Lifecycle labels exactly "R_t: observation role", "a_t: state progress", "M_t: mask". The black-white mask is NOT purple-filled.
- SAHC runs all N nodes through both independent directional recurrent paths. Show purple S and orange A named ports into completion. MLP label "MLP fusion of ρ→_t, ρ←_t, m_t"; keep observed values to obtain X^c, all blue. Interpolation title "Window interpolation (ι)"; two exact short notes "q_n > 0: linear fill of missing slots" / "q_n = 0: keep X^c". "Shared history X̄". NO asterisks on any processing title. Alpha-v derives from last actually observed history, not from a hidden state or from filled values.
- Temporal F_n uses normalized calendar, label F_n = [X̄, S, τ/288, δ/7]; "Temporal conv (kernel = stride = L)" yields "e_n (64)" and EXACTLY ONE temporal patch per node. Other concatenation components are labeled "Time-of-day E^tod", "Day-of-week E^dow", and "φ(g)" from A through structural MLP 7→64→32. z⁰_n = e_n || E^tod || E^dow || φ(g), 160 = 64+32+32+32. NOT a time branch forecast.
- Partition grid MUST use ROWS = partitions p, COLUMNS = node slots c, P=32 only a label; draw representative rows with ellipses, one last hatched slot denotes actual-node duplicate context only, not zero padding or extra output. "T ∈ ℝ^{P×C×160}". Stage 1: VERTICAL column highlighted "cross-partition (T_{:,c})". Stage 2: HORIZONTAL row "within-partition (T′_{p,:})", Gather → EXACTLY FOUR teal stars "Q_p: 4 latent tokens" → Broadcast. Restore node order BEFORE "Z (N×160)".
- FiLM HEAD only takes Z, outputs Δγ and β (each N×64), no direct S→FiLM-head edge. Parameter connector is a solid data arrow and must land on yellow FiLM box between encoder and decoder, not on decoder. FiLM formula "h_mod^(l) = (1+Δγ)⊙h_L^(l)+β"; two encoder terminal states h_L^(1),h_L^(2), each N×64, both modulated once.
- Decoder starts y_0=xbar_L, uses S_L (subscript). Show TWO recurrent layers, feedback from unrefined y_k not hidden h or final Ŷ. Caption y_k = y_(k−1) + d Δy_k for inference. Teacher-forcing grey dotted arrow from tiny training-target label into decoder input ONLY. No new loss panel.
- After Restore +αv, label blue input to refinement "Ỹ" (Y with tilde, not Y_0). Refiner three blue traffic slabs "[ỹ_k, P⁺ỹ_k, P⁻ỹ_k]", orange A port; then "Linear–GELU–Linear" → "tanh" → purple multiplier "×0.05 s(R_L)" → circled plus with an ORIGINAL ỹ_k residual bypass. Final result alone enters Ŷ curve. NO bypass from Restore directly to final forecast. R_L only drives scale, no full S into refiner.
- Zoom-in has NO data-flow arrow from FiLM or decoder to itself; a fine no-arrow leader from encoder/decoder indicates same cell definition. Title "Graph GRU cell with OS-routed mixture of graph experts (encoder & decoder)". S_t→Linear→Softmax gives weights, feature U goes separately to every expert. Expert labels exactly "P₁ = I", "P₂ binary", "P₃ weighted", "P₄ forward", "P₅ backward". P₃ and P₄ show the SAME pattern but different identity tints, because operators coincide but learnable projections differ.
- Zoom formulas: "w_t = softmax(S_t W_g)"; "G(U,S_t) = Σ_j w_(t,j) ⊙ (P_j U Θ_j) + b". GRU only candidate symbol h̃, never c: "r = σ(G_r([x,h],S))"; "u = σ(G_u([x,h],S))"; "h̃ = tanh(G_c([x,r⊙h],S))"; "h′ = u⊙h + (1−u)⊙h̃". Footer "Encoder: S_t; Decoder: S_L; 12 modules (2 layers × 3 gates × enc/dec)".
- Bottom legend includes ⊗ matrix multiplication in addition to ⊕ and ⊙. Role glyph VS is dashed HOLLOW grey circle; AAS filled blue circle, NAS green triangle, FS red cross.
- Exactly one continuous architecture, no duplicated overview, no repeated inputs, no arbitrary extra modules. Keep color legend and all panels within canvas, no overlaps. Curves are conceptual examples. Spacious gutters for long FiLM connector. Prioritize structural accuracy and readable diagram over decoration. Do not include the prohibited names GRIN, BiGRIL, DCRNN or KD-tree anywhere.
```

### 5. 曲线修正

```text
Edit the supplied THREE-TILE scientific curve asset sheet while retaining its crispness, pure white canvas, blue palette, thin axes, smooth observed curves, and ample crop margins. Do not add titles, labels, legends, or lettering beyond small axis ticks.
1. LEFT tile: keep all three swim lanes. Change ONLY the green arrival triangle in the middle lane to point UP (▲), same position, green #70AD47. Preserve natural missing hollow circle, red failure cross, all observed-blue vs missing-grey styles.
2. MIDDLE tile: retain observed points, two internal gaps, constant dashed horizontal extensions at beginning/end. Make EVERY internal blue dashed interpolation gap a mathematically STRAIGHT segment between its two solid observed endpoints, with hollow points exactly on the segment. No curved interpolation, no smoothing across gaps.
3. RIGHT tile: replace its three separate axes with ONE SINGLE COMPACT FORECAST PANEL: one horizontal time axis, one vertical value axis, one vertical dashed separator at L. In this SAME coordinate system, draw THREE slightly vertically offset blue-shade history lines, each solid before L and dashed after L, all three continuous at L. Three dashed forecasts distinguish three different nodes; no confidence band, no mean prediction, no uncertainty. Pale blue fill below short history only, not below future. Do not use three separate subpanels. Tiny tick labels 1, L, L+H only, no Node labels.
All three asset tiles remain in one horizontal row on white. Render at highest native resolution with clean vector-like lines.
```

### 6. 模块修正

```text
Edit ONLY the supplied eight-tile scientific module asset sheet. Preserve every icon's crisp black outlines, isometric feature cubes, exact serif labels, positions, sizes, eight-tile 4×2 grid, data colors, four stars, role glyphs and graph-GRU wiring. Keep high native resolution.

MANDATORY WHITE ASSET SHEET: remove ALL eight large rounded background rectangles and their colored border outlines. Remove ALL tinted washes / gradients behind modules. Everything outside the actual small icons must be pure white, with clear white gutters between eight modules and wide outer margins. Keep ONLY the small operator boxes and tensor fills. Keep tiny (a)–(h) and short titles if needed but use regular serif rather than giant bold titles. Do not remove actual translucent partition blobs, colored grid cells or colored data shapes.

MATRIX REPAIR IN (d): Each of the FIVE matrix icons must remain EXACTLY SIX rows by SIX columns. Keep labels P1=I, P2 binary, P3 weighted, P4 forward, P5 backward. P1 has exactly six main-diagonal colored cells. Use the following exact nonzero row,column positions (all other cells WHITE):
P2, P3 and P4: (1,2),(1,4),(2,3),(3,4),(4,5),(5,6),(6,1).
P2 row1's two cells equal intensity; P3/P4 row1 first lighter and second darker. P3 and P4 identical pattern and intensities, only different hue. P5 BACKWARD must use TRANSPOSED nonzero positions: (2,1),(4,1),(3,2),(4,3),(5,4),(6,5),(1,6). These are diagrammatic heatmaps, do not write coordinate lists or numbers inside. Preserve 'P₃ = P₄; separate projections' caption.
Tile (h): two output labels at right use history-terminal subscript L, h_L^(2), h_L^(1), not t. No other diagram edits. Do not introduce missing modules or extra rows/columns.
```

### 7. 选定整图修正

```text
Perform ONE final targeted correction of image 1 (the complete ALFNet draft). Preserve its overall two-column inputs/SAHC and upper-temporal/lower-graph layout, serif font, clear pastel panels, cube tensors, colors, all unchanged labels, and all original scientific mechanisms. Image 2 is a reference ONLY for the correct single-axis three-curve forecast icon. Native canvas request 4096×2048, at least 3000 pixels wide if supported. Do not add new subplots or modules.

Make these corrections with highest priority:
A. The Stage 2 latent tokens currently shows THREE teal stars while caption says four. Replace with EXACTLY FOUR teal stars aligned side by side. Gather arrows enter all four; Broadcast returns to slots. Keep Q_p: 4 latent tokens.
B. Graph refinement and forecast MUST connect correctly. Remove erroneous A symbols over traffic inputs. Restore +αv output is Ỹ (capital Y with tilde). Route it to three blue slabs labeled [ỹ_k, P⁺ỹ_k, P⁻ỹ_k], then Linear–GELU–Linear → tanh → ×0.05 s(R_L) → PLUS. The original ỹ_k has ONE short residual bypass to the left input of that plus. Draw a solid arrow from the RIGHT side of that plus into the final forecast panel, with a clear arrowhead pointing RIGHT. The small ORANGE A road-graph port feeds only the neighbor-propagation stage, never the final plus. No missing output wire, no bypass that skips refinement, no Y_0.
C. Temporal FiLM parameter line must be DARK SOLID, not rose/purple. It ends at the gold FiLM box between encoder and decoder. The two parameter BARS themselves stay rose. Refiner role scale parameter block should be PURPLE, not rose.
D. In SAHC Reference centering panel: BLUE input tensor above minus is X^fill; output below minus is Xbar. Gold αv is a SIDE INPUT to minus; keep the actual subtraction connection. Under output use literal 'Shared history Xbar'. The fork labeled To ALFTemporal and To ALFGraph must lead, through a reserved white gutter, into the actual blue input tensor of each branch. Do not simply terminate at the outer module border.
E. All full lifecycle sequence tensors S have dimension L×N×6 (not L×N); last-step context S_L has N×6 (not N). X has L×N, output Ŷ H×N, hidden h_L^(1),h_L^(2) N×64. The calendar input sequences are length L, not L×N, while broadcasting over nodes is implicit. Do not change F_n dimension L×9.
F. Remove invented parenthetical expansions after the four role labels in the Inputs panel: show ONLY AAS, NAS, VS, FS and their four correct glyphs, plus existing R_t: observation role. In particular never call VS 'variable sampling' or AAS 'always observed'. Keep Steady's natural missing hole.
G. Correct zoom title to exact text 'Graph GRU cell with OS-routed mixture of graph experts (encoder & decoder)', not QS-routed. Candidate is only h̃; no c candidate. Main weight formula uses P_j U Θ_j, not P_j U_j. Keep both feature-U and weight inputs clearly distinct.
H. Final output icon: borrow RIGHTMOST tile of image 2, a SINGLE coordinate system containing THREE blue-shaded slightly vertically offset curves, each solid history before shared vertical L line and dashed forecast after L. No paired mean/uncertainty curves, confidence fill, or invented uncertainty legend. Only history has very light blue area. Ŷ ∈ ℝ^{H×N}.

Do not erase any of the shared-history completion, optional interpolation switch iota, graph correction, two recurrent layers, Z node-order restoration, five experts, or bottom legend. Do not introduce forbidden names GRIN, BiGRIL, DCRNN, KD-tree. Prioritize accurate connectors, star count and variable dimensions.
```


