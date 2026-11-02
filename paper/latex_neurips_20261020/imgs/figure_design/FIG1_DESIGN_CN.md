# 图 1（ALFNet 架构图）重绘方案

依据：方法章定稿（`sections/method.tex`，式 1–30）、当前 `imgs/ALFNet.svg`（渲染见 `current_ALFNet_render.png`）、同门参考图 `imgs/Reference_Image/1–7.png`。

## 0. 总体原则

1. **颜色表示“是什么数据”，不表示“在哪个模块”。** 同一变量从输入到输出始终同色；拼接时按来源分段着色（例如 z⁰ = 蓝绿段 e_n ‖ 金色 E^tod ‖ 金色 E^dow ‖ 橙色 φ(g)），读者一眼能看出信息从哪里来。
2. **形状表示“张量的结构”。** 时间×节点的张量用立方体板，逐节点向量用竖条，逐步序列用横条，可学习 token 用星形，算子用圆，网络层用梯形或圆角矩形。这样形状不再单一，也不靠颜色单独区分。
3. **模块面板只用很浅的底色加彩色边框**（SAHC 珊瑚红、ALFTemporal 灰绿、ALFGraph 钢蓝），饱和色只留给数据。
4. **线型只各表示一种含义**：实线箭头为数据流；紫色虚线为 OS 条件（S 进入路由、FiLM、修正倍率）；灰色点线为仅训练时存在的路径（Teacher Forcing）。
5. **字号**：PPT 画布宽 33.87 cm 时，缩到 NeurIPS 正文宽 5.5 in 约为 41%。正文标签不小于 14 pt（缩放后约 5.7 pt），模块标题 16–18 pt，加粗只用于顶层模块名。全图一种字体（参考图 3/7 的 Times New Roman，或 1/4/5 的手写风格二选一，不要混用），公式用 PPT 公式编辑器输入。

## 1. 颜色与形状词表

| 数据 | 颜色（主 / 浅） | 形状 | 出现位置 |
|---|---|---|---|
| 交通值：X、X^c、X̄、y_k、Ỹ、Ŷ | 蓝 #2E75B6 / #BDD7EE | L×N 立方体板；曲线为蓝色 | 输入、SAHC、ALFTemporal 输入、解码、输出 |
| 生命周期上下文 S = [R, a, M]、路由权重 w_t、倍率 s(R_L) | 紫 #7F5FA8 / #E4DFEC | 逐步横条（R 为 4 格 one-hot，a 为阶梯柱，M 为黑白方格） | 输入、SAHC、ALFTemporal 输入、路由、修正 |
| 观测角色 | AAS ● 实心蓝圆；NAS ▲ 绿色三角；VS ◌ 灰色虚线空心圆；FS ✗ 红叉 | 形状加颜色，灰度打印也能区分（借鉴图 1(c) 的 Observed/Missing/Corrupted 图例） | 输入曲线、场景条、图例 |
| 日历 τ、δ、E^tod、E^dow | 金 #BF9000 / #FFF2CC | 时钟与日历图标（图 3 现成）、金色向量段 | ALFTemporal |
| 静态路网 A、结构特征 g、φ(g) | 橙 #C55A11 / #FBE5D6 | 节点-边小图、橙色向量段 | 输入、SAHC、分区、专家、修正 |
| 学习到的隐状态：h、ρ_t、e_n、T、Z、B_p | 青 #31859C / #DAEEF3 | 逐节点竖条（宽度标 64 或 160） | 各模块内部 |
| 4 个潜在 token Q_p | 青色实心星 ★ | 星形（图 4 的 Prototype 星） | 两阶段注意力 |
| FiLM 参数 Δγ、β | 玫红 #C0508A / #F2D0E3 | 两根并排竖条 | ALFTemporal → ALFGraph |
| 五个图专家 P_1…P_5 | 每个专家一种浅色小矩阵热图 | 小方阵图标（图 3 的 A_dyn 风格）+ 下标 | 图专家混合放大框 |
| 算子 ⊕ ⊙ ⊗、σ、tanh | 白底黑线 | 圆或小圆角矩形 | 各处 |

## 2. 版式（从左到右）

1. **输入列**：可见历史 X（三条传感器曲线，见第 3 节）；生命周期上下文 S（R_t 角色条、a_t 阶梯、M_t 方格，加四类场景的角色轨迹条）；静态路网 A。
2. **SAHC 列**（自上而下）：双向图循环补全（Forward 1→L、Backward L→1、MLP 融合，输出 X^c）→ 窗口内插值（开关 ι）→ 参考中心化（−αv）→ 共享历史 X̄，分两路送往右侧。
3. **ALFTemporal（右上）**：F_n = [X̄, S, τ, δ] → 时间卷积（宽口进、窄口出的梯形，标“kernel = stride = L”）得 e_n(64) → 拼接 z⁰_n(160) → 空间分区（第 4 节）→ 两阶段注意力（第一阶段竖向高亮一列 T_{:,c}；第二阶段一行 T'_{p,:} 经 4 颗星汇聚、回传）→ 恢复节点顺序 Z → FiLM 参数头（窄腰形 160→32→128）→ Δγ、β。
4. **ALFGraph（右下）**：图编码器（两层图 GRU 单元按 t = 1…L 展开，紫色虚线 S_t 从上方进入每个单元）→ FiLM（⊙ 与 ⊕）→ 残差自回归解码（k = 1…H，y_{k−1} 回送，紫色虚线 S_L，灰色点线 Teacher Forcing 只在训练）→ 参考恢复（+αv）→ 角色缩放图修正 → 输出 Ŷ。
5. **放大框（ALFGraph 下方虚线框）**：图专家混合：S_t → 路由器（Linear + Softmax，配 5 根彩色柱的权重图标 w_t）→ 五个专家 P_jUΘ_j → 加权求和 Σ → G；右侧为门控式 r、u、h̃、h'。
6. **底部图例**：⊕ ⊙ ⊗；实线、紫色虚线、灰色点线；四种角色形状；颜色词表的六种数据颜色。

## 3. 三处曲线（问题 1）

参考图 1(a) 的平滑实线加浅色面积、图 6(a) 的真值/预测双线、图 7 解码输出框的多色波浪线，统一成一种曲线风格：平滑样条，线宽 2.25 pt，曲线下方 15% 透明度浅色填充，观测点用小实心圆，缺失点用空心圆，缺失段用灰色点线。

- **可见历史 X**：三条泳道共用时间轴（刻度 1 与 L）。
  - 第一条（AAS，Steady）：完整蓝色曲线，中间一个空心圆表示自然缺失（不改变角色）；
  - 第二条（NAS，Join）：前 L−m 步灰色点线，最后 m 步出现蓝色曲线，起点画绿色三角；
  - 第三条（FS，Fault）：前 L−m 步蓝色曲线，失效处画红叉，之后灰色点线；
  - 可选第四条（VS，Unseen）：全程灰色点线加问号。
  这样输入面板本身就说明了 OS 与四类角色。
- **窗口内插值**：一条曲线，可见点实心、缺失点空心，缺失段用蓝色虚直线连接（线性插值），首末可见点之外画水平虚线（取最近可见值），下方小字“ι = 1, q_n > 0”。
- **输出 Ŷ**：小面板内左侧短段为历史（实线，浅蓝填充），竖直虚线标 t = L，右侧 H 步为预测（同色虚线，2–3 条节点曲线略错开），右下角标 Ŷ ∈ ℝ^{H×N}。

## 4. 空间分区（问题 2）

不再使用 PatchSTG 的 KD-tree 图。参考图 1(c) “Source Region / Target Region”的画法：

- 左半：一张浅灰路网（节点小圆 + 细边），叠加 3–4 块半透明圆润色块（分区），色块内的节点用该分区颜色描边；角标“P = 32”。
- 右半：P×C 方格（行 = 分区，行首色点与左侧色块同色；列 = 槽位 c = 1…C），最后一行末尾一格用斜线填充表示补足副本（只提供上下文，不产生输出）；方格下方标 T ∈ ℝ^{P×C×160}。
- 两阶段注意力沿用这张方格：第一阶段竖向高亮一列，第二阶段横向高亮一行并连到 4 颗星。

## 5. 图中公式与符号的修改（问题 4，对照方法章）

| 位置 | 现在 | 改为 |
|---|---|---|
| 可见历史 | X: 12×N×1；t−11 … t | X ∈ ℝ^{L×N}；时间轴 1 … L |
| 生命周期上下文 | R_t: role trajectory | R_t: observation role；a_t: state progress；M_t: mask |
| SAHC 补全 | t−11 → t、t → t−11 | 1 → L、L → 1 |
| SAHC 补全 | Align all steps + MLP fusion (fusion of bidirectional representations) | MLP fusion of ρ→_t, ρ←_t, m_t；输出标 X^c |
| 插值面板标题 | No observations: keep completed value | Window interpolation (ι) |
| 插值说明 | Observed window: replace missing slots / No observations: retain GRIN result | q_n > 0: linear fill of missing slots；q_n = 0: keep X^c |
| 中心化输出 | Shared filled history | Shared history X̄ |
| ALFTemporal 输入 | History embedding [X, S, calendar] | F_n = [X̄, S, τ, δ]；补画时间卷积 → e_n (64) |
| ALFTemporal 日历 | Today embedding / Week embedding | Time-of-day E^tod / Day-of-week E^dow |
| ALFTemporal 分区 | LKDT / Padded / Patched new indices | 按第 4 节重画 |
| 两阶段注意力 | Crosspartition / Withinpartition / 4 latenttokens（缺空格） | Stage 1: cross-partition (T_{:,c})；Stage 2: within-partition (T′_{p,:})；Q_p: 4 latent tokens |
| ALFGraph 编码器 | H: N×64；方框 G | h^{(1)}_L, h^{(2)}_L ∈ ℝ^{N×64}；G 改为路网 A 小图标或删去 |
| FiLM | H′ = (1+Δγ)⊙H+β | h^{(l)}_mod = (1+Δγ)⊙h^{(l)}_L + β |
| 解码器 | S^L、X̄^L | S_L、x̄_L（下标） |
| 修正模块 | Y_0；[ỹ_k, P+ỹ_k, P−ỹ_k]；×0.05 s(R^L)；R^L；+Y_0 | 输入 Ỹ；[ỹ_k, P⁺ỹ_k, P⁻ỹ_k]（上标，矩阵乘）；Linear–GELU–Linear；×0.05 s(R_L)；R_L；+ỹ_k；输出 Ŷ |
| 放大框标题 | Graph GRU cell with OSouted mixture of graph experts (used in encoder &decoder) | Graph GRU cell with OS-routed mixture of graph experts (encoder & decoder) |
| 专家名 | I / binary mean / weighted mean / Forward / Barkward | P₁ = I, P₂ binary, P₃ weighted, P₄ forward, P₅ backward |
| 混合式 | G(U,S) = Σwj ⊙ (PjUj) + b；Dense mixture · P(A) | G(U, S_t) = Σ_j w_{t,j} ⊙ (P_j U Θ_j) + b；w_t = softmax(S_t W_g)；删去 “Dense mixture · P(A)” |
| 门控 | c = tanh(G_c)；h′ = u⊙h + (1−u)⊙c；同时出现 h̃ | 只用 h̃：r = σ(G_r([x,h],S))，u = σ(G_u([x,h],S))，h̃ = tanh(G_c([x, r⊙h], S))，h′ = u⊙h + (1−u)⊙h̃ |
| 放大框说明 | Encoder S_t, decoder S^L -- separate parameters per layer/gate | Encoder: S_t；Decoder: S_L；12 modules (2 layers × 3 gates × enc/dec) |
| 星号 | Visible-window interpolation\*、Reference centering\* | 删去星号（各数据集结构相同，只是 ι、α 取值不同） |
| 名称 | GRIN、BiGRIL、DCRNN cell | 全部删去 |

## 6. 生图的用法

AI 生成的整图只作为**版式与风格目标**，文字和公式在 PPT 里重打；可以直接用的是单独生成的**小模块素材表**（曲线、分区色块图、星形 token、路由柱状图标、专家矩阵、FiLM 窄腰图标），白底或透明底，方便抠出来贴进 PPT。参考图里已有的模块（立方体、时钟/日历图标、⊕⊙⊗、梯形、圆角框）直接从同门的原 PPT 复制。Codex 的提示词见 `CODEX_PROMPT.md`。
