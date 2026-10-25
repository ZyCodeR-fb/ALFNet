# ALFNet 架构图 v4（Nano Banana）设计说明

日期 2026-10-19；分支 `paper/final`；HEAD `f06367e08e05e33f2364e7639dc614ab53c983f3`；开工时工作区仅有未跟踪的交接目录
`paper/architecture_20261019/claude_handoff_20261019/`。本目录为新增独立绘图目录，不修改历史图、模型、配置或论文。

## 1. 依据分工

| 输入 | 角色 | 不得从中继承 |
|---|---|---|
| `imagegen_v3/alfnet_imagegen_r1.png`（1639×960） | **视觉底稿**：整体分区、配色、立体薄张量、图标风格 | 下文列出的 12 处连线/机制错误 |
| `v2_20261019/alfnet_architecture.png`（6000×3960） | **内容与连接权威**：计算顺序、端点、维度 | 框图式外观 |
| `Reference_Image/4.png`、`3.png`、`5.png` | **图元/风格**：薄张量、token 组、时钟日历、图节点、圈运算、浅色容器 | 其各自的模型机制 |
| R2–R10 | 仅失败经验 | 不作底稿、不作输入 |

## 2. 本轮复核的代码事实（当前 HEAD）

只复读与图中关键关系有关的前向路径，其余沿用第一阶段审计和 v2 README 映射。

| 关系 | 证据 | 结论 |
|---|---|---|
| 唯一共享补全，两分支切同一张量 | `model/ALFNet/ALFNet.py:384-392` | 一次 `shared_input_imputer`，graph/patch 取同一 `filled` 前缀 |
| 插值仅 q>0 节点的缺失位置 | `experiments/flow_unified.py:36-37`，`flow_support_transfer.py:9-31` | `(~mask)&available` 时用 bridge 覆盖，否则保留 GRIN；随后 `raw−center` |
| 中心化 v、α | `flow_unified.py:10-22`、`flow_support_transfer.py:28-30` | v 为最后可见值，无观测为 0；`halfcenter` 即 α=0.5 |
| 恢复 αv 先于 Refiner | `flow_unified.py:46`（`pred+center`，替换 `_run_experts`）；`ALFNet.py:456` 在其后调用 `graph_refiner` | Restore → Refinement |
| FiLM 作用位置 | `ALFNet.py:403-407`；`model/ALFGraph/ALFGraph.py:315-330` | `film_head(Z)`→(1+Δγ, β)，广播到 encoder 两层 hidden，一次，在 decoder 之前 |
| FiLM head | `ALFNet.py:244-249` | `Linear(160→32)`, ReLU, `Linear(32→128)` |
| 解码初值与递推 | `ALFGraph.py:97-118, 331` | `initial_input = x_enc[:,-1]`=X̄ᴸ；`output = anchor + scale·projection(h_top)`；反馈为 output（训练时 teacher mask） |
| Decoder 条件 | `ALFGraph.py:342-344` | `state_sequence[:, -1]`=Sᴸ |
| Refiner | `ALFNet.py:26-91` | A+I 行归一化 P⁺ 与转置 P⁻；`[Y,P⁺Y,P⁻Y]`→Linear→GELU→Linear→tanh→×0.05·s(Rᴸ)→+Y |
| 两阶段注意力 | `model/ALFTemporal/patchstg.py:57-117,128` | 第一阶段沿分区轴，第二阶段 `router[seg, factor=4]` gather/broadcast；`WindowAttBlock` 默认 factor 4 |
| 时间特征 | `model/ALFTemporal/ALFTemporal.py:188-198` | ToD=12·hour+⌊minute/5⌋，DoW |

未发现与交接文件事实相矛盾之处。未运行模型、未读数据或权重。

## 3. R1 实图复核（逐项）

| # | 问题 | R1 实际所见 | 修复目标 |
|---|---|---|---|
| 1 | FiLM 端点 | Δγ,β 长线从右上折下，落在 "Residual AR decoder" 框顶 | 参数线落入金色 FiLM 框，分成 Δγ→⊙、β→⊕ 两个端点 |
| 2 | 输出旁路 | Restore ⊕ 右侧横线越过 Refinement 上沿直达 Forecast | 删除；Forecast 唯一入口为 Refinement 末端 ⊕ |
| 3 | Y₀ 命名 | Decoder 内小张量标 "Y₀ : N×1" | Decoder 内只用 yₖ / y₁:₁₂；Y₀ 只出现在 Restore 之后 |
| 4 | 静态图长线 | A、G 端口线伸到 Reference centering 附近 | 去掉长线；以命名端口 A/G 在 GRIN、Struct MLP、Encoder、Refinement 处引用 |
| 5 | q>0 位置 | "Missing (q>0)" 写在 Keep observations 行 | Keep observations 无 q 条件；q>0 仅附于 Interpolation* |
| 6 | 嵌入合流 | Conv、patch、ToD/DoW/Struct 小块散放，合流不明 | 单一拼接条 [patch 64 | ToD 32 | DoW 32 | Struct 32]=160，四路各一箭头 |
| 7 | Patch 表达 | 绿色长方体 + 多格，易读成多个 patch | 12 列时间切片 + 覆盖全窗口括号 → Conv(k=s=12) → 单个 token 列（64 维） |
| 8 | 分区交互 | 大网格中一道色带，未说明交互对象 | P×C 网格，同一槽位行高亮，行内节点间弧线连接 |
| 9 | GRU 候选 | "c = tanh(G_c(r⊙h))" 漏 x | `c = tanh(G_c([x, r⊙h], S))` |
| 10 | 层与步 | 立体块+虚线回环，难分层/步 | L1/L2 两层叠放并标 "2 layers"；回环为 "×12 steps"，取 yₖ 回到输入 |
| 11 | αv 与 Y₀ 残差 | Refinement 底部 "+Y₀" 和 αv 线条混杂 | αv 只进 Restore ⊕；Y₀ 旁路只进 Refinement 末 ⊕ |
| 12 | 分辨率/小字 | 1639×960，公式发虚 | 4K 原生重新生成；减少次要字，公式移图注 |

R1 还有：Lifecycle 区角色条与 S 端口紫色虚线直连 GRIN 区左缘（可保留为 S 端口）；"Cross-partition attention" 与 "Within-partition" 次序可读，但缺少方向箭头。

## 4. 参考图元 → ALFNet 机制 → PPT 重建

| 参考图具体元素 | 用于 | PPT 基础图元重建 |
|---|---|---|
| 图4 "Temporal Guide Construction"：带 B/N/T 轴的薄立方张量 + 时钟/日历图标 + 嵌入条 | History embedding：12 步切片、ToD/DoW | 平行四边形侧面+矩形正面组成薄 slab；12 个竖条格；时钟=圆+两线；日历=圆角矩形+格 |
| 图4 "Guide Vector" 分色拼接长条 | [patch|ToD|DoW|Struct]=160 拼接 | 4 段不同色矩形首尾相接，段下标维数 |
| 图4 左下 "ST-Guided Context Attention" 中小方格 K/V 组与 ⊕/⊗ | 4 latent tokens、gather/broadcast | 4 个等大方块；两组弧形箭头分别标 Gather、Broadcast |
| 图4 "Spatial Guide" 中节点图→嵌入矩阵 | Struct MLP（G→7→64→32） | 小节点图 + 梯形 MLP + 短条 |
| 图3 "DGFA" 小彩色矩阵 + Softmax 小框 | 五专家 support 矩阵 + router 权重条 | 4×4 方格矩阵 ×5；竖直柱状权重条 ×5 |
| 图3 圈加/圈乘与短正交连线 | FiLM 的 ⊙、⊕；Restore ⊕；Refinement ⊕；Σ 加权和 | 圆 + 符号文本 |
| 图3 输入节点图（彩色节点+边） | Static Graph 输入、P⁺/P⁻ 传播示意 | 圆 + 直线 |
| 图5 浅色圆角容器、白色运算框、正交黑箭头 | 全图分区与框体规范 | 圆角矩形浅填充、细描边；肘形连接线 |
| 图5 "GeoModulated Cross-Attention" 中 Vg/Vt 分支汇入 ⊗/⊕ | FiLM：Δγ→⊙，β→⊕，H 主路贯穿 | 两个圆运算符串联在 H 主路上，参数从上方各进一个 |
| 图7 左栏 "Space/Time" 轴立方 + Mask 立方 | Lifecycle Context 的 M 网格、X 窗口时间轴 | 小方格矩阵，紫色深浅 |
| 图6 (d) 中 "Input ST tokens" 网格 + 高亮单格 | 分区网格中同槽位高亮 | P 列×C 行方格，一行描深色边 |
| 图2 节点身份保持（同色节点跨面板） | Restore node order：同色条重排 | 彩色短条 + 交叉连线（索引置换） |
| 图1 观测曲线 + 缺失 ×/虚圆 | 可见历史、Keep observations（实心=观测，空心=填补） | 折线 + 实/空心圆 |

## 5. 布局（保留 R1 骨架）

```
| Inputs | Support-Aware History Completion | ALFTemporal (top, green): embed → partition → cross → within → restore → Z → FiLM head |
|        |  (pale red, vertical)             |-----------------------------Δγ,β ↓--------------------------------------------------|
|        |                                   | ALFGraph (bottom, blue): Encoder → FiLM → Decoder → Restore ⊕ → Refinement → Forecast |
|        |                                   |   [one cell-definition inset under Encoder/Decoder, dashed leader lines]             |
| Footer: Training only ... | legend                                                                                        |
```

## 6. 关键连接清单（生成后逐条核对）

| ID | 起点 → 终点 | 线型 |
|---|---|---|
| C1 | X (Visible history) → GRIN/BiGRIL | 实线 |
| C2 | S 端口 → GRIN；Mₜ → Keep observations | 紫虚线 |
| C3 | A 端口 → GRIN | 灰端口 |
| C4 | GRIN 前向链 ∥ 后向链 → Align+merge（不串联） | 实线 |
| C5 | Align+merge → Keep observations → Interpolation*（q>0）→ Reference centering* → X̄ | 实线 |
| C6 | X̄ → History embedding；X̄ → Graph Encoder（同一 X̄ 分叉） | 实线 |
| C7 | S → History embedding 输入；S_t → Encoder；Sᴸ → Decoder；Rᴸ → ×0.05 s(Rᴸ) | 紫虚线 |
| C8 | 12 步切片 → Conv(12) → 1 patch(64) → Concat | 实线 |
| C9 | ToD 32、DoW 32、Struct MLP 32（G 端口）→ Concat=160 | 实线 |
| C10 | Concat → Partition (P×C) → Cross-partition → Within-partition (C→4→C) → Restore order → Z | 实线 |
| C11 | Z → FiLM head → Δγ, β → 金色 FiLM（Δγ 入 ⊙，β 入 ⊕） | 实线（参数） |
| C12 | Encoder（2 层末状态 H）→ FiLM → H_mod → Decoder | 实线 |
| C13 | X̄ᴸ → Decoder 初值 y₀；yₖ → 反馈回 Decoder 输入（×12 steps） | 实线 |
| C14 | Decoder y₁:₁₂ → Restore ⊕ ← αv | 实线 |
| C15 | Restore → Y₀ → [Y₀,P⁺Y₀,P⁻Y₀] → MLP → tanh → ×0.05 s(Rᴸ) → ⊕ | 实线 |
| C16 | Y₀ 旁路 → 末 ⊕（唯一） | 实线 |
| C17 | 末 ⊕ → Ŷ → Forecast（唯一出口） | 实线 |
| C18 | Cell inset ↔ Encoder/Decoder | 无箭头虚引线 |
| 禁止 | Δγ,β→Decoder；Restore→Forecast 直连；hidden→反馈；A/G→centering；refined→decoder | — |
