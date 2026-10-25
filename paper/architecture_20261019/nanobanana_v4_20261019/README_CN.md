# ALFNet 架构图 v4（2026-10-20）

本目录为新增独立绘图目录。未修改模型、配置、权重、结果、论文正文或任何历史图；未 commit/push，未切换分支。

## 交付文件与选择

| 文件 | 性质 | 状态 |
|---|---|---|
| `alfnet_architecture_v4_gpt_P1_textfix.png`（1672×941） | ChatGPT 图像工具按代码骨架风格化（P1）＋脚本修正 3 行文字（`_source/textfix_p1.py`） | **推荐的风格版主图候选**；23 项语义核验全部通过；分辨率偏低、字号偏小，见核验记录 |
| `raw/P1_gpt_raw.png` | P1 原始输出，未改动 | 含 3 处文字错误 |
| `alfnet_architecture_v4_vector.svg` / `.png`（7200×4050） | 代码绘制的结构锁定版，全部为可编辑基础图元与文本，无位图 | **结构权威版**；全部核验通过；供 PPT 重建与高分辨率印刷 |
| `raw/G4_raw.jpg` | Gemini（Nano Banana 系列）在同一骨架上的最佳风格稿 | 设计稿，约 20 处文字错误，不可直接投稿 |
| `raw/G1–G5`、`raw/P2_gpt_raw.png` | 其余轮次原始输出 | 历史，未通过 |

选择理由：Nano Banana 在本任务密度下（2752×1536 输出）无法稳定保持小字与符号，逐条修字会引入新的语义错误（G5），按交接规则停止该链；ChatGPT 在只改风格时文字保真度最高（P1 仅 3 处错），但它的编辑同样是整图重绘（P2 修 3 处、坏 3 处）。因此采用"代码骨架锁定结构 → ChatGPT 单图风格化 → 对残留的 3 行文字做确定性补丁"的组合；Gemini 版本作风格对照。补丁只覆盖 3 行公式和 1 个标签，其他像素均为模型原始输出，脚本可复现。

**说明：** 交接要求主要视觉由 Nano Banana 完成；用户在本轮中途授权改用/结合 ChatGPT。最终推荐图的风格化来自 ChatGPT，不是 Nano Banana。

其他文档：[设计说明](DESIGN_CN.md)（依据分工、源码复核、R1 问题、参考图元映射、连接清单）、[生图过程](GENERATION_LOG_CN.md)（模型渠道、每轮输入/提示词/结果）、[核验记录](VERIFICATION_CN.md)、[PPT 图元指南](PPT_REBUILD_GUIDE_CN.md)。

## 英文图注

> **Architecture of ALFNet under changing observation support.** A single shared GRIN/BiGRIL module reconstructs the full 12-step history for all nodes by aligning and merging forward and backward recurrences; observed values are kept, missing positions are filled, and dataset-specific window interpolation (only for nodes with at least one visible reading, q > 0) and reference centering (X̄ = X_filled − αv, where v is the last visible reading or 0) produce one history X̄ shared by both branches. ALFTemporal compresses each node's window into one 64-d temporal patch by a stride-12 convolution, concatenates time-of-day, day-of-week and shared structural embeddings (160-d), arranges nodes into P = 32 spatial partitions of C slots, applies one two-stage attention block—cross-partition attention over the same slot, then within-partition gather/broadcast through four latent tokens—and restores the node order. A FiLM head maps the resulting Z ∈ ℝ^{N×160} to (Δγ, β), which modulate the terminal states of both graph-encoder layers once before decoding. ALFGraph uses lifecycle-conditioned DCRNN cells whose gates densely mix five propagation supports via a softmax router on the lifecycle state, and a two-layer residual autoregressive decoder initialized at X̄ᴸ that feeds back its unrefined step predictions. After adding back αv, role-scaled graph refinement adds a bounded neighbourhood residual scaled by the last role Rᴸ to produce the single 12-step forecast Ŷ. Unseen, Steady, Join and Fault denote observation scenarios realised by time-varying roles (AAS/NAS/VS/FS), state ages and observation masks. * Dataset-specific processing: C = 7/16/10/6, interpolation off/off/on/on, decoder residual scale d = 0.3/0.3/1/1, α = 0/0/0/0.5 for METR-LA/PEMS-BAY/PEMS04/PEMS08. Each layer and gate has its own router; WM and Fwd share a propagation pattern but have separate parameters; each attention stage includes residual, normalization and feed-forward layers. Batch dimensions are omitted; curves and matrix entries are schematic.

## 中英文模块名称

| 图中名称 | 中文 | 说明 |
|---|---|---|
| ALFNet — Adaptive Lifecycle Forecasting Network | 自适应生命周期预测网络 | 完整模型 |
| Support-Aware History Completion | 观测支持感知历史补全 | GRIN/BiGRIL 为既有方法，不称原创 |
| ALFTemporal — Spatial-Context Temporal Conditioning | 空间上下文时间条件分支 | 既有模块标识 |
| Temporal patch embedding | 时间 patch 嵌入 | 每节点 1 个 patch |
| Spatial partitioning / Partitioned Spatial Context Encoding | 空间分区 / 分区空间上下文编码 | 组织借鉴 PatchSTG |
| One two-stage attention block (Cross-partition, Within-partition) | 两阶段注意力（跨分区、分区内） | 沿用并适配 Crossformer 结构 |
| Restore node order | 恢复节点顺序 | 索引置换，非新增网络层 |
| FiLM head / FiLM | FiLM 参数头 / 逐特征线性调制 | 经典 FiLM |
| ALFGraph — Observation-Conditioned Graph Forecasting | 观测条件化图预测 | 既有模块标识 |
| Lifecycle-Conditioned Graph Propagation（DCRNN cell） | 生命周期条件化图传播 | 五专家 dense 路由；DCRNN/GRU 为既有单元 |
| Residual AR Decoder | 残差自回归解码器 | |
| Restore reference | 恢复参考值 | 加回 αv，非逆标准化 |
| Role-Scaled Graph Refinement | 角色缩放图修正 | 只用末角色固定倍率 |

图形分区不等于代码类边界：FiLM head 与 refiner 在代码中属于 `ALFNet`，画在蓝色区域只表示功能位置。

## 符号与维度

沿用 [v2 README](../v2_20261019/README_CN.md)（"符号与维度"一节） 的符号表；本图新增/变更：`h₀ = H_mod`（解码器初始隐藏状态）、`y₁:₁₂`（解码序列，Restore 之前）、`dense w₁…w₅`（五专家 softmax 权重）。上标 L 表示历史末步。

## 代码依据

本轮复核的源码行号见 [DESIGN_CN.md 第 2 节](DESIGN_CN.md)；完整模块映射见 [v2 README](../v2_20261019/README_CN.md)（"模块—代码依据"一节） 与 [第一阶段审计](../STAGE1_IMPLEMENTATION_AUDIT_CN.md)。

## 论文与代码冲突

本轮未发现新的论文—代码冲突。仅记录一处表述建议：Concat 顺序在代码中为 patch|ToD|DoW|Struct（`model/ALFTemporal/patchstg.py:291-324`），正文如描述拼接顺序应与此一致。

## 复现

```bash
python paper/architecture_20261019/nanobanana_v4_20261019/_source/build_skeleton.py
node paper/architecture_20261019/v2_20261019/_source/render.mjs paper/architecture_20261019/nanobanana_v4_20261019/alfnet_architecture_v4_vector.svg paper/architecture_20261019/nanobanana_v4_20261019/alfnet_architecture_v4_vector.png 3
python paper/architecture_20261019/nanobanana_v4_20261019/_source/textfix_p1.py
```

`build_skeleton.py` 写出 `_source/skeleton.svg`，`alfnet_architecture_v4_vector.svg` 为其副本。`skeleton_input.png`（G3/G4 输入，早期版本，缺 yₖ→Restore 连线）与 `skeleton_input_v2.png`（P1 输入，修正版）保留为生图输入证据。生图本身不可确定性复现；原始输出见 `raw/`。

浏览器下载过程中在用户下载目录（`D:/ZyCodeR/下载/`）留下了 Gemini/ChatGPT 原始下载文件，内容已复制进 `raw/`，未删除原件。
