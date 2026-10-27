# 图 1（PPT 版 `imgs/alfnet.png`）与方法正文的对照，供作者重绘时参考

公式编号按 2026-10-27 第四轮修改后的正文：3.1 问题设置、3.2 SAHC（式 4 为通用图门控循环单元）、3.3 ALFTemporal、3.4 ALFGraph、3.5 训练目标。

## 一、命名与正文不一致

| # | 图中位置 | 现状 | 建议 |
|---|---|---|---|
| A | SAHC 内部虚线框 | "BiGRIL"、"Forward GRIL"、"Backward GRIL" | "Bidirectional graph recurrent completion"、"Forward pass"、"Backward pass"（式 5–8） |
| B | Visible-window interpolation 说明 | "No observations: retain GRIN result" | "No observations: keep completed value"（式 9） |
| C | ALFGraph 下方虚线框 | "DCRNN cell – lifecycle-conditioned propagation"，五个方块标为 I/Mean/WM/Forward/Barkward | 标题改为 "Graph GRU cell with OS-routed mixture of graph experts"；五个方块是五个图专家（Expert 1–5：I、binary mean、weighted mean、forward、backward）；"Linear → Softmax" 即路由器（式 18–19） |
| D | 左列 Lifecycle context | a_t | 正文称“状态进度（state progress）”，图中若写文字名用 state progress |
| E | 模块名 | — | Support-Aware History Completion（SAHC）、ALFTemporal、ALFGraph；Role-Scaled Graph Refinement 属于 ALFGraph 的输出端 |
| F | Visible-window interpolation\*、Reference centering\* | 星号表示“只在部分数据集启用” | 正文图注已删去该说明（所有数据集结构相同，只是 ι、α 取值不同），建议去掉星号，或改为在图中标出开关 ι 与强度 α |

## 二、符号与公式不一致

| # | 图中位置 | 现状 | 正文 |
|---|---|---|---|
| 1 | ALFTemporal / History embedding | 只画了 X、S、日历 | 还有结构嵌入 φ(g)（式 13）；拼接为 64 ‖ 32 ‖ 32 ‖ 32 = 160（式 12） |
| 2 | Graph-GRU gates | `c = tanh(G_c(r ⊙ h))` | `h̃ = tanh(G_c([x, r⊙h], S))`（式 4 的门控形式，Φ 取图专家混合 G） |
| 3 | Role-Scaled Graph Refinement | `[Y0, P + Y0, P − Y0]` | `[ỹ_k, P⁺ỹ_k, P⁻ỹ_k]`（式 27）：P⁺、P⁻ 是矩阵乘 |
| 4 | Residual AR decoder / Restore | `Y0: N×1`，`yk = yk−1 + d·Δyk` | 解码器内部为 y_k（式 24），恢复参考值之后为 Ỹ（式 26） |
| 5 | Two-stage attention | Cross-partition → Within-partition | 第一阶段对每个槽位跨分区（式 15），第二阶段在分区内经 4 个潜在 token 汇聚与回传（式 16），与图一致 |
| 6 | Spatial partitions | "LKDT new indices / Padded / Patched new indices" | 坐标可靠时 KD-tree，否则 BFS；不足槽位用邻接排序的真实节点补足（3.3 节） |
| 7 | FiLM | `H' = (1+Δγ)⊙H + β` | 正文写作 h_mod（式 23） |
| 8 | Static graph | 只在左下角出现 | SAHC、结构嵌入、编码器、修正都用到 A |
