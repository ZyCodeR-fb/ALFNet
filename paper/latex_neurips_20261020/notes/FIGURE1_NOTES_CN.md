# 图 1（PPT 版 `imgs/alfnet.png`）与方法正文的对照，供作者后续修图

只列与第 3 节公式或源码不一致、或可能被误读的地方；文字定稿后由作者统一修改。公式编号按 2026-10-25 第二轮修改后的正文（图 1 已移到第 3 节开头，方法章为 3.1 问题设置、3.2 SAHC、3.3 ALFTemporal、3.4 ALFGraph、3.5 训练目标、3.6 数据集特定设置）。

## 一、命名与正文不一致（第二轮修改新增）

正文方法章已不再出现其他模型的名称，图中以下标注建议同步改掉：

| # | 图中位置 | 现状 | 建议 |
|---|---|---|---|
| A | SAHC 内部虚线框 | "BiGRIL"、"Forward GRIL"、"Backward GRIL" | 改为 "Bidirectional graph recurrent completion"、"Forward pass"、"Backward pass"（式 4–7） |
| B | Visible-window interpolation 说明 | "No observations: retain GRIN result" | 改为 "No observations: keep completed value"（式 8） |
| C | ALFGraph 下方虚线框标题 | "DCRNN cell – lifecycle-conditioned propagation" | 改为 "Graph GRU cell – lifecycle-conditioned propagation"（式 17–19） |
| D | 左列 Lifecycle context | 角色图例 AAS/NAS/VS/FS、a_t、M_t | 与正文一致；a_t 正文称“状态进度（state progress）”，图中若写文字名请用 state progress，不要用 state age |
| E | 各模块 | 图中英文模块名 | 与正文对应：Support-Aware History Completion（SAHC）、ALFTemporal、ALFGraph、Role-Scaled Graph Refinement（正文“角色缩放图修正”，属于 ALFGraph 的输出端） |

## 二、符号与公式不一致（上一轮已列，编号已更新）

| # | 图中位置 | 现状 | 正文 / 源码 |
|---|---|---|---|
| 1 | ALFTemporal / History embedding | 只画了 X、S、日历与 Today/Week embedding | 还应有结构嵌入 φ(g)（静态图属性 7→64→32，式 12）；拼接为 patch 64 ‖ ToD 32 ‖ DoW 32 ‖ Struct 32 = 160（式 11） |
| 2 | Graph-GRU gates | `c = tanh(G_c(r ⊙ h))` | `h̃ = tanh(G_c([x, r⊙h], S))`（式 19），缺少输入 x；正文候选状态记为 h̃ |
| 3 | Role-Scaled Graph Refinement | `[Y0, P + Y0, P − Y0]` | `[ỹ_k, P⁺ỹ_k, P⁻ỹ_k]`（式 25）：P⁺、P⁻ 是矩阵乘，不是加减；Y₀ 改为 Ỹ |
| 4 | Residual AR decoder / Restore | `Y0: N×1`，`yk = yk−1 + d·Δyk` | 解码器内部为 y_k（式 22），恢复参考值之后才是 Ỹ（式 24） |
| 5 | DCRNN cell 五个支持 | "Barkward" | Backward |
| 6 | Spatial partitions | "LKDT new indices / Padded new indices / Patched new indices" | 速度数据 KD-tree、流量数据 BFS；不足槽位用邻接排序的真实节点补足（3.3 节） |
| 7 | FiLM | `H' = (1+Δγ)⊙H + β` | 正文写作 h_mod，作用于两层编码末状态一次（式 21）；符号可保留，但建议与正文一致 |
| 8 | Static graph | 只在左下角出现 | SAHC、结构嵌入、编码器、修正都用到 A；可用命名端口标出 |
