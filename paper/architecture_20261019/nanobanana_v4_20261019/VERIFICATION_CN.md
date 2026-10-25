# v4 架构图核验记录

2026-10-20；分支 `paper/final`；HEAD `f06367e08e05e33f2364e7639dc614ab53c983f3`（本轮未提交，工作区仅新增本目录及交接目录）。本轮为静态绘图与源码对照，没有运行模型、训练、评估或读取数据/权重。

被核验对象：

- **V** = `alfnet_architecture_v4_vector.svg/.png`（代码绘制，结构权威版）
- **P1fix** = `alfnet_architecture_v4_gpt_P1_textfix.png`（ChatGPT 风格化 P1 + 3 行脚本文字补丁）
- 参考：G4/G5（Gemini）、P2（ChatGPT）均未通过，问题列于末尾。

判定：通过 / 待修复 / 未验证。依据见 `DESIGN_CN.md` 第 2 节源码行号。

## 一、语义逐项

| # | 检查项 | V | P1fix | 说明 |
|---|---|---|---|---|
| 1 | 输入 X、Lifecycle、Static Graph 各只画一次；无独立 Visibility 分支 | 通过 | 通过 | S/M/A/G 仅以端口标签引用 |
| 2 | 场景 Unseen/Steady/Join/Fault 与内部角色 AAS/NAS/VS/FS 区分；Join=VS→NAS，Fault=AAS→FS，窗口内变化 | 通过 | 通过 | 角色色块图例单列 |
| 3 | 唯一共享 GRIN/BiGRIL；两方向并行、各自汇入 Align+merge，不串联 | 通过 | 通过 | |
| 4 | 顺序：GRIN → Keep observations → Interpolation*（q>0 覆盖缺失；q=0 保留 GRIN）→ Reference centering → X̄；q 条件只在插值处 | 通过 | 通过 | Keep observations 不含 q |
| 5 | X̄ = X_filled − αv；X̄ 一条线分叉到时间分支与编码器；A/G 无长线进中心化 | 通过 | 通过 | |
| 6 | 单个 temporal patch：12 步切片 + 全窗括号 Conv k=s=12 → 一列 64 维 | 通过 | 通过 | |
| 7 | Concat 160 = patch 64 + ToD 32 + DoW 32 + Struct 32，顺序与 `patchstg.py:291-324` 一致；Struct MLP 7→64→32 仅取 G | 通过 | 通过 | |
| 8 | P=32 分区、C 槽轴标注；C 不当作分区数 | 通过 | 通过 | C=7/16/10/6 移至图注 |
| 9 | 先 Cross-partition（同槽位跨 P，高亮行+弧）再 Within-partition（C→恰好 4 latent tokens→C，Gather/Broadcast）；单块 1 head | 通过 | 通过 | |
| 10 | Restore node order → Z (N×160) → FiLM head 160→32→128 → Δγ, β (各 N×64)；时间支路无独立预测 | 通过 | 通过 | |
| 11 | Δγ,β 参数线终点在金色 FiLM 框，分到 ⊙(1+Δγ) 与 +β；不进 Decoder | 通过 | 通过 | |
| 12 | FiLM 作用于编码器两层末状态 H，一次；H_mod 作为解码器 h₀ | 通过 | 通过 | |
| 13 | Encoder/Decoder 各 2 层（L1/L2），hidden 64；编码用 Sₜ，解码用 Sᴸ | 通过 | 通过 | |
| 14 | 解码初值 y₀ = X̄ᴸ；回环取未修正的 yₖ（不是 hidden、不是 refiner 输出）；×12 | 通过 | 通过 | |
| 15 | Decoder 输出 y₁:₁₂ → Restore ⊕ ← αv → Y₀；Y₀ 不在 Decoder 内出现 | 通过 | 通过 | |
| 16 | Refinement：[Y₀, P⁺Y₀, P⁻Y₀] → MLP 3→32→1 (GELU) → tanh → ×0.05·s(Rᴸ) → ⊕；Y₀ 旁路只进该 ⊕；只用 Rᴸ | 通过 | 通过（补丁后） | 原 P1 顺序颠倒，已补丁 |
| 17 | 唯一 Forecast，唯一输入为 Ŷ；无 Restore→Forecast 旁路 | 通过 | 通过 | |
| 18 | 五专家 I/Mean/WM/Fwd/Bwd，dense softmax 权重 w₁…w₅，Σ 汇总；WM 与 Fwd 图案相同 | 通过 | 通过 | 未画 top-k |
| 19 | G(U,S)=Σⱼ wⱼ⊙(PⱼUΘⱼ)+b；各层各门独立 router | 通过 | 通过 | P1fix 下标写作 i（同义记号） |
| 20 | GRU：r=σ(G_r([x,h],S))，u=σ(G_u([x,h],S))，c=tanh(G_c([x,r⊙h],S))，h′=u⊙h+(1−u)⊙c | 通过 | 通过（补丁后） | 原 P1 u 门写 G_r、x 写成 X，已补丁 |
| 21 | Cell inset 仅以无箭头虚线引线连接 Encoder/Decoder | 通过 | 通过 | |
| 22 | 训练区独立：0.9L_mean+0.1L_role+L_route，masked teacher forcing；无真值进入推理 | 通过 | 通过 | |
| 23 | 无凭空新增机制（无多 patch、无多 head、无 S→FiLM head、无额外损失/预测头） | 通过 | 通过 | |

## 二、视觉与可读性

| 检查项 | V | P1fix |
|---|---|---|
| 线型：数据黑实线、生命周期紫虚线、静态图灰端口、cell 蓝虚线 | 通过 | 通过 |
| 自动检查：文字互相重叠 / 越界 / 宽度溢出 | 0 / 0 / 0（`qa/alfnet_architecture_v4_vector_audit.json`） | 不适用（位图），人工逐块检查无遮挡 |
| 连线穿字（自动） | 仅 "Δγ, β" 一处，为有意在白底标签上压线 | 人工检查同 |
| 拼写与符号 | 通过（188 个文本对象） | 补丁后人工逐块通过；补丁区与周边字体存在轻微风格差（Cambria vs 生成字形） |
| 分辨率 | 矢量；PNG 7200×4050 | 1672×941（ChatGPT 网页版原生输出）；按 180 mm 宽约 236 dpi，低于常见 300 dpi 印刷建议 |
| 180 mm 通栏字号 | 设计单位 2400 宽，最小 19 单位 ≈ 1.43 mm ≈ 4.0 pt；主标签 22–30 单位 ≈ 4.7–6.4 pt | 与 V 相同比例 |
| 结论 | **字号偏小**：次要注释约 4 pt，低于 v2 代码图的 7.1 pt。已把 v 定义、C 值、FFN、node-ID 说明等移入图注。若期刊要求 ≥6 pt，需再删减图内公式（如 GRU 四式、G(U,S)）至图注，或改为整页宽横排 | 同左 |

`qa/vector_180mm_150dpi.png` 为 180 mm 宽 150 dpi 的缩小预览。

## 三、未通过的版本与剩余问题

| 版本 | 结论 | 主要问题 |
|---|---|---|
| R1（底稿） | 未通过 | 交接列出的 12 项全部经实图确认（`DESIGN_CN.md` 第 3 节） |
| G1 | 未通过 | 结构性错误多处，提示词标记被绘入图 |
| G2 | 未通过 | 定点修改大多未生效，新增错字 |
| G3 | 未通过 | 多图输入导致重构图，丢失 4 tokens、GRU 乱码 |
| G4 | 风格稿，未通过 | 布局/连线保留；约 20 处文字/符号损坏，Refinement 内箭头反向，且继承早期骨架缺少 yₖ→Restore 连线 |
| G5 | 未通过 | 修字同时引入语义错误（输入 X→X̄、h₀=H_mod 丢失、C 值错） |
| P1（原始） | 3 处文字错误 | 已在 P1fix 中以脚本补丁修复 |
| P2 | 未通过 | 修了 3 处，但 Refinement 中 [..]→MLP 箭头消失，s(Rᴸ)→s(R⁻¹)，Rᴸ 标签变 "RL"；且全图重绘（16% 像素变化） |

## 四、未验证项

- PowerPoint 中"转换为形状/取消组合"对 V 的实际拆分效果未在本机实测。
- P1fix 的补丁字体在印刷尺寸下与生成字形的观感差异只做了屏幕检查。
- 生图模型的具体版本号（两家网页界面均未披露）无法核实。
