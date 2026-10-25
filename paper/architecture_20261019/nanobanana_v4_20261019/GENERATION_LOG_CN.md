# 生图过程记录（v4，2026-10-19/20）

所有原始输出保存在 `raw/`（未经任何像素修改），提示词保存在 `prompts/`，检查截图/裁切在 `qa/`。

## 实际使用的模型与渠道

| 渠道 | 能确认的事实 | 不能确认的事实 |
|---|---|---|
| Google Gemini 网页版（用户已登录的 Chrome，Pro 账户），"制作图片"工具，聊天模型选择器为 "4 Argon"，宽高比 16:9 | 输出 JPEG 2752×1536；C2PA 清单写明 "Created by Google Generative AI"，`digitalSourceType=trainedAlgorithmicMedia`；Google 官方模型列表（ai.google.dev，2026-10-19 查询）最新图像模型为 **Nano Banana 2.1 / `gemini-nano-banana-2.1`**，另有 Nano Banana Pro `gemini-3-pro-image`、Nano Banana 2 `gemini-3.1-flash-image` | 网页版界面和 C2PA 都**未披露具体是哪一个 Nano Banana 版本**；网页版没有提供 4K 选项。因此本记录只称"Gemini 网页版图像工具（Nano Banana 系列）"，不宣称一定是 2.1 |
| ChatGPT 网页版（用户 Plus 账户），"创建图像"工具 | 输出 PNG 1672×941 | 界面未显示具体图像模型标识；不在此编造版本 |

无 Gemini API 密钥，未调用 API；所有生成通过用户浏览器界面完成。附带发现：R1（`imagegen_v3/alfnet_imagegen_r1.png`）作为输入时，其 C2PA 成分记录显示 `softwareAgent: ChatGPT / gpt-image`，即 R1 本身并非 Nano Banana 生成。

## 轮次

| 轮 | 渠道 | 输入图 | 提示词 | 结果 | 处理 |
|---|---|---|---|---|---|
| G1 | Gemini | R1、代码版 PNG、参考图 4、3 | `PROMPT_G1.txt`（完整内容规格） | 文字比 R1 清楚，但把提示词的 "REGION 1""4a)""FOOTER" 等结构标记画进图中；GRIN 两方向画成闭环串联；Struct MLP 从 patch 取输入；额外一条线落入 Refinement；y₀ 落在 FiLM→Decoder 主线；Y₀ 在 Restore 之前；五专家缺一 | 淘汰 |
| G2 | Gemini | 仅 R1 | `PROMPT_G2.txt`（11 条定点修改） | 保留 R1 外观且清晰，但多数修改未生效（FiLM 线仍落 Decoder、候选门仍缺 x），并新增 "micture"、R^S、R¹、S₁、Y_k 等错误 | 淘汰；证实"对 R1 文字式改线"不可靠 |
| — | 本地代码 | — | `_source/build_skeleton.py` | 改变制作方式：先用代码绘制结构锁定的骨架（全部连线、端口和机制图元精确），再交给生图模型只做风格化 | 骨架通过自动几何检查 |
| G3 | Gemini | 骨架、R1、参考图 4 | `PROMPT_G3.txt` | 多图输入使模型重新构图：4 latent tokens 消失、GRU 公式乱码、Restore 结构错误 | 淘汰 |
| G4 | Gemini | 仅骨架（早期版本 `skeleton_input.png`） | `PROMPT_G4.txt`（只改风格） | 布局与连线完整保留；约 20 处文字损坏（BIGRIL、X̂、S→5、C clots、X×160、100→32→128、1×Δγ、q≠0、Softmos、σ→c、y_θ、P″Y₀、×0.08·c(R)…），Refinement 内一处箭头反向 | 作为 Gemini 最佳风格稿保留 |
| G5 | Gemini（G4 同一对话） | — | `PROMPT_G5.txt`（逐条改字） | 多数错字修正，但引入新语义错误：输入 X 被改成 X̄、Decoder 的 h₀=H_mod 被替换为重复 y₀、C 值变 7/10/10/6；仍有 BIGRIL、marge、takens | 按交接规则停止 Gemini 修改链 |
| P1 | ChatGPT | 仅骨架（修正版 `skeleton_input_v2.png`） | `PROMPT_P1_gpt.txt`（只改风格） | 布局、连线、符号几乎全部保留，外观接近 R1；仅 3 处错误：Refinement 写成 [Y₀, P⁻Y₀, P⁺Y₀]（顺序颠倒）；u 门写成 G_r；门公式中 x 写成大写 X | 见下 |
| P2 | ChatGPT（P1 同一对话） | — | `PROMPT_P2_gpt.txt`（只改上述 3 处） | 见 `VERIFICATION_CN.md` | — |

注：G4 使用的早期骨架缺少 Decoder 输出 yₖ 到 Restore 的连接线，该缺陷在 G4/G5 中被继承；P1 使用的修正版骨架已补上。

## 结论（制作方式）

1. 直接让生图模型"读懂"架构并画图（G1）或按文字改线（G2、G5），在该信息密度下都会产生新的结构错误；这与此前 R2–R10 的退化现象一致。
2. "代码骨架锁定结构 + 单图风格化"能保住全部连线（G4、P1）；剩余问题集中在小字与符号。
3. 在该任务上，ChatGPT 图像工具的文字保真度明显高于 Gemini 网页版（P1 约 3 处错误 vs G4 约 20 处），但分辨率较低（1672×941 vs 2752×1536）。Gemini 版本作为同一骨架的风格对照保留。
