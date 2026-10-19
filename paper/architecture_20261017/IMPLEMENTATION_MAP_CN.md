# ALFNet 架构图：最终实现核对与绘图交接

核对日期：2026-10-17。分支 `paper/final`；HEAD `e02b036d665e0c82f33afd4195b7a6ea8cad1c18`。开始时工作区及暂存区干净，相对本地 `origin/paper/final` 为 0/0，未 fetch。

当前阶段只核对代码、配置与已有论文文字，等待用户模板后布局绘图。没有训练、模型 forward、GPU 工作、原始数据/test 读取、权重加载、实验结果修改或提交推送。用户指定阅读的论文和配置清单中已有测试结果，不等于本轮重新打开 test。

## 1. 权威与冻结配置

按 WORKFLOW → PROJECT_CONTEXT → HANDOFF → 当前任务交接恢复上下文，并读取 FINAL_PROJECT_INDEX、RESEARCH_LESSONS、v5 正文和实验稿。历史研究记录不作为恢复废弃模块或重启实验的依据。

最终权威：[selected_configuration_manifest.json](../pems08_wd_test_20261012/selected_configuration_manifest.json)。四份外层 YAML 的 SHA256 均与清单一致；已逐一解析其引用的八份专家 YAML。核对使用 `/home/xd/anaconda3/envs/ReaCH/bin/python`，只导入标准库与 YAML；未解决 environment.yml 与安装环境的历史差异。

| 数据集 | 分区容量 C | 窗口插值 | 解码倍率 d | 中心化强度 c | 生效适配 |
|---|---:|---|---:|---:|---|
| METR-LA | 7 | 关闭 | 0.3 | 0 | 原生 ALFNet |
| PEMS-BAY | 16 | 关闭 | 0.3 | 0 | 原生 ALFNet |
| PEMS04 | 10 | 开启 | 1 | 0 | flow_support_transfer：bridge_delta1 |
| PEMS08 | 6 | 开启 | 1 | 0.5 | flow_unified：triple_halfcenter |

论文记录四数据集均为32个空间分区；代码以 runtime_context.layout_parts 的长度取得实际分区数。C 是每个分区节点槽位容量，不是分区数量。本轮未打开数据资产重新数分区。容量不足时拒绝构建；不足 C 的分区使用邻接排序的真实节点补足，不是补零节点。

四份图专家配置均为 `moge_router_mode=soft`、五个支持算子、负载平衡系数0.01、隐藏宽度64、两层循环编码/解码。配置中的 top_k=1、noisy_gating=true 不改变 soft 分支的实际执行。四份时间专家均为 `emit=features`、lifecycle_input=true、tem_patchsize=12、tem_patchnum=1、layers=1、factors=1、router factor=4，条件特征维度160。`recur_times=5` 未在该 backbone 中形成五次迭代，不应据此画五层。

## 2. 模块—输入输出—代码依据

代码行号以本次 HEAD 为准；路径相对仓库根目录。

| 模块 | 实际输入 → 输出 / 条件注入 | 代码依据 |
|---|---|---|
| 输入及观测支持构造 | 标准化历史、自然可用性、节点暴露范围、生命周期轨迹 → 可见历史 Xvis 与六维 S=[角色one-hot(4), 状态年龄a, 掩码M]；缺失信号置标准化空间的0 | data_provider/v4_lifecycle_loader.py:204、239、319；data_provider/lifecycle_roles.py:12 |
| ALFNet 输入装配 | Xvis 与 S 沿通道拼接；历史12步、完整N节点轴、单一速度/流量通道；日历和邻接单独传入 | model/ALFNet/ALFNet.py:321 |
| 共享观测感知历史重建 | A、Xvis、M、角色及年龄 → GRIN/BiGRIL 双向图循环重建；观测值保留，只有不可见位置被学习值替代；原始S原样保留 | layers/source_grin.py:338、416；model/ALFNet/ALFNet.py:368 |
| 数据集相关窗口处理 | 在共享重建之后、两分支之前：可选可见窗口插值；可选减去 c×最后可见值；无观测节点保留GRIN结果，中心为0 | experiments/flow_support_transfer.py:9、31、40；experiments/flow_unified.py:10、29 |
| ALFGraph 编码器 | 处理后历史+S、静态A → 两层图循环隐藏状态 H；每一历史步的S控制该步图卷积路由 | model/ALFGraph/ALFGraph.py:16、282；layers/DCRNNCell.py:60 |
| 条件化图传播 | S → dense softmax五路权重；五种支持矩阵分别传播并线性变换，再按节点加权求和；作用于GRU各门，编码与解码均使用 | layers/lifecycle_moge.py:44、61、269、334、445 |
| ALFTemporal 历史嵌入 | 同一处理后历史+S、历史日历、共享结构节点编码 → 历史窗口压缩特征；结构编码来源为拓扑与可用坐标，无逐节点可学习ID表 | model/ALFTemporal/ALFTemporal.py:188、201；model/ALFTemporal/patchstg.py:289；layers/structural_node_encoder.py:403 |
| ALFTemporal 空间上下文 | 重排并补足的分区token → 两阶段注意力 → 还原原节点顺序 → 每节点条件向量 Z∈R^160；没有独立预测头 | model/ALFTemporal/ALFTemporal.py:19、105；model/ALFTemporal/patchstg.py:89、160、265 |
| Temporal–FiLM | Z → MLP → Δγ、β；H′=(1+Δγ)⊙H+β，广播到所有循环层；注入点在编码结束与解码开始之间 | model/ALFNet/ALFNet.py:394；model/ALFGraph/ALFGraph.py:327 |
| 单一自回归残差解码 | H′、处理后最后历史值、末历史步S → 12步预测；每步 yk=anchor+d×Linear(hk)，推理以先前预测递推；S固定为历史末步 | model/ALFGraph/ALFGraph.py:64、338 |
| 中心还原 | 解码输出加回 c×最后可见值，然后进入图细化；这不是在最终图细化之后才还原 | experiments/flow_support_transfer.py:54；experiments/flow_unified.py:46 |
| Graph Refinement | 预测、A、末步角色 → 加自环的正反向一步扩散特征 → MLP+tanh残差；残差乘0.05与角色倍率[0,1,1,0.25] → 最终预测 | model/ALFNet/ALFNet.py:26、456；四份外层配置 family.recipe |
| 训练目标 | 最终预测和合法目标 → 0.9普通均值+0.1角色均衡预测损失，再加图路由平衡项；没有独立重建监督项 | experiments/focal_loss.py:9；experiments/v4_losses.py:25；experiments/v4_runner.py:380；layers/lifecycle_moge.py:334 |

## 3. 画图必须保持的连线关系

主干顺序：可见历史/支持描述 → 共享重建 → 可选窗口处理 → ALFGraph编码器 → FiLM调制 → 单一残差解码器 → 可选中心还原 → Graph Refinement → 未来预测。

时间条件支路从“共享重建/窗口处理后的同一历史”分出，经 ALFTemporal 输出 Z，再到 FiLM。不要画第二套预测、预测级融合权重、ALFGraph到ALFTemporal的反馈，或FiLM逐预测步重新调制。

观测条件连线分别到：GRIN的显式M与角色/年龄辅助输入；ALFGraph每个历史步的路由及末步固定解码路由；ALFTemporal输入通道；Graph Refinement末步角色倍率。S是显式六维上下文，不是独立学习出的统一状态向量。ALFNet虽然计算 `state_encoder` 输出，但该输出被丢弃，只有末步角色被使用；不能将该MLP画成有效预测模块。

静态图A同时支持重建、图推理、结构编码/分区上下文和输出细化。分区布局构造与训练统计标准化属于静态/数据准备；GRIN和窗口可见插值都属于预测时输入处理。插值只访问当前历史窗口，可在历史缺口两侧线性插值，边界使用最近可见端点；并非使用预测未来值的离线全序列插补。

建议主图以实线表达预测计算、细虚线表达条件注入，训练目标放在外围小框并使用单独图例。训练时的合法 masked teacher forcing 仅作为训练注记，不能连成推理需要真实未来的路径。

## 4. 易误画点与文字对应

1. **时间分支注意力轴。** Conv2d将12步压成1个时间token；WindowAttBlock将输入重排成 `[B×Tpatch, C, P, D]`。底层名为 time_attention 的运算实际沿P个空间分区计算，后续latent sender/receiver在分区内C个槽位间交互。可概括为“历史嵌入＋跨分区/分区内上下文交互”，不要画成12步时间自注意力或套用原始Crossformer轴标签。
2. **五个算子槽位不保证五个数值不同的矩阵。** identity、mean、weighted_mean、forward_diffusion、backward_diffusion为真实名称；weighted_mean与forward_diffusion均是邻接行归一化，不能宣称五种必然独立的传播几何。各槽位有独立Theta。
3. **重建是内部表示。** GRIN返回值参与预测梯度；当前GRIN包装器没有暴露重建loss，外层仅传递graph_router辅助项。最终输出不是“重建+预测双任务输出”。
4. **完整模型保留Graph Refinement。** 正文不展示其消融行，不代表从前向路径删去。AAS倍率为0，其他角色按上述倍率修正。
5. **方法正文缺失。** v5目前是第1、2、4节；实验稿只有第4节。因此只能确认引言中的“共享重建、条件图推理、时间特征FiLM、单一预测”与实现一致，不能声称已和第3节公式逐项对齐。引言的FiLM说明较宽泛，正式方法应明确注入编码器末状态。
6. **训练文字一致。** v5第4.1的单元素损失可写成 ell(e)=(e/σ) tanh(0.1e)，e是原单位绝对误差；与代码sigmoid写法等价。模型辅助损失应具体写“图路由平衡”，不要补成重建损失。soft模式load的CV项为0，importance项仍有效；系数0.01在各路由内部生效，经步/模块均值汇总。

本轮未修改论文与模型；上述注意点作为绘图与后续方法写作约束，不推导新的优化方案。

## 5. 候选图注与符号（待模板定版）

**图1 ALFNet整体架构。** 模型由可见历史及其观测角色、状态持续时间与观测掩码构造生命周期上下文。共享图循环重建提供历史表示，ALFGraph依据观测条件混合图传播算子，ALFTemporal从共享历史中提取空间上下文增强的条件特征，并通过FiLM调制图编码器的末状态。单一自回归残差解码器生成未来序列，经中心还原（若启用）和角色相关的图细化得到最终预测。窗口插值、中心化强度及残差倍率按数据集配置；预测损失及路由平衡项仅用于训练。

| 符号 | 定义 |
|---|---|
| Xvis、M | 标准化可见历史及逐时刻观测掩码，历史长度L=12 |
| R、a、S | 四类角色one-hot、状态持续时间编码、六维上下文S=[R,a,M] |
| A | 已知完整节点轴上的静态邻接 |
| P、C | 空间分区数量、每分区节点token容量，二者不可混用 |
| Z | ALFTemporal每节点160维条件特征 |
| H、H′ | 图循环编码末状态及FiLM调制状态，两层、每层64维 |
| Δγ、β | FiLM增益偏移与平移；有效乘性因子为1+Δγ |
| d、c | 解码残差倍率、窗口中心化强度；避免与损失的0.1系数混用alpha |
| Yhat | 最终12步预测；图内保持标准化空间，原单位恢复作为输出表示说明 |
| Lpred、Lroute | 角色混合的误差幅度加权预测目标、图路由平衡辅助项 |

## 6. 下一步与验证边界

- Facts / Completed：上下文、当前Git、四配置身份、八专家配置、关键前向与训练损失静态核对完成；本文件保存映射、图注与符号草案。
- Hypotheses：无新增科学假设或效果主张。
- Modified but unverified：图注尚未与缺失的第3节方法正文逐项对齐；未执行数值推理或重新核验权重二进制。
- Not started：等待用户模板；随后在本目录另存可编辑矢量源与SVG版本，保留已有图。最终需视觉检查连线、符号、文字可编辑性及PPT导入表现。
- Ruled out：重新优化模型、训练、打开test、改实验结果、恢复废弃模块、提交或推送。

验证限于源码阅读、配置解析/SHA256比对和文档检查；不将历史数值测试报告当成本轮运行结果。
