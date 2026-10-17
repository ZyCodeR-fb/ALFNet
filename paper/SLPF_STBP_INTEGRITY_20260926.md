# SLPF / STBP 完整性复核

日期：2026-09-26。核验HEAD `fd84280cf20034601fcbd98d7d8ed821fa3049d2`，分支paper/final。范围：本地参考源码、当前模型、历史失败记录、CPU合成数据核验；不读取真实训练/验证/test样本，不训练，不改变最终模型、现有基线权重和结果。

## 结论

适配完整性影响实验结论的对象，而不只是命名。可以比较协议适配版本，但不能把其结果解释为完整原方法的性能。两个当前实现均有实质性变化，不存在仅恢复一个开关即可完整复现的路径。按“可以轻松实现才恢复”的要求，本轮不重新引入复杂训练流程；修正审计文件的过时状态，保留具体修复边界供后续选型。

| 方法 | 保留内容 | 缺少或替换内容 | 完整恢复难点 |
|---|---|---|---|
| SLPF | rank/calendar表示、源MLP及空间转移公式；源文件仍含三个stage分支 | wrapper仅运行forecast；所有节点列为sensed，unsensed为空；位置嵌入改共享结构编码 | 三阶段监督、伪未感知分组、动态掩码/部分历史、空观测边界、阶段选点与恢复 |
| STBP | 频域层、DLGA、编码器、预测头 | 逐节点pattern_bank改共享结构编码；无扩展、无持续训练调度 | 原方法需要增量阶段监督并更新模式库，与当前固定权重归纳评价不同 |

## SLPF：原方法假设与历史失败

参考 `references/SLPF/model/BasicTrainer.py`、`BasicInferencer.py`、`lib/dataloader.py`：

1. ADP用已感知历史重建未感知位置历史，训练损失读取`unsensed_X`。
2. forecast使用重建历史预测已感知位置未来，监督来自`sensed_Y`。
3. aggregation结合全位置历史表示与已感知未来预测，监督未感知位置未来`target`。
4. 三阶段各有模型、优化器和选择过程。源代码向后续阶段复制前阶段权重。

“未感知”在源训练中并不意味着始终没有这些位置的历史或未来标签；而本项目真实V/T节点不能提供训练动态信号。因此直接运行源loader不符合协议。可以在K内部制造伪未感知节点学习三阶段，但这是需要明确验证的归纳适配，并非源设置原样复现。

当前 `model/comparison_model/SLPF/__init__.py` 设置`num_unsensed_locs=0`、`sensed_locations=range(N)`、`stage='forecast'`，运行的是单个预测主干。核心文件内有ADP/agg分支，不代表训练器实际调用它们。

只读核查 `archive/research_history_20260917.tar.gz` 内日志与代码：

- `outputs_v4/baseline_retrain_seed32_20260916/5ebe616f.../METRLA_SLPF.log` 明确在ADP的partition步骤报`ValueError: SLPF requires at least one sensed history location`。
- 同目录PEMSBAY日志是`KeyboardInterrupt`，不能声称两个数据集都因同一异常失败。
- 归档`staged_adapter.py`有159行，`slpf_pretraining.py`有122行，另有已退役的runner接线；并非现有单阶段接口只需开关即可恢复。
- 旧partition用`mask.any(dim=0)`划分节点，任一时刻有观测就算sensed。这样Join/Warm和Fault/Decay的部分历史节点仍走sensed路径，ADP主要处理整窗无观测节点，不能宣称恢复了所有缺失历史。
- 旧ADP无法处理零可见节点或不足两个可见节点。仅在预训练跳过空批不够：最终推理仍需定义空输入如何输出，且不能删除此类评价窗口来规避。
- 历史决定 `D-20260916-SIMPLIFY-SLPF-AND-RERUN-ONLY-NECESSARY` 明确由用户要求简化为源forecast主干，当前实现符合该决定。

若重新做三阶段，应独立设计：仅K内合法辅助监督、部分观测保留规则、零/单个可见节点边界、阶段权重移交与冻结、恢复状态、最终canonical33选择。零观测fallback或跳过预训练样本均须明确标为适配，不能借用ALFNet模块或放宽信号边界。这是一项独立实现工作，不是轻量bug修复。

## STBP：不仅是没有跨年训练

源 `references/STBP/src/model/model.py` 创建`[base_node_size,3*hidden_channel]`模式库；`expand_adaptive_params()`为新增节点拼接新参数。`references/STBP/src/trainer/default_trainer.py`在后续年份冻结其他参数、训练pattern_bank；`main.py`按年份读取新图和新数据。

当前 `model/comparison_model/STBP/stbp.py`使用`pattern_encoder = StructuralNodeEncoder(...)`，无模式库扩展；wrapper固定全N图。空间注意力仍存在，所以不能将“固定图”误写为没有动态空间运算。但参数化已不同于原始模式库，不只是数据加载形式不同。

恢复一个原始参数表本身不难，恢复完整方法却还需要增量数据阶段、节点顺序、监督范围、扩展后优化器和检查点管理。若新节点持续训练阶段使用其动态监督，就改变了本项目未见节点及固定权重评价的条件；若从不训练新节点模式库，则也不是原持续学习方法。

节点ID参数本身不等于读取了保留节点真实信号。结构编码器替换是本项目既定的归纳能力约束，不能把这项实验设计选择概括成所有节点嵌入都必然数据泄漏。但放宽这一约束也不是当前协议下的等价修复，需要独立说明与新实验身份。

## CPU核验

使用`/home/xd/anaconda3/envs/ReaCH/bin/python`、seed32、单CPU线程、6节点、12→12合成输入。两份源码模块分别加载，共享同名权重，将适配器结构编码产生的表示复制到源位置表/模式库，模型均eval。

| 核验 | 输出形状 | 最大绝对差 |
|---|---|---:|
| SLPF ADP | 2×12×2×1 | 0 |
| SLPF forecast | 2×12×4×1 | 0 |
| SLPF aggregation | 2×12×2×1 | 0 |
| STBP backbone | 12×12 | 0 |

上述检查定位的是“给定相同表示，保留算子是否一致”，不是训练等价、模型容量等价或性能验证。没有运行优化器，没有使用GPU和真实数据。脚本与结果见[核验目录](slpf_stbp_integrity_20260926/)。

## 处置建议

- 保留历史结果，不声称两个算法已经完整复现。
- SLPF继续标`SLPF（单阶段预测主干适配）`；STBP更准确标`STBP（结构模式离线适配）`。
- 若篇幅只允许少数最新核心基线，优先选择与当前训练信号/固定权重设置相容、能保留核心方法的论文；不能只以STBP的2026年份抵消实质性适配差异。
- 本轮不实现半套新模式库、不恢复已失败的三阶段开关，不将未验证新代码标为“完整适配完成”。无需为了本次审查重训已有结果。
