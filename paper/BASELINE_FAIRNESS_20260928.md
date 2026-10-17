# 十个对比模型的实现与训练公平性审计

日期：2026-09-28。范围：只读当前模型、源代码、配置及已保存报告；没有训练、GPU推理或test访问。本文件不改变模型/计划身份。当前结果来源为 `paper/validation_seed32_baseline_extension_20260927/results_and_provenance.json` 的55条记录。

## 结论

当前十个baseline没有调用ALFNet专属ALFGraph、ALFTemporal、生命周期role/age条件化或Focal目标。共同静态图、掩蔽输入、训练监督有效mask和日历不属于ALFNet独有创新。但是，“没有使用ALFNet模块”不等于“完整原样复现源方法”：五个baseline引入共享StructuralNodeEncoder，SLPF/STBP的训练范式也有实质改动。论文必须称其为共同协议下的适配结果，不应声称全部源模型与超参数原样不变。

本结论依据当前调用边界与直接代码检查，并非本次重新执行所有反事实泄露测试。`family_utils.SourceNativeV4Mixin`读取common masked traffic；独立UIGNN/GinAR/Merlin适配器也不请求生命周期特征。`experiments/v4_runner.py`的Focal分支要求Full role-balanced contract，各baseline实际loss均不是Focal。

## 逐模型源架构及参数差异

| 模型 | 保留内容及当前模型设置 | 必须披露的差异 |
|---|---|---|
| UIGNN | 官方full-graph IGNNK量化/重建分支，隐藏宽度100，12步历史 | 输出层改为每个预测步独立3分位数，即3×12输出；非单步复制。共同mask监督替代私有数据管线 |
| GinAR | 源GinAR循环/注意力核心，当前16维隐藏、8维图表示、2层、dropout .15 | 图embedding、节点矩阵/偏置替换为StructuralNodeEncoder参数化；LayerNorm节点仿射共享。这是参数化改动，非单纯张量重排 |
| Merlin | STID teacher/STID_CL student；node/embed/TOD/DOW维度64，3层，teacher CL宽4、student CL宽16，TOD288/DOW7 | 保留源node-ID参数；共同train-K scaler替代全数据minmax；合法mask替代零sentinel；全节点CL投影前屏蔽非K。训练预算改为每阶段100/patience10，非源教师200/学生104固定预算 |
| MoGERNN | MoGE补全专家、路由及DCRNN预测结构；当前batch64自回归recipe | 同一full-N掩蔽输入、冻结拓扑、合法teacher-forcing及common目标；不是ALFGraph替代。实际超参数与源默认数据配置不可混称逐项相同 |
| ST-FiT | 源图学习、合成节点/混合、可选VAE分支、STGCN核心；hidden64、kernel3、K3 | donor限制在合法训练支持；修复batch图偏移及有限归一化；等价dense Cheb计算。源Gumbel评估采样仍保留；早停按forecast epoch计数 |
| SLPF | 源forecast核心，embed64、3层 | wrapper把所有N位置交给forecast分支，num_unsensed_locs=0；不执行ADP/aggregation完整三阶段。节点位置embedding也改StructuralNodeEncoder。不能标称完整SLPF复现 |
| STGCN | 采用references/ST-FiT中的STGCN_n实现，hidden64、kernel3、K3；关闭图学习/合成分支 | 是该源仓库的STGCN实现，不是从IJCAI原仓库逐字复制；共同全N图及监督 |
| STID | 源日历embedding、3层MLP与预测头；node/embed/TOD/DOW各32，TOD288/DOW7 | node-ID表改StructuralNodeEncoder；已修复真实日历索引。与Merlin的64维STID骨干不同，不能称同容量对照 |
| PatchSTG | 源patch attention，5层，input64/node32/TOD32/DOW32，temporal12×1，factors1 | node-ID表改StructuralNodeEncoder；空间分区由manifest决定。flow使用32个BFS分区，容量03/04/08为12/10/6；容量不是分区数。speed采用现有坐标布局 |
| STBP | 源DLGA/频域/encoder/head，hidden192，12→12，dropout0 | 源可扩展逐节点pattern bank改StructuralNodeEncoder，且不执行持续扩展训练。应称离线结构编码适配，不能仅写“接口适配” |

这里列出已核验的当前关键参数，**不是逐篇原论文所有超参数默认值的完全等同性证明**。来源位置：`model/comparison_model/{模型}/`、`configs/paper/final_seed32/`、`configs/paper/baseline_extension_20260927/`，以及 `references/` 对应源代码。SLPF/STBP逐核心对照见 `paper/SLPF_STBP_INTEGRITY_20260926.md`。

StructuralNodeEncoder使用冻结图度数与可用坐标，是共享结构函数，不含role/age/Focal。它并不是ALFNet创新模块，但替换node-ID容量会改变基线归纳能力与函数族。尤其Merlin此次保留node-ID，五个其他基线替换node-ID的政策并不完全一致；这应披露，不能以“所有节点embedding都统一替换”概括。node-ID本身不等于读取held-out动态真值；具体是否违规要区分拓扑可见、动态信号可见及参数训练路径。

## Merlin额外训练输入与源文件核验

本次重新读取字节并计算SHA256，`SOURCE_MANIFEST.json`列出的4个文件（stid_arch.py、mlp.py、Loss.py、mask_metric.py）均与references逐字节一致且与manifest一致。**字节一致仅覆盖这4个源文件**；adapter和训练程序有上述协议改动。

教师可以访问当前train origin中自然可用、但被合成支持掩码隐藏的train-K历史；学生不能直接访问该完整历史，只通过教师表示/输出蒸馏获益。这是源方法所需的训练期特权信息，确实不同于仅接收当次masked history的普通baseline，应公开说明。`teacher_history_for_batch`先限定train split/origin及K列，非K为零；教师/学生监督与KD作用域限定K，验证两者都只使用共同masked输入。没有额外V/T真值，也没有扩大验证可见历史。

## 实际训练配置：以执行代码为准

| 模型 | max epoch / patience / start | batch | 实际optimizer；LR；weight decay | 实际scheduler |
|---|---|---:|---|---|
| UIGNN | 100 / 10 / 0 | 4 | Adam；.001；0 | 无 |
| GinAR | 100 / 10 / 0 | 16 | Adam；speed .006、flow .001；0 | MultiStep [1,15,40,70,90]，gamma .5 |
| Merlin | **每阶段**100 / 10 / 0 | 16 | Adam；.0002；.0001 | teacher [1,10,25,50,75,90,100,125,150,190]；student [4,13,28,53,78,93,103]；gamma .5，超预算里程碑自然不触发 |
| MoGERNN | 100 / **3 / 20** | 64 | AdamW；.001；.0001 | 无 |
| ST-FiT | 100 / 10 / 0，patience按forecast epoch | 16 | Adam；.002；.0001 | 无 |
| SLPF | 100 / 10 / 0 | 16 | **Adam**；.001；.0001 | 无 |
| STGCN | 100 / 10 / 0 | 16 | Adam；.002；.0001 | 无 |
| STID | 100 / 10 / 0 | 16 | Adam；.002；.0001 | MultiStep [1,50,80]，gamma .5 |
| PatchSTG | 100 / 10 / 0 | 16 | AdamW；.002；.0001 | 无 |
| STBP | 100 / 10 / 0 | 16 | AdamW；.001；.0001 | 无 |

证据：`scripts/train_v4.py:425`的optimizer模型分派、`:440`的scheduler分派、`args.py:52`的start默认0、`scripts/train_merlin_v4.py`的两阶段覆盖。特别是SLPF YAML写AdamW但实际走Adam；GinAR YAML scheduler None但实际运行MultiStep。这是描述/配置可读性问题，不能用YAML文本推断已训练优化器。

Merlin已保存报告实际教师/学生epoch数：METRLA 45/40、PEMSBAY 24/65、PEMS03 55/100、PEMS04 95/86、PEMS08 85/80。不能说Merlin总预算与单阶段模型完全相同，也不能继续沿用早先“200/104固定训练”的陈述。

GinAR flow .006在初轮出现非有限梯度后，独立恢复计划冻结.001，完整一轮训练+33格验证后运行；speed配置未改。证据 `configs/paper/baseline_recovery_20260927/plan.json` 与 `paper/baseline_extension_20260927/RECOVERY.md`。这不是按test选参。

## 应统一什么，以及现有结果受影响范围

必须统一：数据/节点划分、train-K scaler、输入预测长度、合法动态信号范围、测试封存、验证checkpoint endpoint（33格LifeAvg MAE）、指标计算与比较样本。已经采用这些共同规则；同一指标和样本比“所有模型使用同一LR”更重要。

可以按源方法保留并披露：结构宽度/层数、优化目标、optimizer、LR schedule、batch及必要阶段训练。强制所有模型相同超参数会改变源方法，不能自动提高公平性。现有optimizer/LR也不能笼统称“全部源默认”，有共同常数LR适配及flow稳定性修正。

如决定统一训练预算，建议先定义“每阶段或总量”“epoch或optimizer步”“patience按何种阶段计数”，再发独立计划；不应事后把现有不同patience的结果重标为相同。MoGERNN的3/20与其余10/0、Merlin两阶段总量及ST-FiT阶段计数是明确例外。统一这些政策会影响该模型所有数据集已训练结果，需要新训练，而不是只改配置文本。

现有55条记录仍是已冻结适配协议下的有效validation结果，但不能据此宣称十篇源方法的完整重现。恢复StructuralNodeEncoder为node-ID将影响GinAR、SLPF、STID、PatchSTG、STBP全部五数据集；恢复SLPF三阶段或STBP持续学习则是新的研究/协议范围。上述审计不自动授权这些改动或重训。

## 补充：本地源入口的逐项默认值对照

以下直接读references中的可执行训练入口/配置，不用论文记忆代替。相同指列出的参数相同，不代表整体方法等价；统一12→12、单目标通道、节点数及信号边界属于明确协议适配，不作为未修复漏洞。

| 模型 | 本地源权威 | 已核对相同项 | 明确不同项／无法唯一确定项 |
|---|---|---|---|
| UIGNN | `references/UIGNN/main_full_graph.py:22`及`structure_full.py` | z100、K1；EDL=False、MC_dropout=False；Adam .001、batch4 | 源Max_episode300，当前100＋patience10；源history24及单步输出已按12→12任务适配。源另有main.py分支，本项目选择full-graph分支，不混称所有分支默认 |
| GinAR | `references/GinAR/main_ginar.py:37` | emb16、graph8、layer2、dropout.15、Adam、batch16、speed LR.006、源milestones[1,15,40,70,90] | flow LR.001是已记录稳定性修正；源3输入通道改本协议1交通通道；节点参数化改变见上文。源固定100，当前100＋早停 |
| Merlin | `references/Merlin/train_teacher.py`及`train_student_merlin.py` | 四类embedding宽64、3层、dropout.15、teacher CL4/student CL16；Adam .0002、WD.0001、batch16、clip5；源scheduler边界一致 | 源teacher200/student104固定预算改每阶段100＋patience10；train-K合法信息限制、scaler及目标mask为协议适配。四个源文件字节相同不覆盖训练入口 |
| MoGERNN | `references/MoGERNN/scripts/MoGERNN-train.sh`及`args.py:11` | d_model64、encoder2/decoder1、k_hop1、5类专家、top3；AdamW .001、WD.0001、batch64、无scheduler | 源启动脚本max_epoch200/patience10，当前100/patience3/start20；所以当前“source early stopping”配置名不能作为与这个源入口完全一致的证明。源stride12改协议stride1 |
| ST-FiT | `references/ST-FiT/configs/STGCN_ST/PEMS04.yml`及`src/trainer/train.py:90` | hidden64、kernel3、K3、dropout0；graph64、合成/VAE配置hidden64/latent64/2层/dropout.1；Adam .002、WD.0001、batch16；anchor.5、sparse.9、gumbel tau1、is_vae False | **源实际MultiStep[1,50,100]/gamma.5，当前恒定LR**。窗口24在源为历史＋horizon的总窗口，不能误读成当前少用12步历史。源另有数据集/模型配置，不存在跨所有配置的唯一默认 |
| SLPF | `references/SLPF/model/configs/{LA,BAY,PEMS03,PEMS04,PEMS08}.conf`；`model/run_SLPF.py:124`；`model/SLPF.py` | 3层、MLP dropout.15、实际Adam .001、无LR衰减、max epochs100 | **五个源配置embed32，当前64**；源batch64→16，WD3e-4→1e-4，patience30→10；源三阶段→forecast单阶段。输入/输出窗口差异属任务适配，不能掩盖上述结构宽度差异 |
| STGCN | `references/ST-FiT/configs/STGCN/PEMS04.yml`及`src/trainer/train.py` | hidden64、kernel3、K3、dropout0、关闭图学习/合成；Adam .002、WD.0001、batch16 | **源MultiStep[1,50,100]/gamma.5，当前恒定LR**；来自ST-FiT仓库内STGCN_n版本，不代表IJCAI原仓库唯一配置 |
| STID | `references/STID/stid/{METR-LA,PEMS04}.py`及`stid/arch/mlp.py` | node/embed/TOD/DOW各32、3层、dropout.15、288/7、两日历开启；Adam .002、WD.0001、milestones[1,50,80]/gamma.5、max100 | **源batch64，当前16**；node-ID→StructuralNodeEncoder改变结构；共同早停另加。所核对speed/flow配置上述宽度一致 |
| PatchSTG | `references/PatchSTG/config/{SD,CA,GBA,GLA}.conf`；`main.py:47`；`models/model.py` | 5层、input64、TOD/DOW32、temporal12×1；AdamW .002、WD.0001、batch16；核心未添加可配置dropout | **源factors为8/32/128，当前1**，改变跨patch交互分组；源spatial512/2048/4096分区，当前32，属于规模/布局适配；node维源SD64、其余32，当前32，不能称与所有源配置相同。源MultiStep[1,35,40]，当前恒定LR；源max50→100。本地无对应五数据集源配置，无法唯一确定“该数据集官方默认” |
| STBP | `references/STBP/conf/STBP_PEMS.json`及`src/trainer/default_trainer.py:47` | hidden192、in/out12、dropout0、AdamW | **源LR.01→当前.001；batch64→16；源epoch200→100**。源AdamW未传WD，PyTorch默认.01，当前显式.0001；这是实际优化差异，不是源JSON缺字段就等于0。pattern bank及持续学习变化见上文 |

因此，用户要求的“模型相关超参数按源代码”目前**尚未完全满足**：最清晰的结构超参数差异是SLPF embed32→64、PatchSTG factors→1及其布局尺度；训练recipe差异还包括ST-FiT/STGCN/PatchSTG取消scheduler、STBP LR/WD、STID/SLPF batch和MoGERNN早停。它们不证明已有结果错误，但需要明确选择保留适配还是另起修复重训；不能只改审计措辞宣称已恢复源默认。本次仅补充证据，没有改变运行中冻结代码、配置或训练队列。
