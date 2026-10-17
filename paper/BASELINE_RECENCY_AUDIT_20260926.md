# 对比方法时效性核验（2026-09-26）

范围：以 manuscript_draft_cn-v4.md 为已汇报版本，核验九项方法的发表信息，检索2025–2026年更近的相关方法。使用 Sciverse 做线索检索，以官方论文集、出版社、作者论文及官方代码核验。本次为定向文献审查，不是穷尽性系统综述；未检出不能解释为不存在。未改稿、模型、配置、结果，未训练或打开test。

## 结论

九项方法的年份和会刊组合均有出处支持，但不能统一称为三个方向的“最新模型”。第一类已有 GinAR+（TKDE2025）、Merlin（KDD2025）及 DCDM（TMC2026卷期，2025在线）；第二类已有 SIML-MSIGN（TRC2026）、GenCast（AAAI2026）。第三类已有2025/2026方法覆盖，STGCN与STID适合作为经典结构参照。

评价选择建议同时考虑任务接近程度、正式发表质量、时间、完整实现和共同数据边界。此处不作未经核验的统一CCF等级或期刊分区判断。TMLR、TRC、ICDM、CIKM与ICLR/KDD的定位各有不同，不能全部笼统标为同一等级“顶会”。

## 现有方法

| 方法 | 核验出处 | 建议 |
|---|---|---|
| UIGNN | [IEEE ICDM2023](https://ieeexplore.ieee.org/document/10415744/)，DOI10.1109/ICDM58522.2023.00152 | 保留为无历史位置预测代表；不是最新缺失预测方法；Xplore入库2024不改变会议年份 |
| GinAR | [作者KDD2024代码](https://github.com/GestaltCogTeam/GinAR)，DOI10.1145/3637528.3672055 | 保留既有结果，优先核查后续GinAR+ |
| MoGERNN | [TRC2025](https://www.sciencedirect.com/science/article/pii/S0968090X25000841)，DOI10.1016/j.trc.2025.105080 | 保留，最直接的归纳结构参照之一 |
| ST-FiT | [AAAI2025正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/33310) | 保留，直接面向缺少节点训练信号 |
| SLPF | [TMLR2025年8月正式PDF](https://openreview.net/pdf/39eedad6807366f97e5af769f94c6677d1880590.pdf) | 年份正确；当前单阶段版本应继续明确标为适配 |
| STGCN | [IJCAI2018正式论文](https://www.ijcai.org/proceedings/2018/505) | 经典图卷积参照，非近年SOTA |
| STID | [作者CIKM2022代码](https://github.com/GestaltCogTeam/STID) | 简洁MLP参照；作者明确为short paper |
| PatchSTG | [作者KDD2025代码](https://github.com/lmissher/patchstg)，DOI10.1145/3690624.3709177 | 保留；2024预印本与2025正式会议年份分开 |
| STBP | [ICLR2026正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7547048d73ed056286762b17434466a7-Abstract-Conference.html) | 已正式确认，保留；当前离线版本不能当作完整持续学习流程 |

## 更新候选与适配边界

### 第一类：优先补充最新、直接相关的预测方法

**GinAR+，TKDE2025。** 作者发表列表明确给出题名、年份及DOI10.1109/TKDE.2025.3569649，是GinAR方向的后续工作。应优先核查完整论文、对比学习过程与对应实现，不能将现有GinAR权重改名当作GinAR+。IEEE页面本次遇到机器人验证，发表信息由作者一手列表确认；尚未确认独立官方GinAR+代码发布。
[作者发表列表](https://finleywang.github.io/publications/)，[IEEE入口](https://ieeexplore.ieee.org/document/11002729/)。

**Merlin，KDD2025。** 正式研究轨道论文，针对变化缺失率，使用完整观测教师蒸馏与多视图对比学习；已有官方代码。与动态观测支持的研究问题接近，优先级高。适配时教师只能使用训练节点K的合法训练信号，不能使用V/T完整历史；还需保留方法本身的训练步骤，明确所搭载预测主干。此为适配要求，不是声称已完成适配。
[正式论文名单](https://www.kdd.org/kdd2025/research-track-papers-2/)，[作者全文](https://arxiv.org/html/2506.12459v1)，[代码](https://github.com/ChengqingYu/Merlin)。

**DCDM，TMC2026。** A Unified Diffusion Framework for Traffic Imputation and Prediction With Physical Priors，卷25期1，341–357页；2025-07-21在线，2026年1月卷期。联合填补与预测，有物理先验和扩散推理；属于更新的候选，但代码完整性、额外物理量与计算预算尚需核验，暂列备选。
[IEEE论文](https://ieeexplore.ieee.org/document/11087699/)。

**机制补充而非年份补充：HD-TTS（ICML2024）、BiTGraph（ICLR2024）。** 前者依据观测和缺失模式组合多尺度表示；后者让时序卷积与图运算显式处理缺失。二者直接检验“支持条件影响预测计算”，比添加无关的新模型更有辨别力。具体12步输入及节点暴露适配仍待检查。
[HD-TTS正式论文](https://proceedings.mlr.press/v235/marisca24a.html)，[BiTGraph正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html)。

### 第二类：2026年已有直接相关研究

**SIML-MSIGN，TRC2026。** Structure-inductive Meta-learning Empowers Partial-to-global Traffic State Forecasting in Dynamic Sensor Networks，DOI10.1016/j.trc.2026.105564。固定物理路网、变化观测掩码、从局部到全局预测，问题非常接近；稿件第[10]项已引用，不能说此方向最新止于2025。建议首先查代码能否获得，以及zero-step评价能否在相同信号边界下实现。若使用few-step适应，须单独说明可用监督，不能与固定权重混为一个设置。此次未确认公开官方代码。
[出版社入口](https://www.sciencedirect.com/science/article/abs/pii/S0968090X26000525)；本地全文 `references/paper/SIML-MSIGN.md`。

**GenCast，AAAI2026。** Generalising Traffic Forecasting to Regions Without Traffic Observations，使用物理约束、空间分组及外部天气信号。已有官方代码。其未观测区域预测很相关，但全文定义包含ERA5天气及12小时天气上下文，与本项目纯交通12步输入不同。宜纳入相关工作，实验纳入需先统一额外数据条件或明确单独扩展设置；简单删除这些模块后不可宣称完整复现。
[AAAI正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38607)，[作者全文](https://arxiv.org/html/2508.08947v2)，[作者代码](https://github.com/suzy0223/Generalising-Traffic-Forecasting-to-Unobserved-Regions-GenCast)。

### 第三类：已有近年覆盖，不必整体换新

建议保留PatchSTG2025与STBP2026作为近年代表，STGCN2018与STID2022作为经典对照。最新年份不保证在共同归纳协议下最合适或最强。现有SLPF单阶段和STBP离线适配是否保留论文核心能力，是比补一个更新年份更需说明的问题。

检索中也发现2026年纯填补、事故引导及外部知识方法，但任务/输入并不自动满足本实验边界，不因更新而直接纳入。例：[STAMImputer，IJCAI2025](https://www.ijcai.org/proceedings/2025/382)是交通填补，不能不加预测流程便替代未来预测基线。

## 建议顺序（尚非训练计划）

1. 第一类先审查GinAR+和Merlin，至少纳入一个可忠实实现的2025方法；若只能选一个，Merlin已有公开实现，实施可行性更明确，GinAR+则是最直接的版本更新。
2. 第二类优先审查SIML-MSIGN2026，其动态传感器问题比单纯追逐通用模型年份更贴近本文；GenCast作为有额外信息要求的候选。
3. 若预算允许补一个机制强对照，优先考虑HD-TTS或BiTGraph。
4. 保留现有强相关及经典结果；是否移附录依照类别覆盖和篇幅预先决定，不依据新模型输赢决定。

建议表述：“与经典时空预测方法及近年来不完整观测、归纳与部分感知预测的代表方法比较”，不写“与三个方向全部最新SOTA比较”。文献选择不等于已证明新增方法在本文协议下更强。

## 已汇报稿的引文订正建议

- STBP [18]可用ICLR2026官方论文集替换“作者仓库标注”。
- SLPF [9]当前URL中ID为`Ff08aPjVjD`，正式PDF给出的ID为`Fj08aPjVjD`（第二字符小写j）。论坛本次访问受限，但正式PDF出版抬头及ID可读取；也可直接链接正式PDF。
- 本次只记录订正建议，未修改用户已汇报的v4稿。
