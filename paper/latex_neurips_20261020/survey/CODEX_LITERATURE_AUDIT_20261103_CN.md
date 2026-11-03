# Claude 文献矩阵核验与下一阶段建议

核验日期：2026-11-03。对象：`LITERATURE_MATRIX_CN.md`（2026-11-02版）。

结论：已有矩阵可作为资料底稿，但尚不能认可“逐篇完成双源核实”、近邻排序和 S1–S9 原稿。问题不只是漏 DOI，还包括漏掉直接近邻、混用论文版本、漏读附录、把原论文设置当成方法能力边界。建议 Claude 先修订阶段1，再形成相关工作组织方案和阶段2标准；本轮不改正文或实验表。

本次按仓库入口阅读上下文、任务说明、当前中文正文、已有调研及适配审计，检查原核实记录，并对关键结论、未核实条目和引用链进行联网复核。使用 nature-ref-verifier 工作流及其要求的并行分组核验。没有重新逐篇精读全部75个编号条目，也不宣称所有字段再次通过双源核对。Crossref直连失败；官方论文集、作者全文与出版方可得页面是本次新增结论的主要依据。访问失败仅表示本次不可读取。

## 1. 优先修正的内容

下列行号针对核验时的原矩阵，后续编辑会漂移，应同时按编号定位。

| 原位置 | 问题 | 修订依据及动作 |
|---|---|---|
| C12，约294–299、510行 | HD-TTS“无交通数据集”错误 | [作者全文v2 Appendix F/Table 6](https://arxiv.org/html/2402.10634v2#A6)包含METR-LA、PEMS-BAY的时间块和时空块缺失实验。补入数据集；正式页码为34846–34865，见[PMLR](https://proceedings.mlr.press/v235/marisca24a.html)。不能以没有交通实验降低它的优先级。 |
| E10，381行 | FUNS被误标为未正式发表 | 正式为**ICDM Workshops 2022**，740–747，DOI **10.1109/ICDMW58026.2022.00101**。作者Andreas Roth、Thomas Liebig；题名 *Forecasting Unobserved Node States with spatio-temporal Graph Neural Networks*。[作者仓库](https://github.com/roth-andreas/funs)与[DBLP记录](https://dblp.org/rec/conf/icdm/RothL22.html)支持；注意不是ICDM主会。撤销“未发表”的排除理由。 |
| C16，334–340行及各汇总 | UniSTOK版本混用 | 现卡片对应[v1](https://arxiv.org/abs/2603.05301v1)。截至核验日[v2](https://arxiv.org/abs/2603.05301v2)已于2026-05-09更新：题名 *Uniform Inductive Spatio-Temporal Kriging*；作者Lewei Xie、Haoyu Zhang、Yulong Chen、Liangjun You、Zongxian Yang、Yifan Zhang；方法改为可靠性引导信号调节RSR与残差偏差校准RBC。主卡更新为v2；如讨论旧版掩码仿射调制，必须明确引用v1，不能混成同一版本。 |
| C05，215、590行 | STSM遗漏正式DOI | **10.48786/edbt.2024.42**，EDBT 2024，488–500；[官方全文](https://openproceedings.org/2024/conf/edbt/paper-123.pdf)首页有DOI。 |
| C05，144、217、222、508行 | STSM输入差异不只是24步历史 | 同一官方PDF §4.1还使用OSM的POI、建筑/公园及道路属性参与选择性遮蔽。需把外部静态特征写入D5/S4，与GenCast采用一致标准。 |
| C02，141、186、506行 | SIML被概括为必须微调 | 本地原文明确比较Step-0与Step-5：MSIGN和SIML学到的初始化均应分别检查零步使用；不能仅给MSIGN标“零步”。原文Eq.(24)同时监督观测与未观测节点，Table 1给出两者损失权重均0.5，D1差异不必只靠掩码轮换推断。见[本地原文](A:/projects/ALFNet/references/paper/SIML-MSIGN.md:252)。 |
| C02及8.2节 | “没有代码”强于证据 | 改为“截至检索日未定位可核验的官方公开实现”。原文确有confidential数据声明，但这既不证明代码不公开，也不能把METR-LA、PEMS-BAY概括成保密数据。记录声明与所用公开基准之间的待澄清处。 |
| C07/C13，146、152、245、310行 | 摘要支撑不了全部否定判断 | SC-SAMG支持期/元学习、MAS-Mamba掩码/观测间隔条件化有依据；“不涉及新增”“D6否”等若未读相关全文，标“未报告/待核实”，不要填确定的“否”。 |
| 125–130行、0节四条件结论 | 把ALFNet的全部场景概括成目标节点训练无信号 | 正文4.1中Unseen/Join用保留节点，Steady/Fault用已知节点。D1必须按场景说明。当前结果不等于在同一批训练未见节点上同时评估接入与故障。 |

两项细节也应修：ST-FiT原设置与Ready的对齐，只是源实验设置对照，不能据此说适配后不能在其他场景运行；SLPF的OpenReview ID存在旧PDF提取值`Fj08aPjVjD`与[当前v3全文](https://arxiv.org/html/2408.02689v3)明载`Ff08aPjVjD`的差异。建议采用当前作者链接并保留核查记录；本次OpenReview直接访问触发验证，未独立读到论坛正文。

## 2. 遗漏与可补条目

### 2.1 最重要的新遗漏

| 工作 | 核验到的书目信息与来源 | 对ALFNet的意义、证据边界 |
|---|---|---|
| **FRIGATE** | Mridul Gupta、Hariprasad Kodamana、Sayan Ranu；*Frigate: Frugal Spatio-temporal Forecasting on Road Networks*；KDD 2023；DOI **10.1145/3580305.3599357**；arXiv:2306.08277。[作者全文](https://arxiv.org/html/2306.08277v1)、[KDD目录](https://kdd.org/kdd2023/wp-content/uploads/2023/08/toc.html)、[官方代码](https://github.com/idea-iitd/Frigate)。仓库BibTeX有`10.1134`笔误，勿照抄。 | **应升级为核心卡片。** 部分感知、无历史位置预测及不规则观测，原实验12→12；§4.6冻结参数做拓扑变化和随机丢弃快照实验。后者不是ALFNet的节点级Join/Fault阶段协议。正式适配前还须查邻域统计量及训练监督范围，不能只凭inductive标签认定完全同协议。 |
| **SCPT** | Arian Prabowo、Hao Xue、Wei Shao、Piotr Koniusz、Flora D. Salim；*Traffic forecasting on new roads using spatial contrastive pre-training (SCPT)*；DMKD **38:913–937 (2024)**，online 2023-09-27；DOI **10.1007/s10618-023-00982-0**；arXiv:2305.05237。[出版方全文](https://link.springer.com/article/10.1007/s10618-023-00982-0)、[代码](https://github.com/cruiseresearchgroup/forecasting-on-new-roads)。 | 新道路预测、推理不重训，但使用目标道路两天历史。§3.6周期分解还涉及主干训练未见道路的信号，必须分别审查预处理与模型训练的暴露范围。是冷启动与输入预算的重要近邻，不是自动相容基线。 |
| **STOP** | Jiaming Ma、Binwu Wang、Pengkun Wang、Zhengyang Zhou、Xu Wang、Yang Wang；*Robust Spatio-Temporal Centralized Interaction for OOD Learning*；ICML 2025，PMLR **267:42165–42192**。[官方论文集](https://proceedings.mlr.press/v267/ma25s.html)、[作者代码](https://github.com/PoorOtterBob/STOP)。 | 结构OOD与归纳学习，需与STONE一起优先核查新增/移除节点设置、目标历史与更新范围。当前已确认出版信息及相关性，完整同协议适用性待核。 |
| **GenST** | Shuhao Li、Weidong Yang、Changan Liu、Wei Zhuo、Yingbo Zhou、Fan Zhang、Siqiang Luo；*Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States*；[arXiv:2610.08818v2](https://arxiv.org/abs/2610.08818v2)，2026-10-08更新，预印本。 | 未观测节点未来生成，与任务直接相关；额外节点语义和LLM预训练资源需要单列。当前仅元数据/摘要级核验，不能宣称完整协议相容或已正式发表。 |
| **GSTAE** | Ao Wang、Yongchao Ye、Xiaozhuang Song、Shiyao Zhang、James J. Q. Yu；*Traffic Prediction With Missing Data: A Multi-Task Learning Approach*；TITS **24(4):4189–4202 (2023)**；DOI **10.1109/TITS.2022.3233890**。[作者正式全文](https://file.jqyu.me/pub/traffic-prediction-with-missing-data-a-multi-task-learning-approach.pdf)。 | 插补预训练与预测任务共享编码器，不是简单“先填值再预测”；补齐联合/多任务脉络。当前不据此推断节点归纳能力。 |

### 2.2 从“未核实线索”升级

| 工作 | 可补信息 | 归类 |
|---|---|---|
| STA-GANN | CIKM 2025，1726–1736；DOI **10.1145/3746252.3761045**；arXiv:2508.16161；[作者全文v2](https://arxiv.org/html/2508.16161v2)首部有正式出版信息。作者Yujie Li、Zezhi Shao、Chengqing Yu、Tangwen Qian、Zhao Zhang、Yifan Du、Shaoming He、Fei Wang、Yongjun Xu。 | 归纳时空克里金；不能把同时刻估计直接并入未来预测表。 |
| RADGCN | *Relation-diffusion augmented network-wide traffic state estimation under sparse sensor deployment*；Tao Guo、Ningkang Yang、Kevin Yu、Panagiotis Angeloudis、Constantinos Antoniou；Communications in Transportation Research **6(3):9640046 (2026)**；DOI **10.26599/COMMTR.2026.9640046**。[出版方](https://www.sciopen.com/article/10.26599/COMMTR.2026.9640046)。 | 稀疏传感的网络级状态估计。 |
| ST-TTC | Wei Chen、Yuxuan Liang；*Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting*；NeurIPS 2025；arXiv:2506.00635；DOI **10.52202/085713-5210**。[官方论文集](https://papers.nips.cc/paper_files/paper/2025/hash/e4a04155055153f18b93824825cd6e3d-Abstract-Conference.html)。 | 测试期校准/适应，作为固定参数预测的边界对照。 |
| RT-GCN | Yutian Liu、Soora Rasouli、Melvin Wong、Tao Feng、Tianjin Huang；*RT-GCN: Gaussian-based spatiotemporal graph convolutional network for robust traffic prediction*；Information Fusion **102:102078 (2024)**；DOI **10.1016/j.inffus.2023.102078**。[出版方](https://www.sciencedirect.com/science/article/pii/S1566253523003949)、[机构全文](https://pure.tue.nl/ws/files/309332629/1-s2.0-S1566253523003949-main.pdf)。 | 旧调研已有但本轮漏核；高斯表示及方差注意力应补入缺失/噪声鲁棒预测。 |
| GMN/SGMN | Zhiyong Cui、Longfei Lin、Ziyuan Pu、Yinhai Wang；*Graph Markov network for traffic forecasting with missing data*；TRC **117:102671 (2020)**；DOI **10.1016/j.trc.2020.102671**。[出版方记录](https://www.sciencedirect.com/science/article/pii/S0968090X20305866)。 | 旧调研已有的联合状态推断与预测脉络；无须因不是近三年而删除经典相关工作。 |

NeurIPS 2026的 *Traffic STGNNs across Sensor, City, and Time Shifts: Routing Concentration Tracks Sensitivity to Sensor Dropout* 确实出现在[官方2026目录](https://neurips.cc/Downloads/2026)，链接到poster/139842。本次未读到poster正文，**作者、轨道、全文及具体结论仍待核**。保留线索，不能因缺DOI/arXiv断定不可引用，也不能仅凭题名确认路由结论。

### 2.3 现有条目的DOI、页码与代码补全

| 编号/工作 | 补充信息与第一手来源 |
|---|---|
| E15 SPIN | DOI **10.52202/068431-2324**；[NeurIPS官方页](https://papers.nips.cc/paper/2022/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html) |
| E26 Taming Local Effects | DOI **10.52202/075280-2417**；[NeurIPS官方页](https://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html) |
| E42 STRAP | DOI **10.52202/085713-3934**；[NeurIPS官方页](https://papers.nips.cc/paper_files/paper/2025/hash/ab062c6281873221743383f9619e07d2-Abstract-Conference.html)。需读持续学习与推理检索的区别，不能由“无task-specific fine-tuning”推断全流程从不更新。 |
| E44 GMoE | DOI **10.52202/075280-2211**；[NeurIPS官方页](https://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html) |
| E45 Flex-MoE | DOI **10.52202/079017-3135**；[NeurIPS官方页](https://papers.nips.cc/paper_files/paper/2024/hash/b2f2af5403042b1344f4e93b35fb67d9-Abstract-Conference.html) |
| E50 LargeST | DOI **10.52202/075280-3293**；[NeurIPS官方页](https://papers.nips.cc/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html) |
| E07 SATCN | [官方代码](https://github.com/Kaimaoge/SATCN)可补；本轮仍只确认预印本arXiv:2109.12144，不猜正式DOI。 |
| E12 SUSTeR | [官方代码](https://github.com/ywoelker/SUSTeR)可补；2023会议与2025 arXiv上传年份不同不构成错误。 |
| E19 BiTGraph | [官方代码](https://github.com/chenxiaodanhit/BiTGraph)可补；[ICLR永久论文页](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html)足以确认出处，无需强求DOI。 |
| E31 OpenCity | [官方代码/权重](https://github.com/HKUDS/OpenCity)可补；当前记录的TIST 2025及DOI 10.1145/3773912有[arXiv](https://arxiv.org/abs/2408.10269)支持。 |

E20 S4M第一作者是**Jing Peng**，现表Peng没有错，不能照DBLP键误改为Jing。E23 IBN、E37 TFMoE本次未找到可靠正式录用证据，保留预印本状态。AGCRN、ICLR/TMLR条目没有核到正式DOI时，保留官方永久页面和arXiv即可；arXiv/DataCite DOI与正式论文DOI需分列。

## 3. 调研记录本身应整改

1. S2返回DBLP键仍是同一条聚合记录，不构成独立访问DBLP。原始`s2_ids2.json`中FUNS、IBN只有429，不能由现存记录证明“全部S2成功”。将“全部两路核验”改为逐篇、逐字段的真实成功路径。
2. 扩展表大量只写缩写和“某某等”，缺完整题名及作者清单。正文可简写，但核验附表应有完整元数据、版本、来源链接、访问日期及未核字段。
3. “旧28项全部重核”与9.3明确说STGformer、RT-GCN、GMN未重核矛盾；补旧清单映射，或收回全量承诺。
4. “18+57”是编号条目数，C18含两篇、E43又列前作，另有未编号STGCN。重新区分条目数、唯一论文数和版本数，不能直接把编号算作唯一论文数。
5. F只表示读过指定段落，不表示整篇已核。给设置否定项、实验✓、代码R分别附节/表/仓库依据；“未在所读部分发现”与“论文明确没有”不能混用。
6. 代码状态分为：未检索、论文给链接、仓库已访问、关键实现已检查、复现已运行。此次公开链接核验不等于可运行性验证。
7. 不可用“没找到同时满足四条件的论文”直接推导方法原创性；特别是把自定义分阶段评价放入合取条件，会过度缩小近邻范围。明确区分问题覆盖、计算机制和评价设计三种贡献。

## 4. 近邻排序与相关工作组织建议

原排序不建议原样认可。下面是**精读及相关工作论证优先序**，不是性能排名，也不是自动必跑十篇。相邻位次不代表可量化的严格距离。

| 顺序 | 工作 | 保留在核心讨论的原因与主要边界 |
|---|---|---|
| 1 | MoGERNN | 最直接的问题/图专家近邻；已有新增与失效传感器设置。差异须落到窗口内轨迹、逐步条件传播及评价，不能写它不研究动态传感器。 |
| 2 | SIML-MSIGN | 部分到全局、掩码门控、变化布局；零步/少步分开，训练监督边界和历史预算均不同。 |
| 3 | FRIGATE | 本次关键漏文；部分感知、归纳预测、不规则观测和固定参数适应。须完成统计量/监督审计后裁定直接基线资格。 |
| 4 | UIGNN | 无历史位置未来预测的直接近邻；不把源24步输入当作一票否决理由。 |
| 5 | ST-FiT | 训练无目标信号、部署有历史的近邻；对应Ready侧，归在节点归纳/冷启动，不归为缺失鲁棒模型。 |
| 6 | FUNS | 未观测节点预测的早期直接脉络，已有正式workshop发表和公开实现。 |
| 7 | STSM | 区域级无观测未来预测；需要披露外部静态特征和源输入长度。 |
| 8 | GenCast | 区域外推的新近方法；天气等额外信息需要独立预算层。 |
| 9 | HD-TTS | 缺失模式条件化计算的强近邻，并有交通实验；节点身份参数和填补范围须审计。 |
| 10 | Merlin | 不固定缺失率下的预测与训练期蒸馏；相关工作地位与现有结果是否有效是两件事。 |

GinAR/GinAR+、GCN-M、SLPF仍必须在相应类别中讨论，不因不在这十项就排除。SCPT、STOP、STONE、STRAP列为优先补读对象；其完整协议核清后允许据证据调整上述顺序。MAS-Mamba、TESTAM、GRU-D、UniSTOK用于机制边界；不宜把摘要级MAS-Mamba排在已全文确认的直接任务工作之前。发表渠道不决定概念相关性。

建议相关工作按四个论点组织，而不是逐篇罗列，也不必把八个检索方向都变成正文小节：

| 建议类别 | 文献组织 | 对应实验对象 |
|---|---|---|
| 未观测/新位置的归纳预测与部分感知 | FUNS、UIGNN、MoGERNN、FRIGATE；ST-FiT/SCPT的部署历史；STSM/GenCast区域外推；SLPF训练期临时传感器 | 直接近邻及明确标注的受限适配 |
| 不完整观测下的预测与支持条件化 | GMN、GCN-M、GSTAE、GinAR/GinAR+、Merlin、HD-TTS、MAS-Mamba；GRIN等只说明插补来源 | 缺失鲁棒预测、简单参照、两阶段参照 |
| 传感网络变化与适应 | SIML的零步/少步、SC-SAMG；TrafficStream/PECPM/EAC/STBP/STRAP；STONE/STOP；ST-TTC | 固定参数、目标支持期、在线更新三个设置明确区分 |
| 条件图传播与跨节点表示 | TESTAM/GMoE、FiLM、UniSTOK指定版本、PatchSTG；克里金工作点明任务边界 | 机制参照、通用主干和内部机制控制，不自动变成所有论文都必跑 |

ST-FiT与Merlin均应进入文献地图；阶段4实际改正文仍待授权。对应算法是否比较，依据上述类别提出的研究问题、实现完整性及结果有效性统一判断。不能因为引用就要求全部跑，也不能因为没进表就从相关工作删掉。现有结果“有文件”不证明有效；已有适配差异也不自动证明无效。本轮不恢复二者数值、不据胜负选标准。

## 5. S1–S9建议修订稿

首先分三种资格：**相关工作纳入**看主题与证据；**同协议实验候选**看信息边界与忠实适配；**已有结果可复用**另核计划、代码、权重与结果来源。三个问题不能共用一组排除门槛。

| 标准 | 建议 |
|---|---|
| S1 预测任务 | 保留未来多步预测要求。克里金/插补可组成明确的预测流水线或文献已有预测适配，但要单列比较对象，不能把原插补方法名直接当完整预测方法。 |
| S2 信号边界 | 保留“保留节点动态信号及标签不得用于训练”，并覆盖预处理、归一化、预训练、蒸馏、统计记忆和辅助损失。区分真实保留节点V/T与训练K内合成遮蔽：K内的重建监督不等于用了V/T。原论文直推不等于不可适配；适配须另行核验。 |
| S3 参数更新 | 主协议推理参数固定；有零步版本的单独判断。少步微调/持续更新属于另一设置，而非从文献中排除。变化的前向隐状态不等于梯度更新，但跨窗口状态可能扩大历史预算。 |
| S4 输入预算 | **主比较保留12→12**，但按下面的信息账本判断。原论文用24/48步不等于必须用长历史；标准窗口配置可调整，若删除核心记忆/支持期/外部特征才可运行，则属实质受限适配或扩展预算。不能对UIGNN放行24→12，却仅凭48步排除SIML。 |
| S5 可复现性 | 官方实现优先；可验证的忠实重实现也可成为候选。没有定位代码是复现障碍，不是科学排除理由。最接近但暂无法复现的工作仍需讨论并标缺口。 |
| S6 出版状态 | 正式同行评审版本优先并区分主会/workshop；高相关预印本保留，注明版本和证据成熟度，不必一律禁止实验，也不能自动要求实验。 |
| S7 覆盖 | 删除“每个方向一个近期+一个经典”的机械配额。围绕直接归纳任务、缺失鲁棒预测、通用结构/时间主干三组覆盖关键主张；机制来源和纯克里金无需各凑两条。系列优先较新且忠实可复现版本，但不能把GinAR代码当GinAR+。 |
| S8 简单与流水线参照 | 保留且具体化，见下文。STID是简单神经网络，不能替代无神经网络的参照。HA不能用保留节点自己的训练期均值。 |
| S9 自适应图参照 | 至少一个Graph WaveNet或AGCRN合理，不要求两者都跑。Graph WaveNet因已有实现/历史审计可优先核查，历史结果不直接冒充最终协议结果。节点嵌入和图生成改造必须披露。 |

建议补三条：**S10适配完整性**（原模型/协议适配/只剩主干分别命名；同一原则作用于SLPF、STBP、GinAR、Merlin等全部方法）；**S11选择公平性**（相同数据与评价、验证选点、可比且披露的调参资源，非强制所有模型同LR/同阶段数）；**S12完整定义与报告**（四场景、零观测/无邻居/孤立节点均有定义，不能删困难样本；失败、限制和全部预定指标保留）。这是建议，未变更既有协议。

### S4应列的信息账本

- 推理动态值：仅当前12步合法可见交通值；拓扑、可得静态坐标、日历及观测掩码分别列项。预处理不得暗中读未来或窗外序列。
- 历史统计与参数：训练K合法数据学到的参数/统计可用；这与推理时额外读取过去一日/一周信号不同。额外交通通道、天气、POI、LLM预训练、目标支持期分别注明。
- 教师信息：Merlin教师使用K内被合成遮蔽的自然观测，现审计未发现V/T真值；应列为训练视图差异，不直接叫测试泄漏。ALFNet或其他方法的辅助重建监督也按同一原则登记。
- OS信息：ALFNet使用角色、状态进度；通用基线实际接收什么分别登记。遵守现协议，不向基线私自塞入ALFNet模块或role/age。若需证明增益来自条件化机制而非额外身份/状态信息，另建议匹配信息的机制控制，而不是假装现有方法输入完全相同。
- HD-TTS是具体例子：[官方LA配置](https://raw.githubusercontent.com/marshka/hdtts/main/config/dataset/la.yaml)为24→12；[处理代码](https://raw.githubusercontent.com/marshka/hdtts/main/experiments/run_realworld.py)在切窗前`ffill().bfill()`，仅改window=12不保证12步预算。需审查限定窗口填补、节点嵌入及下采样适配；此处是静态代码风险发现，未实际运行。

### S8建议的具体参照

建议至少**两个简单参照+一个流水线**，各自回答不同问题，先写定义再决定是否运行：

1. **Train-K季节均值**：仅用训练时间、K节点拟合分时段（可分星期）均值；对保留节点使用跨K的共享均值，不能拟合该节点自己的均值。空时间桶退回训练K总体均值。它估计仅靠日历与训练统计能达到什么水平。
2. **窗口内持续值+空间回退**：有本节点合法观测则沿用窗口内最后值；零历史则使用按冻结图距离确定的可见邻居估计；无可用邻居则退回上一参照。固定邻居定义、可见时刻和回退顺序，全部预测步与评价样本保留。不能从窗口之前找“最近读数”，不能用被遮蔽的真值。
3. **独立补全→预测**：优先评估合法的GRIN/归纳插补器加公开预测主干，分别训练并冻结插补器、杜绝未来/保留节点训练信号；同时说明与ALFNet共享补全的区别。若成本高，可先列空间KNN填补+共享时间预测器的低成本版本，但它只能回答传统流水线是否足够，不能代替强学习型插补对照。

“免训练”不等于免评估：简单参照也需计算预测/指标，仍须后续实验授权。本轮仅建议。

## 6. 实验缺口的排序应补足主张对应关系

不建议按文献出现频次直接排序，优先看它能排除哪一种替代解释：

1. **比较有效性/覆盖**：先核实近邻遗漏及已有适配完整性，再决定复用、新增、待修复；不要先增加模型数量。
2. **OS整体机制控制**：原G9路由可视化有用，但不能替代模拟评审R9的整体条件控制。须覆盖补全、FiLM、路由、末步修正等所有OS入口；历史Mask-only改变多条路径，不能只改名宣称干净对照。先整理已有证据与其限制。
3. **可重复性与选择稳健性**：多训练种子、独立节点划分是不同变异来源；现项目seed可能同时改变二者，应分别说明。节点/时间bootstrap不能替代独立训练重复；重叠滑窗也不是独立样本。不把“多种子/显著性检验”当可随意替换的一个项目。
4. **简单/流水线与自适应图参照**：按S8/S9补齐关键替代解释，具体取舍由作者/导师决定。
5. **观测预算与过程**：保留节点比例、成片缺失、角色误标/延迟；再考虑分步长、效率、路由行为、扩展数据。核验来源后再填写文献实验✓。

G5“必须重新推理”应改成条件句：若已有保存的逐样本预测包含预测步维度，可在授权的`paper/`证据中离线汇总；否则才需重新推理。当前只知道未定位逐步长指标，不能推断原始预测必然不存在。

## 7. 给Claude的执行边界与本轮记录

下一条指令见同目录 `PROMPT_FOR_CLAUDE_SURVEY_REVISION_20261103_CN.md`。它只授权修订调研、分类提纲和标准建议，保留阶段确认，不自动进入正文改写或实验。

本轮HEAD为`10cb3969b2a1bbffa62d1f93426a9764e135216f`，分支`paper/final`，本地与上游记录0/0（未fetch）。原有HANDOFF改动、survey及comparison_ten_baselines目录均保留。只新增本报告与prompt；不改原矩阵、正文、模型、配置、权重和结果，不训练、不评估、不打开测试数据、不提交/推送。

核验前原矩阵SHA256：`3A954F97061532C1BFD4A289EDB912D50405EC91A29F251620113CB941C572FC`。
