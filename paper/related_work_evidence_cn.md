# Related Work 文献筛选与证据说明

日期：2026-09-14。用途：作者核对与后续基线适配，不并入 Related Work 正文。

## 检索范围与证据等级

本轮围绕固定的 observation-support-aware inductive forecasting 主线，使用 Sciverse 的 semantic_search、search_papers 与 read_content，并以作者全文、正式出版页面及官方代码纠正元数据。筛选涉及下表 28 项相关工作；这不是 28 篇全文精读，也不是穷尽式系统综述。没有把泛领域误召回计为相关论文，未因未检索到而声称不存在相关研究。

- **F**：读取了原论文相关全文段落（本地参考论文、Sciverse read_content 或作者/出版方全文）；不等于逐页读完整篇。
- **S**：Sciverse 返回的正文片段，尚不足以确认全文中的所有实验设定。
- **A**：原始摘要或出版记录；只支持问题与主要方法的概括。
- **R**：官方代码/README 或本地代码核对；支持实现来源，不自动支持论文性能结论。

Sciverse 查询和命中文献标识保存在 [related_work_search_20260914.json](related_work_search_20260914.json)。正式论文的年份以出版版本为准，而非 Sciverse 中的 arXiv 年份。检索返回的 `total_count=10000` 可能是宽松召回，不能当作精准相关文献数量。

## 文献筛选表

| 工作 | 年份 / 发表状态 | 证据 | 核心能力与边界 | 本文用途与决定 | 可核对来源 |
|---|---|---|---|---|---|
| GRIN | ICLR 2022 | S+R | 图循环缺失重建；原任务不是直接多步未来预测 | A 类基础文献、重建思想来源；不直接将插值分数列入预测表 | [论文](https://openreview.net/forum?id=kOu3-S3wJ7)，本地 `references/grin/` |
| PriSTI | ICDE 2023 | A+R | 条件扩散时空插值，已使用全局上下文与地理条件 | A 类代表引用；若做预测需明确额外预测器，不作为默认新增训练 | [官方代码](https://github.com/LMZZML/PriSTI) |
| GCN-M | DMKD 2023；预印本 2022 | F+R | 输入明确包括 X、完整 mask 序列和图；多尺度记忆与动态图联合预测 | A 类优先新增；不能声称已有缺失方法只做填充或不调整空间关系 | [官方代码](https://github.com/JingweiZuo/GCN-M)，Sciverse 正文 §3–4 |
| GinAR | KDD 2024 | F+R | 整变量历史缺失，插值注意力与自适应图递归预测 | A 类最优先新增；需要核对训练节点隔离与变量嵌入 | [作者全文](https://finleywang.github.io/files/2024_KDD_GinAR.pdf)，[官方代码](https://github.com/GestaltCogTeam/GinAR) |
| GMN / SGMN | TRC 2020 | S+R | 图 Markov 过程联合缺失状态推断与未来预测 | 早期联合预测脉络；为控制新增数量不优先于 GCN-M/GinAR | [官方代码](https://github.com/zhiyongc/Graph-Markov-Network) |
| RT-GCN | Information Fusion 2024 | S | 高斯表示与鲁棒预测，涉及缺失和噪声 | 有关鲁棒性方向的候选；代码未在本轮核验，不作为默认新增 | [作者机构全文入口](https://research.tue.nl/files/309332629/1-s2.0-S1566253523003949-main.pdf) |
| IBN | arXiv 2025；本轮未核实正式录用 | S+A+R | 缺失变量预测、重建不确定性和高斯核图建模 | 新近跟踪项；不因更新而替代已发表 GinAR | [作者论文](https://arxiv.org/abs/2509.07725)，[官方代码](https://github.com/zhangth1211/NICLab-IBN) |
| IGNNK | AAAI 2021 | S+A | 通过随机子图学习未采样位置插值 | B 类发展脉络；不把同时间恢复当未来预测 | [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/16575) |
| INCREASE | WWW 2023 | S+A | 异质空间关系与时间模式的归纳插值 | 支持“归纳插值已有成熟研究”；正文篇幅不足时与 IGNNK 合并引用 | [作者论文页](https://xiaoliangfan.github.io/publication/publication/www2023/) |
| FUNS | 作者预印本 2022 | S+A | 对完全未观测位置预测未来状态 | B 类早期近邻；不能宣称未见位置未来预测由本文提出 | [作者论文](https://arxiv.org/abs/2211.11596) |
| UIGNN | ICDM 2023，short paper | F+A+R | 无历史位置未来预测与不确定性；训练不使用缺失位置目标 | B 类直接近邻，P1 新增；原 24 步输入与预测时距需要声明适配 | [IEEE](https://ieeexplore.ieee.org/document/10415744/)，[全文](https://arxiv.org/html/2309.06800)，[代码](https://github.com/lijunxian111/UIGNN) |
| MoGERNN | TRC 2025 | F+R | MoGE 稀疏路由与图编码—解码；含加入/失效与可用传感器研究 | 必须比较；禁止将其说成仅适用静态传感器集合 | [论文](https://doi.org/10.1016/j.trc.2025.105080)，本地 `references/paper/MoGERNN.md` |
| ST-FiT | AAAI 2025 | F+R | 缺少节点训练数据；时间合成和拓扑学习，推理不需微调 | 已适配，优先复用；不是 missing-aware 类代表 | [AAAI 全文](https://ojs.aaai.org/index.php/AAAI/article/download/33310/35465)，[作者全文](https://arxiv.org/html/2412.10912v1) |
| SLPF | TMLR 2025 | F+R | 训练可访问全部位置，部署固定部分感知，长期预测 | 保留受限训练信号适配；不能代表原任务完整复现 | [论文](https://openreview.net/forum?id=Ff08aPjVjD)，[代码](https://github.com/zbliu98/SLPF) |
| SIML-MSIGN | TRC 2026 | F+A | 固定物理图上的 mask-defined PG-TSF；掩码门控扩散、跨任务稳定性、零步与少步适应 | 必须作为最近邻讨论；官方代码未核实，是实验覆盖待解决项 | [出版页](https://www.sciencedirect.com/science/article/abs/pii/S0968090X26000525)，本地 `references/paper/SIML-MSIGN.md` |
| SC-SAMG | Sensors 2026 | A+出版页方法段 | 少量目标支持数据生成节点/图表示，少步参数适应 | 引用以避免将 support-conditioned 命名当创新；非优先训练基线，代码未核实 | [原文](https://www.mdpi.com/1424-8220/26/15/4995) |
| DGAE | 作者更新稿 2026；Sciverse 另标 TITS 2025，年份未裁定 | S | 稀疏传感器下的网络状态估计，Dirichlet 图正则 | 原任务与未来预测不同，不作为直接基线；不采用不确定年份 | [作者论文](https://arxiv.org/abs/2503.15845) |
| STGCN | IJCAI 2018 | A+R | 路网图卷积与时间卷积 | 已完成，保留固定图参照 | [作者论文](https://arxiv.org/abs/1709.04875)，本地适配审计 |
| Graph WaveNet | IJCAI 2019 | A+R | 扩张时间卷积与结构自适应邻接 | 已有强结果，必须纳入候选；自适应邻接不等同于逐输入动态邻接 | [正式论文](https://www.ijcai.org/proceedings/2019/264)，本地 `references/GWNet/` |
| AGCRN | NeurIPS 2020 | A+R | 节点自适应图卷积循环建模 | 相关工作经典引用；不为凑数量优先重做已有 GWNET 的相近比较 | [官方代码](https://github.com/LeiBAI/AGCRN) |
| STID | CIKM 2022 | A+R | 时空身份嵌入与简洁预测器 | 已完成并保留，作为非显式邻居聚合参照 | [官方代码](https://github.com/GestaltCogTeam/STID) |
| PDFormer | AAAI 2023 | A | 传播时延感知与动态长程交互 | 正文表示学习脉络；不同时追加多个 Transformer，优先复用 PatchSTG | [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/25556) |
| STAEformer | CIKM 2023 | A+R | 自适应时空嵌入增强 Transformer | 备用新增，优先级低于补齐缺失预测类 | [官方代码](https://github.com/XDZhelheim/STAEformer) |
| PatchSTG | KDD 2025 | F+S+R | 空间分块与块内/块间表示，面向大规模预测 | 已适配，优先复用；补充 ALFTemporal 来源关系的外部参照 | [作者全文](https://arxiv.org/html/2412.09972v1)，[官方代码](https://github.com/LMissher/PatchSTG) |
| STBP | ICLR 2026，作者仓库标注 | R | 频域/线性图注意力主干加可扩展模式库，持续训练 | 保留离线适配；不能将主干比较等同于持续学习算法比较 | [官方代码](https://github.com/Aoyu-Liu/STBP) |
| AdaFre | AAAI 2026 | F+A | 频率路径选择与频率对应的空间表示 | 增补近期相关工作；本轮未核验到官方代码，暂不排入训练 | [正式全文](https://ojs.aaai.org/index.php/AAAI/article/download/38595/42557) |
| STGformer | 作者预印本 2024；正式会场未核实 | S+A+R | 高效时空图 Transformer，主要面向 LargeST 规模 | 跟踪候选，已有 PatchSTG 时优先级低 | [作者论文](https://arxiv.org/abs/2410.00385)，[官方代码](https://github.com/Dreamzz5/STGformer) |
| STGODE | KDD 2021 | R | 图微分方程时空预测；当前适配改变语义图构造 | 已完成结果全部保留，正文优先度低于直接相关近邻 | [官方代码](https://github.com/square-coder/STGODE)，本地适配审计 |

## 会改变论文表述的关键证据

| 原本可能使用的表述 | 反向证据 | 本轮采用的表述 |
|---|---|---|
| 现有方法忽略 missingness | GCN-M 的问题定义直接输入完整 mask 序列 | 不完整观测已被建模，进一步考察训练信号限制下支持轨迹的条件化利用 |
| 已有缺失方法只先插值再预测 | GinAR 与 GCN-M 均联合建模 | 明确区分恢复任务、流水线和联合未来预测 |
| 现有归纳方法不考虑传感器变化 | MoGERNN 有加入/失效；SIML-MSIGN 研究变化布局 | 不以传感器生命周期本身作为新问题声明 |
| 加入 support 变量就得到新问题 | mask 序列已存在；SC-SAMG 也用 support-conditioned 命名 | 将差异落实到变量含义、监督可见范围、参数更新和前向计算 |
| SIML 必须微调，所以不必比较 | 原论文同时报告零步与少步适应 | MSIGN 与 SIML 零步都是需认真核对的比较对象 |
| 所有 adaptive graph 都逐时变化 | GWNET/AGCRN 的可学习节点关系可在推理时保持固定 | 区分可学习结构和输入条件化传播 |
| 比三个代表更好即可说明超过三个领域 | 代表只是有限候选，输入/适配与局部排序均有限制 | 报告指定协议下与代表方法的条件性比较，解释结合内部消融 |

## 来源质量与检索局限

Sciverse 对 ST-FiT、GCN-M、SLPF 提供了可读取正文，对 SIML 的结构化搜索主要提供出版元数据，后者的论证来自本地全文与出版方页面。GinAR 的精准标题搜索本轮返回空，语义检索主要返回后继 IBN；因此改用作者全文和官方仓库，并未用空结果否定其存在。UIGNN 的 Sciverse 会场字段误扩展为 Industrial Conference on Data Mining，GitHub About 又标 CIKM；IEEE 正式记录明确是 IEEE ICDM，已纠正。上述情况说明 Sciverse 是检索入口，出版年份、会场与算法细节仍需要原始来源交叉核对。

检索到的新近方法不自动进入主表：IBN 和 STGformer 的正式发表状态未核实，AdaFre 的官方代码未核验；纯插值或状态估计方法需要额外预测头，会改变比较对象。相反，SIML-MSIGN 和 SC-SAMG 与研究概念接近，即使短期不适配，也不能因代码可得性或会场偏好在 Related Work 中回避。

## 本地证据入口

- `docs/comparison_model_audit.md`：已注册模型、原始来源与公平适配。
- `paper/experiments_draft_cn.md`、`paper/experiments_appendix_cn.md`：当前正文表与条件/配对分析。
- `docs/ai/handoffs/T-20260908-support-ablations/METRLA_WAVE_RESULTS_20260913.md`：METR-LA 新 GWNET 参考。
- `docs/ai/handoffs/T-20260908-support-ablations/SPEED_AND_GWNET_20260913.md`：PEMS-BAY corrected GWNET 历史参考与版本差异。
- `outputs_v4/final_comparisons/`：已发现 PEMS-BAY STFiT 与 PatchSTG 历史 validation 表；当前可复用性尚未确认。原配置不能仅因文件存在就直接成为新的实验权威。

以上均只读参考，本轮未修改 AI 管理文档、模型、数据协议或实验结果。
