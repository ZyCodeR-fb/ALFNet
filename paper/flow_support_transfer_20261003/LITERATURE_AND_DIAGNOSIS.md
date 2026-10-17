# 文献与可检验机制

本轮使用Academic Research Suite本地设计流程、Sciverse两次定向语义检索以及作者代码/出版社/会议/arXiv一手页面。不是穷尽式系统综述。下列区分已见来源与本项目推断，不把论文结果当成本项目收益保证。

|工作及来源|核验层级|相关机制与本轮取舍|
|---|---|---|
|[NLinear / AAAI2023](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)|作者完整短源码及仓库说明|减最后值再加回；本轮借鉴为可见历史中心化，保留ALFNet主干，不复制其线性模型。零历史必须特殊处理。|
|[RevIN / ICLR2022](https://github.com/ts-kim/RevIN)|作者仓库说明；单文件访问超时|可逆归一化应对分布变化；本轮不计算无历史节点的均值方差，不宣称完整RevIN。|
|[Dish-TS / AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25914)|出版摘要及PDF检索片段|历史与预测分布变化值得区分；不直接加入新分布预测网络。|
|[DCRNN / ICLR2018](https://arxiv.org/abs/1707.01926)|作者摘要|递归预测及scheduled sampling背景；当前残差倍率是本项目设计，delta1不是论文推荐的最优值。|
|[BRITS / NeurIPS2018](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)|会议摘要|双向时间信息可用于历史填补；窗口内插值是轻量控制，不称为BRITS复现。|
|[GRIN / ICLR2022](https://github.com/Graph-Machine-Learning-Group/grin)|作者仓库说明，前轮已核验|空间时间重建；保留零历史GRIN，仅检测有历史节点是否需要更强自身尺度约束。|
|[SAITS / ESWA2023](https://arxiv.org/abs/2202.08516)|作者摘要及官方仓库书目信息|缺失感知时序重建；不直接大规模替换imputer，先以无新参数控制验证问题位置。|
|[ST-FiT / AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/33310)|会议摘要|增强与拓扑学习促进训练数据稀缺节点迁移；不是本轮引入其代码的授权或必要性。|
|[GinAR / KDD2024](https://arxiv.org/abs/2405.11333)|前轮作者摘要及代码核验|插值与自适应图联合建模；新桥接仅是自身观测插值，不称为GinAR模块。|
|[On the Role of RevIN / 2026preprint](https://arxiv.org/abs/2603.11869)|作者摘要检索|归一化组件未必都有益，作为反证背景，不预设中心化一定提高结果。|

## 本地证据

详见HISTORICAL_RESULTS.md及historical_diagnosis.json；保留既往超参数、unitFocal族、重建/degree/relative20的04/08结果。04test的Cold MAE43.90对最优STBP32.51；08为43.60对24.76。相比之下，04Unseen已领先，08Unseen仍落后UIGNN。这意味着只盯低流量MAPE无法解释全部胜格缺口。

源码明确：共享GRIN填补结果同时进入图与时间专家；残差解码从最后填补值开始，之后每步加.3倍投影；训练decoder teacher在掩码下启用。少量观测时历史中大部分时间位置仍由GRIN决定。推断：局部自身尺度被空间填补稀释、节点间绝对尺度差异、递归增量优化偏置可能影响Cold/Warm；本轮以桥接/中心化/倍率分别检验，尚无因果结论。

不用节点全历史均值或test分箱拟合，不引入额外GPS/OSM、不同评价阈值、隐藏标签、测试集后处理校准。比较算法不变；bridge把过去输入窗口的后侧观测用于前侧缺口，所有时间均早于预测起点，不是读取预测未来。
