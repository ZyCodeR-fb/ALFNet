# 2 相关工作

本文从不完整观测下的重建与预测、部分可观测网络中的归纳预测，以及空间依赖与时空表示学习三个方面回顾相关研究。三个方向分别关注观测缺失、训练信号覆盖和时空信息组织，部分方法同时涉及其中多个方面。

## 2.1 不完整观测下的时空重建与预测

不完整观测会同时影响历史表示的构建和未来状态的预测。针对历史数据恢复，GRIN 通过图循环建模与空间消息传递重建缺失序列，利用关联节点补充局部不可用的信息；PriSTI 则在条件扩散框架中引入时空关联和地理关系，提高对缺失数据分布的建模能力 [1,2]。这些工作表明，观测信息可以参与表示构建和跨节点交互，而不只是作为填充值的依据。与此同时，历史恢复与未来预测具有不同的输出目标，重建能力并不直接等同于不完整输入下的预测能力。

为直接面向预测目标，GCN-M 将缺失处理与多步交通预测联合建模，结合观测掩码、多尺度历史记忆和动态图卷积，利用局部与全局信息缓解复杂缺失的影响 [3]。GinAR 进一步研究部分变量的整段历史不可用时的预测，通过插值注意力和自适应图卷积，在递归计算中恢复变量表示并调整变量间关系 [4]。因此，缺失处理与空间关系学习之间的协同已有明确研究基础。ALFNet 关注的是这一协同在节点训练信号不完整、窗口内观测支持持续变化时的具体组织方式：将可用历史的状态描述作为条件信息，贯穿历史重建、空间传播和预测上下文构建。这里的观测状态提供结构化归纳偏置，其价值不以包含完整掩码之外的新信息为前提。

## 2.2 部分可观测网络中的归纳预测

当目标位置缺少训练信号时，模型需要将已观测位置上学习到的规律迁移至未见位置。IGNNK 通过子图采样学习可迁移的消息传递，用于未采样位置的时空插值 [5]。面向未来状态，UIGNN 将预测扩展至没有历史记录的位置，并估计预测不确定性；MoGERNN 则利用图专家混合与图循环编码—解码结构，对已观测和未观测位置进行预测，并考察传感器加入与失效时的适应能力 [6,7]。这些方法已经突破了预测必须依赖每个目标节点自身历史的限制，也为变化的传感器配置提供了处理途径。

近年来，研究进一步区分了训练期与部署期的观测约束。ST-FiT 通过时间数据增强和空间拓扑学习，提高对缺少训练数据节点的预测泛化能力 [8]。SLPF 研究长期预测中的部分传感器部署，训练阶段允许利用全部位置的观测，部署阶段仅依赖部分位置的持续观测 [9]。SIML-MSIGN 在固定物理路网上将不同观测布局组织为预测任务，通过掩码门控扩散与结构归纳元学习，改善跨布局的准确性、稳定性以及零步和少步适应表现 [10]。另有支持条件化元图学习工作，从少量目标节点数据生成空间关系并进行少步参数适应 [11]。这些进展说明，动态观测与归纳预测并非相互独立的问题。ALFNet 在拓扑已知、部分节点动态信号不参与训练的设置下，研究参数固定时历史观测支持变化对预测计算的影响，重点考察同一预测目标对应的不同支持轨迹，以及支持数量相同时的时间布局差异。其研究重点是观测支持条件下的推理行为，而非新增节点上的参数更新。

## 2.3 空间依赖与时空表示学习

时空预测模型通过不同的信息交互方式刻画交通网络中的关联。STGCN 将图卷积与时间卷积结合，建立基于路网关系的时空建模框架；Graph WaveNet 和 AGCRN 利用可学习节点表示构造自适应空间关系，减少对预设邻接的依赖 [12–14]。这里的“自适应”并不必然意味着传播关系随每个输入窗口变化。面向更灵活的交互，PDFormer 在长程注意力中建模空间依赖与传播时延，PatchSTG 则通过空间分块以及块内、块间交互，提高大规模时空表示的效率 [15,16]。这些方法从空间结构、交互范围和表示粒度等方面扩展了交通预测模型的能力。

空间建模的价值还需要与其他表示能力共同考察。STID 表明，具有时空身份信息的简洁模型也能形成有竞争力的预测基线；AdaFre 根据时变频率信息选择预测路径，并为不同频率配置相应的空间表示 [17,18]。STBP 则结合频域表示、线性图注意力和可扩展上下文模式库，面向节点扩展与分布变化开展持续预测 [19]。这些研究关注的条件包括交通状态、频率结构和部署任务，提示有效的信息组织不局限于固定图传播。ALFNet 进一步将当前历史观测支持作为组织时空计算的条件：在重建历史的基础上调整空间推理，并将跨节点上下文用于时间表示与预测调制。其方法定位在于这些计算如何围绕观测支持协同，而不在于重新提出图循环、空间分块或特征调制等基础算子。

## 参考文献

以下使用临时顺序编号；合并论文时统一转换为文献管理条目。

1. Cini et al. **Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks.** ICLR, 2022. [论文](https://openreview.net/forum?id=kOu3-S3wJ7)。
2. Liu et al. **PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation.** ICDE, 2023. [作者论文](https://arxiv.org/abs/2302.09746)，[官方实现与发表说明](https://github.com/LMZZML/PriSTI)。
3. Zuo et al. **Graph Convolutional Networks for Traffic Forecasting with Missing Values.** Data Mining and Knowledge Discovery, 37:913–947, 2023. [官方实现与正式引文](https://github.com/JingweiZuo/GCN-M)。
4. Yu et al. **GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing.** KDD, 2024. [作者全文](https://finleywang.github.io/files/2024_KDD_GinAR.pdf)，[官方代码](https://github.com/GestaltCogTeam/GinAR)。
5. Wu et al. **Inductive Graph Neural Networks for Spatiotemporal Kriging.** AAAI, 2021. [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/16575)。
6. Mei et al. **Uncertainty-aware Traffic Prediction under Missing Data.** ICDM, 2023. [正式论文](https://ieeexplore.ieee.org/document/10415744/)，[作者全文](https://arxiv.org/html/2309.06800)。
7. Zhou et al. **MoGERNN: An Inductive Traffic Predictor for Unobserved Locations.** Transportation Research Part C, 2025. [论文](https://doi.org/10.1016/j.trc.2025.105080)，[官方代码](https://github.com/ZJU-TSELab/MoGERNN)。
8. Lei et al. **ST-FiT: Inductive Spatial-Temporal Forecasting with Limited Training Data.** AAAI, 2025. [正式全文](https://ojs.aaai.org/index.php/AAAI/article/download/33310/35465)，[官方代码](https://github.com/LzyFischer/InductiveST)。
9. **Spatio-temporal Partial Sensing Forecast of Long-term Traffic.** Transactions on Machine Learning Research, 2025. [论文](https://openreview.net/forum?id=Ff08aPjVjD)，[官方代码](https://github.com/zbliu98/SLPF)。
10. Zhao et al. **Structure-inductive Meta-learning Empowers Partial-to-global Traffic State Forecasting in Dynamic Sensor Networks.** Transportation Research Part C, 2026. [正式论文](https://www.sciencedirect.com/science/article/abs/pii/S0968090X26000525)。
11. Wang et al. **Cold-Start Traffic State Forecasting at Unseen Sensor Locations via Support-Conditioned Meta-Graph Learning.** Sensors, 2026. [正式论文](https://www.mdpi.com/1424-8220/26/15/4995)。
12. Yu et al. **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.** IJCAI, 2018. [作者论文](https://arxiv.org/abs/1709.04875)。
13. Wu et al. **Graph WaveNet for Deep Spatial-Temporal Graph Modeling.** IJCAI, 2019. [正式论文](https://www.ijcai.org/proceedings/2019/264)。
14. Bai et al. **Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting.** NeurIPS, 2020. [官方实现与论文](https://github.com/LeiBAI/AGCRN)。
15. Jiang et al. **PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction.** AAAI, 2023. [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/25556)。
16. Fang et al. **Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective.** KDD, 2025. [作者全文](https://arxiv.org/html/2412.09972v1)，[官方代码](https://github.com/LMissher/PatchSTG)。
17. Shao et al. **Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting.** CIKM, 2022. [官方实现与论文](https://github.com/GestaltCogTeam/STID)。
18. Qin et al. **Adaptive Frequency Pathways for Spatiotemporal Forecasting.** AAAI, 2026. [正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38595)。
19. **A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting.** ICLR, 2026（作者官方仓库标注）. [论文页面](https://openreview.net/forum?id=LHSea6DI8U)，[官方代码](https://github.com/Aoyu-Liu/STBP)。
