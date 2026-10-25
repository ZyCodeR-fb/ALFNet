# ALFNet：动态观测支持下的归纳时空预测

<!-- v6 中文工作稿（2026-10-20）。基于 v5（manuscript_draft_cn-2-v5.md，保留不动）：第1、2、4节只做术语统一与对应矩阵列出的文字修正，数值未改；第3节方法、摘要、第5节结论为新写；原4.9小结已并入第5节。每处改动的依据见 ../ALIGNMENT_MATRIX_CN.md 与 ../notes/V6_CHANGELOG_CN.md。 -->

## 摘要

交通预测系统能够获得的历史观测往往不完整，而且在不断变化：新部署的传感器逐步积累读数，在役传感器可能中断，部分位置在训练期没有任何动态信号。本文研究路网拓扑已知、部分节点不参与训练时的动态观测支持下的归纳时空预测，将窗口内可用观测的位置、数量与时间分布（观测支持）既视为输入有效性的描述，也视为组织预测计算的条件。为此我们提出 ALFNet。它以逐步的观测角色、状态年龄和观测掩码构成生命周期上下文；共享的图循环补全先为两个分支提供同一份历史；图分支依据观测状态对五种传播算子加权组合，并以残差自回归方式解码；时间分支从空间分区上下文生成条件特征，通过 FiLM 调制图编码状态；输出端再施加由末步角色控制幅度的有界图修正。在 METR-LA、PEMS-BAY、PEMS04 和 PEMS08 上，我们以零历史（Unseen）、持续观测（Steady）、逐步加入（Join）和历史中断（Fault）四类条件统一评价。ALFNet 的跨条件平均误差（LifeAvg）三项指标在四个数据集上均优于八个比较方法，相对各数据集最强基线的 MAE 降低7.86%–15.63%；逐步加入（Join）与历史中断（Fault）两类主条件的三项指标在四个数据集上均为最优。组件消融支持观测状态条件化传播与跨节点时间上下文的作用。零历史（Unseen）条件下的优势依赖数据集，在两个流量数据集上未能成立。

## 1 引言

交通预测为路网运行管理、出行规划与拥堵预警提供前瞻性信息。然而，预测系统能够获得的历史数据并不总是连续且完整：传感器逐步部署使部分位置缺少训练期记录，设备失效和通信异常又会改变运行期间的数据可用性。已有关于未观测位置预测和动态传感器网络的研究已认识到这些问题 [6,7,9]。对于同一个预测位置，新近加入、持续观测和近期中断所提供的信息条件并不相同。本文将历史窗口内实际可用观测的位置、数量及其时间分布统称为**观测支持**。观测支持的变化不仅产生需要处理的空缺，也决定模型能从本地历史与关联位置获得怎样的预测依据。

围绕不完整交通观测，已有研究形成了互补的解决途径。缺失感知方法通过图循环重建、条件扩散或联合重建与预测，利用时空关联补充不可用信息 [1–4]；归纳与部分感知预测将预测范围扩展至缺少训练信号或部署期持续观测的位置 [6–9]；自适应图学习及空间分块表示则改善空间关系和多尺度上下文的组织 [10–14]。这些研究已经证明，缺失处理、空间信息传递与未观测位置预测能够协同工作。因而，进一步的关键并非判断模型是否能够接受缺失输入，而是明确：当历史支持在预测窗口内变化时，模型应如何相应调整信息利用方式。

这一问题在归纳预测中尤其突出。目标节点没有参与动态信号训练，并不意味着它在推理时始终没有观测；同样，训练期已有记录也不保证部署时始终拥有完整历史。现有方法分别强调训练信号覆盖、缺失恢复、传感器配置适应或空间交互，但这些能力并不直接回答"同一目标对应不同历史支持时，预测计算应如何变化"。即使可用观测数相同，近期集中、早期集中和分散分布的历史也可能对未来状态提供不同依据。本文因此研究**动态观测支持下的归纳时空预测**：在路网拓扑已知、部分节点动态信号不参与训练的条件下，利用变化的历史观测支持组织空间推理与预测上下文。该设置连接节点归纳与窗口内支持变化，强调固定模型参数下的预测行为。

我们的核心认识是，**观测支持应同时作为输入有效性的描述和预测计算的条件**。观测掩码标明哪些读数可用；在此基础上，观测角色与状态年龄可以为模型组织历史信息提供显式的结构化线索。这里的角色表示活跃（AAS）、新接入（NAS）、未观测（VS）或失效（FS）等状态，状态年龄描述接入或中断过程的进展。它们的作用在于提供一种可检验的归纳偏置，而不要求包含完整掩码之外的额外信息。沿这一思路，历史补全负责提供可用表示，空间传播决定如何交换关联信息，跨节点时间上下文则为预测过程提供条件。三者的协作，使模型能够在本地证据与空间证据之间建立依赖当前支持条件的信息利用方式。

据此，我们提出观测支持感知预测框架 ALFNet。**观测支持感知历史补全（SAHC）**借鉴图循环插补思想 [1]，利用可见观测及图关联构建两个分支共享的历史输入。**观测条件化图预测（ALFGraph）**借鉴图专家混合与扩散卷积循环网络（DCRNN）的循环结构 [7,18]，由观测状态决定传播算子的组合。**空间上下文时间条件分支（ALFTemporal）**借鉴空间分区交互 [14] 与两阶段注意力 [20]，从跨节点历史生成条件表示，并通过特征级线性调制（FiLM）[16]作用于图循环预测过程。预测输出再经过一个由末步角色控制幅度的图修正。图预测与时间分支围绕共同的历史支持协作，最终形成单一预测输出。本文的重点是这一条件化组织方式，而非重新提出图循环、分块注意力或线性调制等基础算子。

本文的贡献概括为三点：（1）围绕拓扑已知、训练信号覆盖不完整的路网，将窗口内变化的观测支持纳入归纳预测设置，并以零历史（Unseen）、持续观测（Steady）、逐步加入（Join）和历史中断（Fault）条件刻画其预测要求；（2）提出 ALFNet，将观测支持感知历史补全（SAHC）、观测条件化图预测（ALFGraph）与空间上下文时间条件分支（ALFTemporal）组织为协同预测框架；（3）在 METR-LA、PEMS-BAY、PEMS04 与 PEMS08 四个数据集上开展统一条件评价，并结合组件消融、配对观测干预和样本层面分析，考察总体收益、机制归因及支持结构对预测行为的影响。总体结果显示，ALFNet 在四个数据集的跨条件平均误差（LifeAvg）三项指标上均优于全部比较方法；在逐步加入（Join）与历史中断（Fault）两类主条件下，这一优势在四个数据集上保持一致；组件消融支持观测状态条件化传播与跨节点时间上下文在完整框架中的作用；配对干预与布局分析表明，这种收益与模型对支持条件变化的响应相关，这一响应是在预测目标与非干预输入保持不变时观察到的。同时，在本地历史完全缺失的零历史（Unseen）条件下，优势并不在所有数据集上成立，我们在实验中对该边界予以说明。

## 2 相关工作

### 2.1 不完整观测下的重建与预测

不完整观测建模包含历史恢复与未来预测两个相关但不同的目标。GRIN 通过图循环计算和消息传递重建多变量序列，PriSTI 则利用条件扩散和时空关联建模缺失数据 [1,2]。二者说明，关联位置能够为局部缺失提供信息来源；但重建历史的目标与预测未来并不相同，插补质量也不能直接代替预测性能评价。

面向未来预测，GCN-M 联合建模缺失处理与交通预测，通过观测掩码、多尺度历史记忆和图卷积利用局部与全局信息 [3]。GinAR 针对部分变量历史不可用的情况，将插值注意力和自适应图卷积引入递归预测 [4]。UIGNN 同时面向缺失位置预测与不确定性估计，与未观测位置泛化形成交叉 [6]。这些工作已经建立了缺失处理和空间推理的联系，并以掩码区分可见与缺失读数。本文在此基础上关注历史支持的时间结构：可见读数数量相同但位于窗口内不同位置时，预测所依据的信息并不相同；重建之外，观测状态如何参与传播组合，跨节点上下文如何进一步条件化未来预测。相应地，实验分别干预观测状态、观测状态条件化传播与跨节点上下文，以区分各机制的贡献，并在固定可见数量下比较不同时间布局。

### 2.2 归纳与部分感知预测

归纳建模研究如何将已观测位置上学习的规律用于缺少训练信号的位置。IGNNK 通过子图采样学习可迁移的消息传递，解决未采样位置的时空插值 [5]；UIGNN 和 MoGERNN 则进一步预测未来交通状态 [6,7]。其中，MoGERNN 采用图专家混合与图循环编码—解码结构，并考察传感器加入和失效情形，说明变化的传感器配置已是归纳交通预测的重要研究对象。

不同方法对训练和推理阶段的观测条件有不同假设。SLPF 研究长期部分感知预测，利用训练期较充分的数据学习向部署期未感知位置的空间转移 [8]。SIML-MSIGN 将不同观测布局组织为任务，通过掩码门控扩散与结构归纳元学习开展跨布局预测及适应 [9]。SC-SAMG 利用少量目标位置数据生成节点与图表示，并进行少步参数适应 [24]；其中的"支持"指用于适应的目标样本，与本文窗口内的观测支持含义不同。因此，本文并不将"变化观测下的预测"视为未经研究的领域。ALFNet 的侧重点是固定参数下的窗口级支持条件化：在统一输入预算内，研究可用历史的状态和布局如何影响空间—时间计算。总体评价考察这一设计在归纳设置中的效果，已知节点配对实验则用于分离支持变化本身带来的预测响应。

### 2.3 空间依赖与时空表示学习

空间依赖建模为交通预测提供了利用关联位置的基本途径。STGCN 结合图卷积与时间卷积；Graph WaveNet 和 AGCRN 利用可学习表示构建自适应关系，降低对固定邻接的依赖 [10–12]。PDFormer 将传播时延纳入长程空间交互，PatchSTG 通过空间分块及块内、块间交互组织局部与全局信息 [13,14]。这些方法分别改善空间关系、交互范围和计算粒度，但学习到自适应图，并不自动意味着传播会根据每个窗口的观测支持进行调整。空间交互还需与时间表示能力共同考察。STID 通过简洁的时空身份表示建立有竞争力的预测基线 [15]；STBP 利用可扩展上下文模式库支持城市持续预测 [17]。本文关注的不是无限扩大空间交互范围或在线更新模式库，而是如何将可用的跨节点历史用于当前预测条件化。ALFTemporal 生成的上下文经 FiLM [16] 调制图循环预测，ALFGraph 则承担观测条件下的空间传播。实验中的同容量局部上下文替代直接检验跨节点信息的作用，避免将模型收益仅归结为更大的表示规模。

## 3 方法

本节先形式化观测支持（3.1），再给出 ALFNet 的整体数据流（3.2），随后按数据流依次介绍观测支持感知历史补全（SAHC，3.3）、观测条件化图预测（ALFGraph，3.4）、空间上下文时间条件分支（ALFTemporal）与 FiLM 调制（3.5）和角色缩放图修正（3.6），最后说明训练目标（3.7）与数据集特定设置（3.8）。

### 3.1 问题设置与观测支持

路网表示为静态图 $\mathcal{G}=(\mathcal{V},\mathbf{A})$，其中 $|\mathcal{V}|=N$，$\mathbf{A}\in\mathbb{R}^{N\times N}$ 为加权邻接矩阵；全部节点的拓扑及可用静态属性在各阶段均可见。节点划分为已知节点 $\mathcal{V}_K$ 以及验证、测试保留节点 $\mathcal{V}_V$、$\mathcal{V}_T$：训练只读取 $\mathcal{V}_K$ 的动态信号，验证与测试分别加入 $\mathcal{V}_V$ 与 $\mathcal{V}_T$ 的历史；标准化统计量仅由训练时段的 $\mathcal{V}_K$ 计算。给定历史长度 $L=12$ 与预测长度 $H=12$，记标准化后的可见历史为 $\mathbf{X}\in\mathbb{R}^{L\times N}$，观测掩码为 $\mathbf{M}\in\{0,1\}^{L\times N}$。$M_{t,n}=1$ 当且仅当该读数自然有效、节点处于当前阶段的信号暴露范围内，且其生命周期状态允许读取；不可用位置的输入值置零，并由 $\mathbf{M}$ 标识。

除掩码外，观测支持还包括两类显式描述。生命周期角色 $\mathbf{R}_t\in\{0,1\}^{N\times4}$ 为四类 one-hot 编码：活跃（AAS）、新接入（NAS）、未观测（VS）与失效（FS）；状态年龄 $\mathbf{a}_t\in[0,1]^{N}$ 描述接入或中断已持续的比例。三者逐步拼接为生命周期上下文

$$
\mathbf{S}_t=[\mathbf{R}_t,\ \mathbf{a}_t,\ \mathbf{M}_t]\in\mathbb{R}^{N\times 6},\qquad t=1,\dots,L. \tag{1}
$$

四种观测场景对应不同的角色轨迹。零历史（Unseen）在整个窗口内为 VS，年龄为0；持续观测（Steady）在整个窗口内为 AAS，年龄为1；逐步加入（Join）在接入前为 VS（年龄0），接入后的最后 $m$ 步为 NAS，年龄依次为 $1/L,\dots,m/L$；历史中断（Fault）在中断前为 AAS（年龄1），中断后的最后 $m$ 步为 FS，年龄同样从 $1/L$ 递增至 $m/L$。自然缺失只改变 $\mathbf{M}$，不改变角色，因此活跃节点的偶发缺测仍记为 AAS。训练阶段四种场景均在已知节点上模拟；评价时，Unseen 与 Join 在对应阶段的保留节点上构造，Steady 与 Fault 在已知节点上构造。窗口内可见读数数 $q_n=\sum_t M_{t,n}$ 只用于描述支持的多少，不作为额外输入。

预测任务是在固定模型参数下学习映射 $f:(\mathbf{X}\odot\mathbf{M},\ \mathbf{S}_{1:L},\ \mathbf{A},\ \text{日历})\mapsto\hat{\mathbf{Y}}\in\mathbb{R}^{H\times N}$。同一节点在不同窗口中可以处于不同角色与布局，因此模型应依据当前窗口的支持条件组织计算，而不是依据节点是否参与过训练。

### 3.2 总体结构

![ALFNet 架构](../imgs/alfnet.png)

<!-- 架构图为作者 PPT 复现版（imgs/alfnet.png），符号与箭头待作者在文字定稿后统一修改；图中 Restore 之后的 "Y₀" 即正文的 Ỹ。英文图注以 nanobanana_v4_20261019/README_CN.md 的版本为基础，阶段3写入 LaTeX。 -->

**图1　动态观测支持下的 ALFNet 架构。** 共享的 GRIN/BiGRIL 模块对全部节点重建完整的12步历史，前向与后向递推对齐后合并；观测值保持不变，缺失位置被补全，并按数据集施加窗口插值（仅对至少有一个可见读数、$q>0$ 的节点）与参考中心化（$\bar{\mathbf{X}}=\mathbf{X}^{\mathrm{fill}}-\alpha\mathbf{v}$，$\mathbf{v}$ 为最后一次可见读数，无观测时为0），得到两个分支共享的历史 $\bar{\mathbf{X}}$。ALFTemporal 以步长为12的卷积将每个节点的窗口压缩为一个64维时间 patch，拼接日内时刻、星期与共享结构嵌入（160维），将节点组织为 $P=32$ 个各含 $C$ 个槽位的空间分区，经一个两阶段注意力块——同一槽位的跨分区注意力，以及经4个潜在 token 的分区内汇聚与回传——后恢复节点顺序。FiLM 参数头将所得 $\mathbf{Z}\in\mathbb{R}^{N\times160}$ 映射为 $(\Delta\boldsymbol{\gamma},\boldsymbol{\beta})$，在解码前一次性调制两层图编码器的末状态。ALFGraph 使用生命周期条件化的 DCRNN 单元，其各门以基于生命周期状态的 softmax 路由稠密组合五种传播支持；两层残差自回归解码器以 $\bar{\mathbf{X}}_L$ 为初值，并回送未经修正的逐步预测。加回 $\alpha\mathbf{v}$ 后，角色缩放图修正施加由末步角色 $\mathbf{R}_L$ 缩放的有界邻域残差，得到唯一的12步预测 $\hat{\mathbf{Y}}$。Unseen、Steady、Join 与 Fault 表示观测场景，由随时间变化的角色（AAS/NAS/VS/FS）、状态年龄与观测掩码实现。\*数据集特定处理：METR-LA/PEMS-BAY/PEMS04/PEMS08 的 $C$=7/16/10/6，插值为关/关/开/开，解码残差倍率 $d$=0.3/0.3/1/1，$\alpha$=0/0/0/0.5。每层每个门各有独立路由；WM 与 Fwd 的传播形式相同但参数独立；每个注意力阶段均含残差、归一化与前馈层。图中省略批量维度，曲线与矩阵元素为示意。

ALFNet（Adaptive Lifecycle Forecasting Network）的"自适应"指计算随输入的观测支持而变化，推理时模型参数保持固定。如图1所示，观测支持感知历史补全（Support-Aware History Completion，SAHC）首先为全部节点生成共享历史 $\bar{\mathbf{X}}$。随后两个分支读取同一份 $\bar{\mathbf{X}}$ 与 $\mathbf{S}$：ALFTemporal 从跨节点历史生成条件特征 $\mathbf{Z}$，经 FiLM 参数头调制 ALFGraph 的编码末状态；ALFGraph 以观测状态决定图传播的组合方式，完成编码与残差自回归解码。解码输出加回中心化参考值后，经角色缩放图修正得到唯一预测。两个分支共享输入张量与补全器，但不共享各自的主干参数；图1中的并列表示依赖关系，计算上先得到时间分支的调制参数，再执行图分支。

各组成部分沿用了已有算子：图循环插补来自 GRIN [1]，扩散卷积循环单元与门控循环结构来自 DCRNN 与 GRU [18,19]，传播算子的专家组合参考 MoGERNN 与稀疏专家混合 [7,21,22]，空间分区组织来自 PatchSTG [14]，两阶段注意力改编自 Crossformer [20]，特征调制采用 FiLM [16]。本文的方法贡献在于以观测支持组织这些计算：它决定哪些信息被补全、图传播如何组合，以及跨节点上下文在何处进入预测。

### 3.3 观测支持感知历史补全（SAHC）

共享补全对全部 $N$ 个节点、完整的 $L$ 步历史只运行一次：

$$
\mathbf{X}^{\mathrm{grin}}=\mathbf{M}\odot\mathbf{X}+(\mathbf{1}-\mathbf{M})\odot\mathbf{X}^{\mathrm{rec}},\qquad
\mathbf{X}^{\mathrm{rec}}=\mathrm{BiGRIL}\big(\mathbf{X}\odot\mathbf{M},\ \mathbf{M},\ [\mathbf{R},\mathbf{a}];\ \mathbf{A}\big). \tag{2}
$$

BiGRIL 沿时间正、反两个方向各运行一个图循环插补层 [1]：每一步先由上一隐藏状态给出初步估计，再经空间消息传递修正，以观测值替换可见位置后更新图循环状态；两个方向在每个历史位置上的表示与掩码经多层感知机（MLP）融合，得到重建 $\mathbf{X}^{\mathrm{rec}}$。角色与年龄作为外生输入进入两个方向的递推，掩码单独使用。补全器没有独立的重建损失，只通过下游预测损失训练。

对流量数据，补全之后再施加两项固定的输入处理。其一是窗口内插值：

$$
X^{\mathrm{fill}}_{t,n}=
\begin{cases}
\mathrm{Interp}(\mathbf{X}_{:,n},\mathbf{M}_{:,n})_t, & \text{插值开启},\ M_{t,n}=0,\ q_n>0,\\
X^{\mathrm{grin}}_{t,n}, & \text{其他情况},
\end{cases} \tag{3}
$$

其中 $\mathrm{Interp}$ 只使用当前窗口内的可见读数：两个可见读数之间线性插值，窗口两端之外取最近的可见值。没有可见读数（$q_n=0$）的节点保留 GRIN 由图消息传递得到的补全值。因此，插值与 GRIN 不是按 $q$ 互斥的两条支路：所有节点先经过 GRIN，插值只覆盖至少有一个可见读数的节点的缺失位置。其二是参考中心化：

$$
\bar{\mathbf{X}}=\mathbf{X}^{\mathrm{fill}}-\alpha\,\mathbf{v}, \tag{4}
$$

其中 $v_n$ 为窗口内最后一个可见的标准化读数，无可见读数时 $v_n=0$，并沿时间维广播；$\alpha$ 为固定的中心化强度，$\mathbf{S}$ 保持不变。中心化在输出端对应加回（3.4节）。速度数据不插值、不中心化，此时 $\bar{\mathbf{X}}=\mathbf{X}^{\mathrm{grin}}$。上述处理均不使用未来或被遮蔽的真值。保留观测值发生在中心化之前；中心化后，可见位置的数值也随之平移。两个分支读取同一份 $\bar{\mathbf{X}}$。

### 3.4 观测条件化图预测（ALFGraph）

**生命周期条件化图传播。** ALFGraph 以 DCRNN 的扩散卷积循环结构为基础 [18]，将门控中的单一传播算子替换为五种传播支持的条件化组合。五种支持依次为恒等矩阵 $\mathbf{P}_1=\mathbf{I}$、二值邻接的行归一化 $\mathbf{P}_2$、加权邻接的行归一化 $\mathbf{P}_3$，以及 $\mathbf{A}$ 与 $\mathbf{A}^{\top}$ 的随机游走转移矩阵 $\mathbf{P}_4$、$\mathbf{P}_5$。在非负邻接上，$\mathbf{P}_3$ 与 $\mathbf{P}_4$ 的归一化形式相同，但二者具有独立的投影参数与路由权重。对第 $t$ 步，路由器只读取生命周期上下文：

$$
\mathbf{w}_t=\mathrm{softmax}\big(\mathbf{S}_t\mathbf{W}_g\big)\in\mathbb{R}^{N\times5}, \tag{5}
$$

softmax 沿专家维对每个节点分别计算。给定门控输入 $\mathbf{U}\in\mathbb{R}^{N\times F}$，条件化传播为

$$
G(\mathbf{U},\mathbf{S}_t)=\sum_{j=1}^{5}\mathbf{w}_{t,j}\odot\big(\mathbf{P}_j\,\mathbf{U}\,\boldsymbol{\Theta}_j\big)+\mathbf{b}, \tag{6}
$$

其中 $\mathbf{w}_{t,j}\in\mathbb{R}^{N}$ 沿特征维广播，$\boldsymbol{\Theta}_j\in\mathbb{R}^{F\times64}$ 为第 $j$ 个支持的投影。五个支持全部参与加权，不做 top-$k$ 截断，也不加入噪声。路由器读取观测状态而非信号值：经过共享补全后，一个新接入（NAS）节点与一个活跃（AAS）节点可能具有相似的输入数值，而二者的生命周期上下文仍然不同。与 MoGERNN 的图专家混合 [7] 相比，这里的组合权重由逐步的观测状态决定，并作用于循环单元内部的每个门。

**图门控循环单元（GRU）。** 编码器第 $t$ 步的输入为 $\mathbf{x}_t=[\bar{\mathbf{X}}_t,\mathbf{S}_t]\in\mathbb{R}^{N\times7}$。每个门使用一个条件化传播模块：

$$
\begin{aligned}
\mathbf{r}&=\sigma\big(G_r([\mathbf{x}_t,\mathbf{h}],\mathbf{S}_t)\big),\quad
\mathbf{u}=\sigma\big(G_u([\mathbf{x}_t,\mathbf{h}],\mathbf{S}_t)\big),\\
\mathbf{c}&=\tanh\big(G_c([\mathbf{x}_t,\mathbf{r}\odot\mathbf{h}],\mathbf{S}_t)\big),\quad
\mathbf{h}'=\mathbf{u}\odot\mathbf{h}+(1-\mathbf{u})\odot\mathbf{c},
\end{aligned} \tag{7}
$$

其中 $\sigma(\cdot)$ 为 sigmoid，隐藏宽度为64。编码器与解码器各有两层，每层的三个门各有独立的传播参数与路由器，共12个条件化传播模块；各模块的参数在递推步之间共享。编码器沿 $t=1,\dots,L$ 递推，得到两层的末状态 $\mathbf{H}^{\mathrm{enc}}=(\mathbf{h}^{(1)}_L,\mathbf{h}^{(2)}_L)$。由于角色与年龄逐步进入路由，Join 节点在接入前后、Fault 节点在中断前后会使用不同的传播组合。

**残差自回归解码与参考恢复。** 解码器以经 FiLM 调制的编码末状态为初始隐藏状态（3.5节），以共享历史的末步作为初值 $\mathbf{y}_0=\bar{\mathbf{X}}_L$；该初值来自补全与中心化后的历史，不一定是最后一次真实观测。对 $k=1,\dots,H$：

$$
\mathbf{y}_k=\mathbf{y}_{k-1}+d\cdot\mathrm{Linear}\big(\mathbf{h}^{(2)}_k\big), \tag{8}
$$

其中 $\mathbf{h}^{(2)}_k$ 为解码器第二层在输入 $\mathbf{y}_{k-1}$ 下更新后的隐藏状态，$d$ 为固定的解码残差倍率。解码阶段没有未来的观测状态，路由统一使用末步上下文 $\mathbf{S}_L$，不对未来角色演变建模。每步回送的是未经修正的 $\mathbf{y}_k$。训练时采用受合法目标掩码约束的教师强制，其比例在前40轮线性衰减至零，评价时关闭。完成 $H$ 步解码后加回中心化参考值：

$$
\tilde{\mathbf{Y}}=[\mathbf{y}_1,\dots,\mathbf{y}_H]+\alpha\,\mathbf{v}. \tag{9}
$$

这一步只恢复3.3节减去的参考值，不是标准化的逆变换；误差计算时再用训练时段的标准化统计量还原到原始单位。

### 3.5 空间上下文时间条件分支（ALFTemporal）与 FiLM 调制

**时间 patch 嵌入。** 对每个节点，ALFTemporal 读取 $[\bar{\mathbf{X}},\mathbf{S},\tau/288,\delta/7]\in\mathbb{R}^{L\times9}$，其中 $\tau=12\cdot\text{hour}+\lfloor\text{minute}/5\rfloor\in\{0,\dots,287\}$ 为日内时段（time of day, ToD），$\delta\in\{0,\dots,6\}$ 为星期（day of week, DoW）。卷积核长度与步长均为12的时间卷积将整个窗口压缩为一个64维向量 $\mathbf{e}_n$，因此每个节点只有一个时间 patch，窗口内读数的位置信息由卷积权重保留。随后拼接

$$
\mathbf{z}^{0}_n=\big[\mathbf{e}_n\ \|\ \mathbf{E}^{\mathrm{tod}}_{\tau_L}\ \|\ \mathbf{E}^{\mathrm{dow}}_{\delta_L}\ \|\ \phi(\mathbf{g}_n)\big]\in\mathbb{R}^{160}, \tag{10}
$$

其中 $\mathbf{E}^{\mathrm{tod}}$、$\mathbf{E}^{\mathrm{dow}}$ 为末步日内时段与星期的32维嵌入。结构嵌入 $\phi(\mathbf{g}_n)$ 由所有节点共享的多层感知机（MLP，7→64→32）作用于静态属性 $\mathbf{g}_n\in\mathbb{R}^7$ 得到：两维坐标、入度、出度、总加权度、对数总度与孤立标志；无坐标的数据集坐标置零。模型不使用逐节点可训练的身份嵌入，以免为保留节点引入可训练的身份参数。

**分区空间上下文编码。** 参照 PatchSTG 的空间分区组织 [14]，节点被划分为 $P=32$ 个分区，每个分区含 $C$ 个槽位：速度数据依据已核验的地理坐标以 KD-tree 划分，流量数据依据图拓扑做考虑连通分量的广度优先搜索（BFS）划分。节点不足 $C$ 个的分区以邻接排序的真实节点补足，这些补足副本只提供上下文，不产生输出。分区后的特征经一个两阶段注意力块处理，该结构改编自 Crossformer [20]，单头，各阶段均含残差连接、层归一化与前馈层。第一阶段为**跨分区注意力**：对每个槽位索引 $c$，在 $P$ 个分区的同位 token $\{\mathbf{z}_{p,c}\}_{p=1}^{P}$ 之间做自注意力；在 Crossformer 中该阶段沿时间轴进行，这里改为沿分区轴。第二阶段为**分区内潜在汇聚与回传**：每个分区 $p$ 有4个可学习的潜在 token $\mathbf{Q}_p\in\mathbb{R}^{4\times160}$，

$$
\mathbf{B}_p=\mathrm{Attn}\big(\mathbf{Q}_p,\mathbf{Z}_p,\mathbf{Z}_p\big),\qquad
\mathbf{Z}_p\leftarrow\mathrm{Attn}\big(\mathbf{Z}_p,\mathbf{B}_p,\mathbf{B}_p\big), \tag{11}
$$

其中 $\mathbf{Z}_p\in\mathbb{R}^{C\times160}$ 为分区内的槽位特征。潜在 token 先从 $C$ 个槽位汇聚信息，再回传给各槽位，分区内的交互因此经过4个 token 的瓶颈，而不是 $C\times C$ 的直接注意力。处理后按原始节点顺序取回，得到条件特征 $\mathbf{Z}\in\mathbb{R}^{N\times160}$。ALFTemporal 不产生独立的预测，只为图分支提供条件。

**FiLM 调制。** FiLM 参数头将 $\mathbf{Z}$ 映射为逐节点的缩放与平移参数，并调制两层编码末状态 [16]：

$$
(\Delta\boldsymbol{\gamma},\boldsymbol{\beta})=\mathrm{MLP}_{160\to32\to128}(\mathbf{Z}),\qquad
\mathbf{h}^{(l)}_{\mathrm{mod}}=(1+\Delta\boldsymbol{\gamma})\odot\mathbf{h}^{(l)}_L+\boldsymbol{\beta},\quad l=1,2, \tag{12}
$$

其中 $\Delta\boldsymbol{\gamma},\boldsymbol{\beta}\in\mathbb{R}^{N\times64}$。参数头的最后一层零初始化，训练开始时调制为恒等映射。同一组参数广播到两层，只在编码完成、解码开始之前作用一次，不在每个预测步重新生成。观测支持对 FiLM 参数的影响只经由 ALFTemporal 的输入，参数头与 $\mathbf{S}$ 之间没有直接连接。

### 3.6 角色缩放图修正

解码与参考恢复之后，输出端施加一个有界的邻域修正。令 $\mathbf{P}^{+}$、$\mathbf{P}^{-}$ 分别为 $\mathbf{A}+\mathbf{I}$ 及其转置的行归一化矩阵，

$$
\mathbf{D}=\mathrm{Linear}_{32\to1}\Big(\mathrm{GELU}\big(\mathrm{Linear}_{3\to32}([\tilde{\mathbf{Y}},\ \mathbf{P}^{+}\tilde{\mathbf{Y}},\ \mathbf{P}^{-}\tilde{\mathbf{Y}}])\big)\Big),\qquad
\hat{\mathbf{Y}}=\tilde{\mathbf{Y}}+0.05\cdot s(\mathbf{R}_L)\odot\tanh(\mathbf{D}), \tag{13}
$$

其中拼接沿每个预测步、每个节点的特征维进行，$s(\cdot)$ 为按末步角色取值的固定倍率，AAS、NAS、VS、FS 依次为0、1、1、0.25。最后一层零初始化；由于 tanh 有界，修正量的绝对值在标准化单位下不超过0.05。末步角色为活跃（AAS）的节点自身不被修正，但仍作为邻居参与其他节点的修正。该模块只读取末步角色，不读取年龄与掩码，也没有独立的路由器；修正结果不回送解码器。

### 3.7 训练目标

令 $e=|\hat{y}-y|$ 为原始单位下的绝对误差，$\sigma_{\mathrm{data}}$ 为训练时段目标变量的标准差。逐元素损失为

$$
\ell(e)=\frac{e}{\sigma_{\mathrm{data}}}\tanh(0.1\,e). \tag{14}
$$

该损失在小误差附近近似 $0.1e^2/\sigma_{\mathrm{data}}$，大误差时趋近标准化 MAE；权重项参与梯度计算。四个数据集取相同系数，但 $e$ 分别以各自的速度或流量单位计量，因此相同系数并不意味着跨量纲的同等加权。记 $\Omega$ 为一个批次内有效的目标元素，$\Omega_r$ 为其中末步角色为 $r$ 的节点所对应的元素，$\mathcal{R}_b$ 为该批次实际出现的角色集合，则

$$
\mathcal{L}_{\mathrm{mean}}=\frac{1}{|\Omega|}\sum_{\Omega}\ell(e),\qquad
\mathcal{L}_{\mathrm{role}}=\frac{1}{|\mathcal{R}_b|}\sum_{r\in\mathcal{R}_b}\frac{1}{|\Omega_r|}\sum_{\Omega_r}\ell(e),\qquad
\mathcal{L}=0.9\,\mathcal{L}_{\mathrm{mean}}+0.1\,\mathcal{L}_{\mathrm{role}}+\mathcal{L}_{\mathrm{route}}. \tag{15}
$$

$\mathcal{L}_{\mathrm{role}}$ 使样本较少的角色在优化中获得与其他角色相同的权重。$\mathcal{L}_{\mathrm{route}}$ 为路由平衡项 [21,22]：每个条件化传播模块在每次调用时计算 $0.01\,[\mathrm{CV}^2(\text{importance})+\mathrm{CV}^2(\text{load})]$，其中 importance 为批次内各专家路由权重之和，$\mathrm{CV}^2$ 为平方变异系数；先在递推调用间取平均，再对12个模块取平均。由于五个支持全部参与加权，load 项恒为零，起作用的是 importance 项。训练目标不含独立的重建损失或额外预测头。该目标是对现有训练方式的描述，不作为独立的方法贡献。

### 3.8 数据集特定设置

四个数据集共享3.3–3.7节的同一可学习结构，参数量均为1,037,529。数据集之间的差异仅在于训练前固定的设置：分区容量 $C$（METR-LA、PEMS-BAY、PEMS04、PEMS08 依次为7、16、10、6），窗口插值（速度数据关闭、流量数据开启），解码残差倍率 $d$（0.3、0.3、1、1）与中心化强度 $\alpha$（0、0、0、0.5）。这些设置不可学习，也不在推理时由路由选择；完整取值与训练超参数见4.1节。

## 4 实验

观测支持的变化不仅改变输入的缺失模式，也改变可用历史与空间信息参与预测的条件。实验围绕四个问题展开：**RQ1**，ALFNet 能否在动态观测支持下取得跨数据集、跨条件的预测收益；**RQ2**，哪些条件化组件贡献了这些收益；**RQ3**，模型是否利用观测支持的生命周期与时间布局，而不仅是重建缺失输入；**RQ4**，样本层面的预测行为如何。实验覆盖 METR-LA、PEMS-BAY、PEMS04 和 PEMS08 四个数据集。总体比较与消融使用测试集；固定权重的配对机制分析与单因素敏感性实验使用验证集，分别回答预测能力、行为解释与参数作用三类问题。

### 4.1 实验设置

**数据集与任务。** METR-LA 与 PEMS-BAY 来自 [18]，PEMS04 与 PEMS08 来自 [23]。四个数据集均以5分钟为采样间隔，使用过去12步预测未来12步。METR-LA 与 PEMS-BAY 预测交通速度，PEMS04 与 PEMS08 预测交通流量；速度与流量在各自原始单位下评价，不跨数据集平均绝对误差。

| 数据集 | 节点数 | 时间点数 | 预测变量 | 空间分区依据 |
|---|---:|---:|---|---|
| METR-LA | 207 | 34,272 | 速度 | 地理坐标，KD-tree |
| PEMS-BAY | 325 | 52,116 | 速度 | 地理坐标，KD-tree |
| PEMS04 | 307 | 16,992 | 流量 | 图拓扑，BFS |
| PEMS08 | 170 | 17,856 | 流量 | 图拓扑，BFS |

时间按60%／20%／20%顺序划分为训练、验证和测试时段。在拓扑已知的归纳设置下，训练仅使用训练节点的动态信号，验证与测试分别引入各自的保留节点；标准化统计量仅由训练时段的训练节点计算。所有实验使用随机种子32，因此不将较小的单次训练差异解释为统计显著。

**观测支持协议。** 零历史（Unseen）、持续观测（Steady）、逐步加入（Join）和历史中断（Fault）四类条件等配额。评价时，Unseen 与 Join 在保留节点上构造，Steady 与 Fault 在已知节点上构造（3.1节），因此条件之间的差异同时包含历史支持与节点群体的差异。Join 按接入进展分为 Cold、Warm、Ready，分别对应1、2–11、12步；Fault 按中断进展分为 Onset、Decay、Outage，对应相同区间。自然缺失与协议共同决定最终掩码，因此阶段长度不总等于有效观测数。所有方法共享划分、预测目标、数据暴露边界与评价掩码。

**比较方法。** 比较方法依据第2节的三个研究方向选取，共八个。不完整观测下的预测（2.1）选取 UIGNN 与 GinAR，分别代表面向缺失位置的不确定性预测和面向变量缺失的插值注意力—自适应图卷积递归预测；归纳与部分感知预测（2.2）选取 MoGERNN 与 SLPF，分别代表图专家混合的归纳预测和长期部分感知的空间转移预测；空间依赖与时空表示学习（2.3）选取 STGCN、STID、PatchSTG 与 STBP，分别代表固定图卷积、时空身份表示、空间分块交互和持续上下文模式库。第2节讨论但未纳入比较的方法及原因如下：GRIN、PriSTI 与 IGNNK 以历史恢复或同时刻插值为目标，不直接预测未来；GCN-M 的多尺度历史记忆需要读取12步输入窗口之外的历史，超出共同输入预算；SIML-MSIGN 未获得可核实的官方实现；PDFormer 与 PatchSTG 同属基于注意力的空间交互，以与 ALFTemporal 分区组织直接相关的 PatchSTG 为代表；Graph WaveNet 与 AGCRN 学习的空间关系在推理时不随输入窗口变化，这类与观测支持无关的可学习传播由4.4节的 w/o Adaptive Routing 在同一模型内检验，但该内部对照不能替代与这些方法本身的比较。各基线保留自身预测主干与目标函数，不引入 ALFNet 独有的生命周期特征、条件化模块或误差幅度加权目标；SLPF 使用单阶段预测适配，STBP 使用离线固定图适配，因而本实验比较的是这些方法在共同输入与评价边界下的适配版本，而非其原始训练或持续学习流程。

**评价指标。** MAE、RMSE、MAPE 均在原速度（流量）尺度上计算，MAPE 以百分比表示。各条件内按日历与事件进展分层平衡汇总，再对四类条件等权平均得到 LifeAvg；LifeAvg RMSE 是条件级 RMSE 的平均。三项指标分别反映绝对误差、大残差和相对误差。

**训练与实现。** 四个数据集的 ALFNet 均含1,037,529个模型参数，训练目标见3.7节，采用 AdamW，批大小64，最多训练100轮；第40轮起使用耐心值10的提前停止，按验证集 LifeAvg MAE 保存权重。教师强制率在前40轮线性衰减，评价时关闭。图预测分支与共享补全、ALFTemporal、FiLM 参数头与图修正等其余参数的基础学习率依次为 $5\times10^{-4}$、$10^{-4}$、$5\times10^{-5}$，各数据集的倍率见下表，训练不使用学习率调度器。四个数据集均使用32个空间分区；容量表示每个分区的节点槽位数，而非分区数量。ALFTemporal dropout 均为0.1。

| 数据集 | 分区数 | 分区容量 | 学习率倍率 | 权重衰减 | 窗口插值 | 解码残差倍率 | 中心化强度 |
|---|---:|---:|---:|---:|---|---:|---:|
| METR-LA | 32 | 7 | 1 | $10^{-4}$ | 关闭 | 0.3 | 0 |
| PEMS-BAY | 32 | 16 | 2 | $10^{-5}$ | 关闭 | 0.3 | 0 |
| PEMS04 | 32 | 10 | 1 | $10^{-4}$ | 开启 | 1 | 0 |
| PEMS08 | 32 | 6 | 1 | $10^{-3}$ | 开启 | 1 | 0.5 |

PEMS-BAY 的三组学习率因此为 $10^{-3}$、$2\times10^{-4}$、$10^{-4}$。流量数据的窗口内插值与中心化见3.3节：所有节点先经过共享补全，插值只覆盖有可见读数的节点的缺失位置，不使用未来或被遮蔽的真值；中心化在输出端对应还原。部分历史配置的选择曾参考测试结果，因此本文测试结果是既定研究流程下的最终评估，而非全程未参与选择的独立确认。

### 4.2 总体预测性能（RQ1）

ALFNet 在速度与流量数据上均取得三项 LifeAvg 最优，表明围绕观测支持组织预测计算的综合收益能够覆盖不同预测变量与路网。表1中，四个数据集相对各指标最强基线的 MAE 降幅依次为15.63%、11.95%、8.66%和7.86%；RMSE 降幅为7.51%–10.56%，MAPE 降幅为15.49%–21.42%。三类误差同步改善，说明收益并非来自对某一种误差尺度的优化。

**表1　四个数据集的总体预测性能（测试集 LifeAvg）。** 最优加粗，次优加下划线，排名与正文降幅均按未舍入值计算；各数据集独立比较，速度与流量的绝对误差不可横向比较。

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">模型</th>
      <th colspan="3" scope="colgroup">METR-LA</th>
      <th colspan="3" scope="colgroup">PEMS-BAY</th>
      <th colspan="3" scope="colgroup">PEMS04</th>
      <th colspan="3" scope="colgroup">PEMS08</th>
    </tr>
    <tr>
      <th>MAE↓</th><th>RMSE↓</th><th>MAPE(%)↓</th>
      <th>MAE↓</th><th>RMSE↓</th><th>MAPE(%)↓</th>
      <th>MAE↓</th><th>RMSE↓</th><th>MAPE(%)↓</th>
      <th>MAE↓</th><th>RMSE↓</th><th>MAPE(%)↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>UIGNN</td><td>4.593</td><td>6.971</td><td>12.547</td><td><u>3.093</u></td><td>4.773</td><td>7.559</td><td><u>43.317</u></td><td><u>57.680</u></td><td><u>29.598</u></td><td>41.924</td><td>53.363</td><td><u>54.342</u></td></tr>
    <tr><td>GinAR</td><td>7.092</td><td>10.201</td><td>23.129</td><td>5.059</td><td>7.246</td><td>13.488</td><td>91.079</td><td>113.366</td><td>57.981</td><td>87.273</td><td>107.519</td><td>151.591</td></tr>
    <tr><td>MoGERNN</td><td><u>4.554</u></td><td><u>6.803</u></td><td><u>12.352</u></td><td>3.116</td><td><u>4.693</u></td><td><u>7.426</u></td><td>49.996</td><td>64.598</td><td>40.308</td><td>58.770</td><td>72.800</td><td>100.006</td></tr>
    <tr><td>SLPF</td><td>5.878</td><td>8.101</td><td>16.362</td><td>4.017</td><td>5.684</td><td>9.700</td><td>93.898</td><td>116.595</td><td>61.203</td><td>86.911</td><td>109.265</td><td>166.290</td></tr>
    <tr><td>STGCN</td><td>4.970</td><td>7.405</td><td>13.857</td><td>3.169</td><td>4.888</td><td>7.844</td><td>45.608</td><td>60.451</td><td>33.381</td><td>44.279</td><td>55.443</td><td>62.934</td></tr>
    <tr><td>STID</td><td>4.923</td><td>7.479</td><td>13.752</td><td>3.239</td><td>5.007</td><td>7.695</td><td>46.385</td><td>60.235</td><td>29.819</td><td><u>39.917</u></td><td><u>51.097</u></td><td>68.921</td></tr>
    <tr><td>PatchSTG</td><td>5.557</td><td>8.045</td><td>17.921</td><td>3.220</td><td>4.842</td><td>7.717</td><td>48.627</td><td>63.670</td><td>32.134</td><td>42.677</td><td>54.924</td><td>74.820</td></tr>
    <tr><td>STBP</td><td>4.965</td><td>7.546</td><td>13.697</td><td>3.186</td><td>4.887</td><td>7.722</td><td>46.905</td><td>61.023</td><td>31.755</td><td>41.698</td><td>53.428</td><td>61.022</td></tr>
    <tr><td>ALFNet</td><td><b>3.842</b></td><td><b>6.085</b></td><td><b>10.337</b></td><td><b>2.723</b></td><td><b>4.201</b></td><td><b>6.276</b></td><td><b>39.565</b></td><td><b>53.346</b></td><td><b>23.257</b></td><td><b>36.780</b></td><td><b>46.567</b></td><td><b>43.000</b></td></tr>
  </tbody>
</table>

一个值得注意的现象是，各数据集上的最强基线并不相同：次优 MAE 在 METR-LA 上来自 MoGERNN，在 PEMS-BAY 与 PEMS04 上来自 UIGNN，在 PEMS08 上来自 STID。缺失感知、归纳预测与通用时空表示三类已有能力各自在某些路网上有竞争力，而 ALFNet 在全部数据集上保持领先，与引言的判断一致：缺失处理、空间传递与归纳迁移是互补而非充分的，关键在于让预测计算随可用信息的条件而调整。LifeAvg 对四类条件等权汇总，反映的是跨条件的综合表现；收益是否覆盖不同支持状态，由下一节检验。

### 4.3 不同观测条件下的预测性能（RQ1）

总体优势是否仅由某个容易预测的条件拉低平均误差？图2将四个数据集的33项结果（4个主条件、LifeAvg、6个阶段，各3项指标）置于同一版面，并以**每一格最强基线**为参照，比固定单一基线更严格地呈现优势的覆盖范围。

![四数据集条件与阶段性能](../../final_figures_20261017/test/fig2_condition_stage.png)

**图2　四数据集的条件与阶段收益（测试集）。** 数值为 $100\times(1-E_{\mathrm{ALFNet}}/\min_b E_b)$，正值表示优于八个基线中该格的最优者。四个面板共享对称色标，保留全部正负收益；每个面板上部为主条件与 LifeAvg，下部为接入与中断阶段。

ALFNet 在 METR-LA、PEMS-BAY、PEMS04、PEMS08 上分别取得33/33、29/33、29/33、30/33项第一。最稳定的结果出现在需要本地证据逐步形成或被打断的条件：Join 与 Fault 的三项指标在四个数据集上均优于全部基线，说明当保留节点开始提供少量观测、或已知节点失去近期读数时，条件化的信息利用能够持续带来收益。这与引言提出的预测要求一致——逐步加入（Join）阶段需要在有限的本地证据与空间上下文之间协调，历史中断（Fault）阶段则需要在历史连续性降低后维持预测依据。

边界同样清晰。在没有任何本地历史的 Unseen 条件下，优势依赖于数据集：METR-LA 上三项指标均为第一，PEMS-BAY 上 MAE 与 MAPE 为第一而 RMSE 比该指标最强基线高0.9%；流量数据上，PEMS04 的 MAE 与 RMSE 分别高10.4%和10.5%（MAPE 仍降低4.5%），PEMS08 的三项误差分别高17.9%、13.2%和5.0%。其余未取得第一的格子集中在本地历史接近缺失的端点（PEMS-BAY 的 Cold 与 Outage），以及 PEMS04 中 Steady 与 Ready 的 MAE（差距分别为1.0%和0.5%）。因此，流量数据上的收益主要来自本地证据存在或部分存在时对历史与空间信息的利用，零历史（Unseen）泛化则是不同的问题，本文未对其差异来源做进一步拆分。

### 4.4 条件化组件的贡献（RQ2）

由观测状态决定的空间路由、跨节点时间上下文与时间条件分支共同支撑了跨数据集的预测收益。表2中，移除 Adaptive Routing、Spatial Context 或 ALFTemporal–FiLM 均使四个数据集的三项 LifeAvg 全部变差；移除 State Age 在12个数据集—指标格中使11格变差。所有变体与对应完整模型共享超参数、随机种子、数据划分和检查点准则。

**表2　四数据集四项组件消融（测试集 LifeAvg）。** 每行为一个变体，各数据集每项指标的最优值加粗、次优值加下划线，按未舍入值排名。PEMS08 的完整模型与四项消融均使用权重衰减 $10^{-3}$，与表1一致。

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col">模型</th>
      <th colspan="3" scope="colgroup">METR-LA</th>
      <th colspan="3" scope="colgroup">PEMS-BAY</th>
      <th colspan="3" scope="colgroup">PEMS04</th>
      <th colspan="3" scope="colgroup">PEMS08</th>
    </tr>
    <tr>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
      <th scope="col">MAE↓</th><th scope="col">RMSE↓</th><th scope="col">MAPE(%)↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ALFNet</td>
      <td><b>3.842</b></td><td><b>6.085</b></td><td><b>10.337</b></td>
      <td><b>2.723</b></td><td><b>4.201</b></td><td><b>6.276</b></td>
      <td><b>39.565</b></td><td><u>53.346</u></td><td><b>23.257</b></td>
      <td><b>36.780</b></td><td><b>46.567</b></td><td><b>43.000</b></td>
    </tr>
    <tr>
      <td>w/o State Age</td>
      <td><u>3.878</u></td><td><u>6.097</u></td><td><u>10.423</u></td>
      <td><u>2.732</u></td><td><u>4.248</u></td><td><u>6.464</u></td>
      <td><u>39.757</u></td><td><b>53.136</b></td><td>24.957</td>
      <td><u>38.162</u></td><td><u>47.792</u></td><td><u>48.701</u></td>
    </tr>
    <tr>
      <td>w/o Adaptive Routing</td>
      <td>3.920</td><td>6.127</td><td>10.602</td>
      <td>2.770</td><td>4.280</td><td>6.514</td>
      <td>39.826</td><td>53.401</td><td><u>23.415</u></td>
      <td>38.783</td><td>49.591</td><td>58.378</td>
    </tr>
    <tr>
      <td>w/o Spatial Context</td>
      <td>4.400</td><td>6.645</td><td>12.529</td>
      <td>2.958</td><td>4.493</td><td>7.011</td>
      <td>40.292</td><td>54.694</td><td>24.877</td>
      <td>42.382</td><td>53.475</td><td>53.061</td>
    </tr>
    <tr>
      <td>w/o ALFTemporal–FiLM</td>
      <td>4.264</td><td>6.560</td><td>11.760</td>
      <td>2.890</td><td>4.487</td><td>6.992</td>
      <td>42.137</td><td>56.293</td><td>26.361</td>
      <td>40.906</td><td>51.874</td><td>52.003</td>
    </tr>
  </tbody>
</table>

在表2所列四项消融中，完整模型在 METR-LA、PEMS-BAY 与 PEMS08 上三项指标均最优，在 PEMS04 上取得 MAE 与 MAPE 最优；唯一例外是 PEMS04 的 w/o State Age 的 RMSE（53.136，较完整模型低0.39%）。本文按观测值报告，未将小差异解释为统计显著。表2只覆盖生命周期信息、观测状态条件化传播与时间上下文四项干预，不据此推断其余设计（例如观测支持感知历史补全（SAHC）与角色缩放图修正）均有必要。

**跨节点上下文与时间条件分支贡献最稳定。** w/o Spatial Context 保持参数规模不变，只让同一组时间处理模块逐节点工作，不再利用跨节点分区上下文；其 MAE 增幅在四个数据集上为1.84%–15.23%，在 METR-LA 与 PEMS08 上超过14%。w/o ALFTemporal–FiLM 移除整个时间条件分支及其调制，MAE 增幅为6.1%–11.2%。前者排除了"单纯参数增多"的解释，后者说明完整分支对预测有实质作用；两项干预的范围有重叠，不视作互不相关的贡献相加。

**观测状态条件化的传播在所有数据集上带来收益。** w/o Adaptive Routing 保留五种传播支持，仅以与输入无关的全局可学习混合替代由观测状态决定的路由，四个数据集的三项误差均上升，其中 PEMS08 的 MAPE 增幅达35.76%。在传播算子不变的前提下，随观测状态变化的组合方式仍然有益，支持依据观测条件调整空间信息交换的设计。

**状态年龄提供补充性的阶段区分信息。** w/o State Age 用掩码替代年龄通道并保留角色输入，在 METR-LA、PEMS-BAY 和 PEMS08 上使三项指标同时上升，在 PEMS08 上对 MAPE 的影响尤为明显（增幅13.26%）。这支持年龄特征在不同历史阶段之间提供区分信息；PEMS04 的 RMSE 例外说明该补充并非在所有误差尺度上同等重要，也不能由这一干预单独推断角色编码本身的贡献。

### 4.5 观测轨迹下的预测响应（RQ3）

条件平均无法区分交通目标变化和观测变化的影响。为隔离支持变化，每个数据集固定64个历史与未来自然完整的已知节点验证窗口，保持预测目标、图结构和其他节点输入不变，只干预焦点节点；样本按固定随机规则选取，不依据预测误差筛选。加入条件保留最近3步或6步，以完整12步为参照；中断条件删除最近3步、6步或全部历史，并同步更新角色与年龄。模型参数固定，因此结果反映对可用证据的响应而非在线学习效果；这是已知节点上的受控输入实验，不等同于未见节点冷启动。

对比的代表基线按各数据集完整验证集 LifeAvg MAE 在三类方法中各取最优：前两类为 UIGNN 与 MoGERNN，第三类在速度数据上为 STBP，在流量数据上为 PatchSTG。后文4.6与4.8沿用这一选择。图3、图4与图6均使用与表1一致的最终权重，包括 PEMS08 的权重衰减 $10^{-3}$ 配置；正文统一展示验证集，按相同抽样与干预规则得到的测试集结果在4.8节末段核验。两套结果完整保留，不在不同图或数据集之间按结果择优切换划分。

![四数据集观测轨迹响应](../../final_figures_20261017/validation/fig3_trajectory.png)

**图3　固定目标下的观测加入与中断（验证集）。** 每行为一个数据集，左列为加入，右列为中断。点为同一组64个窗口、全部12个预测步上的平均 MAE，越靠左越好；同一行共享横轴尺度，离散干预点不连成在线学习曲线。

在加入条件下，ALFNet 在四个数据集的三个设置中均低于三个代表基线，其误差在 METR-LA 与 PEMS08 上随近期历史增加而下降。在中断条件下，速度数据的全部设置中 ALFNet 保持最低误差，完全中断时 MAE 分别为3.628（METR-LA）和2.979（PEMS-BAY）；流量数据上，PEMS08 在完全中断时仍优于三个代表基线，PEMS04 则在部分历史可用时保持优势，完全中断端点略高于 MoGERNN。PEMS08 的部分中断劣于 PatchSTG：中断3步与6步时 MAE 分别为23.126／22.232和24.270／22.880，误差高约4.0%与6.1%，因此不能概括为所有中断强度均占优。整体上，新增本地观测的利用与失去本地观测后的空间补偿共同决定模型响应，而不是单一的缺失率决定性能。这些受控响应与条件化设计相符，但点图本身不能证明某个内部机制的因果作用，需要与表2中移除跨节点上下文和条件分支的干预结合解读。

### 4.6 观测数量与时间布局（RQ3）

相同数量的观测并不提供相同的信息。沿用图3的配对窗口，固定活跃（AAS）角色与年龄，只改变可见读数在12步历史中的位置：Recent 保留近期历史，Early 保留早期历史，Dispersed 分散保留，Middle Gap 保留中间连续缺口两侧的读数。所有布局只删除自然存在的读数，不合成新观测。

![四数据集观测数量与时间布局](../../final_figures_20261017/validation/fig4_layout.png)

**图4　观测数量与时间布局的比较（验证集）。** 行为数据集、列为布局，每组比较相同 $q=3,6,9$ 下的四个模型；柱轴从零开始，同一数据集的各布局共享尺度。共享端点 $q=0$ 与 $q=12$ 不在四种布局中重复展示。

四个数据集都体现出近期信息的价值：固定 $q=6$ 时，ALFNet 在 Recent 与 Early 布局下的 MAE 分别为3.164／3.808（METR-LA）、2.188／3.165（PEMS-BAY）、32.336／40.265（PEMS04）和22.822／28.404（PEMS08），Recent 一致更低。除 METR-LA 外，其余三个数据集上的分散历史略优于单独保留近期历史，说明新近程度与时间覆盖共同决定有效信息，单纯统计可见数量不足以刻画预测条件。这一观察对应引言的论点：观测数量只描述有多少信息，时间布局还决定信息来自何时。

在各数据集12个数量—布局组合中，ALFNet 分别在12、8、11、9个组合上取得最低 MAE，收益覆盖近期、分散与连续缺口等不同模式。未取得最低的组合主要出现在 PEMS-BAY（由 STBP 取得）以及 PEMS08 的 Early 布局（由 PatchSTG 取得），即该布局中最近观测缺失、仅保留早期读数的情形；这与4.3中优势在本地证据偏弱时更不稳定的观察方向一致，但本文未将布局与基线内部机制的差异做进一步拆分。

### 4.7 超参数敏感性

在各数据集固定锚点上，每次只改变分区容量、学习率倍率、ALFTemporal dropout 或 AdamW 权重衰减中的一项，其余设置不变。所有配置以相同种子训练，并按验证集 LifeAvg MAE 选择检查点。速度数据复用已完成的固定锚点实验，流量数据使用最终插值、倍率与中心化方案下的单因素实验。

| 数据集 | 锚点（容量／LR／dropout／权重衰减） | 容量候选 | LR 候选 | dropout 候选 | 权重衰减候选 |
|---|---|---|---|---|---|
| METR-LA | 8 / 1 / 0.1 / 1e-4 | 7、8、10、12 | 0.5、1、2 | 0.05、0.1、0.2 | 1e-5、1e-4、1e-3 |
| PEMS-BAY | 16 / 1 / 0.1 / 1e-4 | 11、12、14、16 | 0.5、1、2 | 0.05、0.1、0.2 | 1e-5、1e-4、1e-3 |
| PEMS04 | 10 / 1 / 0.1 / 1e-4 | 10、14、18 | 0.5、1、2 | 0.05、0.1、0.2 | 1e-5、1e-4、1e-3 |
| PEMS08 | 6 / 1 / 0.1 / 1e-4 | 6、8、10 | 0.5、1、2 | 0.05、0.1、0.2 | 1e-5、1e-4、1e-3 |

![四数据集单因素超参数敏感性](../../final_figures_20261017/validation/fig5_hyperparameters.png)

**图5　单因素敏感性（验证集）。** 行为数据集、列为参数；每个参数值用三种颜色表示 LifeAvg MAE、RMSE、MAPE 相对本数据集锚点的百分比变化。零线为锚点，负值更优；线段连接零线与实测值，不表示参数间插值或置信区间。

参数作用具有数据集依赖性。速度数据上，MAE 的变化多在3%以内，例外是 METR-LA 的学习率减半（MAE 上升6.00%）与容量10（上升4.23%）；两个数据集均在 dropout 0.1 下取得三项指标的最低值。PEMS-BAY 的锚点容量16为所测最大值，较小容量在三项指标上均变差，说明该数据集上更大的分区容量有利。流量数据则呈现更明显的容量—误差尺度取舍：PEMS04 增大容量降低 MAE 与 RMSE，却使 MAPE 上升（容量18时约上升9.5%）；PEMS08 对正则化更敏感，权重衰减 $10^{-3}$ 将三项指标分别降低8.60%、7.56%和6.99%，这也是其最终采用该取值的依据。

同时，部分非锚点配置在验证集上优于锚点（如 METR-LA 的学习率加倍），说明速度数据的锚点并非验证最优，其与表1最终配置也有所不同。单因素实验因此用于解释参数作用，而不用于宣称最终配置在所有指标上全局最优；同一实验内部始终使用各自锚点，不将其误差变化当作相对最终模型的收益。本节不以额外的测试集搜索代替验证分析。

### 4.8 样本层面的预测行为（RQ4）

为观察本地历史完全不可用时预测如何偏离真实值，我们将分析分为全部窗口的误差分布与具体预测目标两部分。图6使用图3相同的64个已知节点窗口，在完全中断条件下比较各模型。

![四数据集误差分布及预测案例](../../final_figures_20261017/validation/fig6_cases.png)

**图6　完全中断条件下的分布与案例分析（验证集）。** 左列为全部64个窗口 MAE 的经验累积分布，同一误差阈值下曲线越高表示更多窗口达到该精度，不截去大误差。右列为优势示例：先确定本数据集该条件下平均 MAE 最低的代表基线，再从 ALFNet 优于它的窗口中选取收益最接近正收益中位数的一个。该选择用于展示实际存在的优势，不代表随机案例或总体胜率；黑线为真值，浅蓝色带为 ALFNet 与真值的差距，不是置信区间。

从分布看，速度数据上 ALFNet 的窗口 MAE 第90百分位均低于三个代表基线，表明速度数据的总体收益包含对较大窗口误差的控制。相对该条件下平均 MAE 最低的代表基线，ALFNet 在四个数据集上的窗口胜率依次为54.7%、51.6%、50.0%和62.5%，说明总体平均排序不等同于逐窗口占优；仅由这些胜率不能推断所有数据集的收益均来自尾部误差抑制。案例曲线仅用于直观展示预测水平：PEMS04 示例中 ALFNet 较好地维持了真实流量区间，减轻了其他方法的持续低估；PEMS08 示例中其预测跟随流量下降趋势，并减轻部分基线持续低估的偏差。局部波动仍无法完全追随，但预测依据并未因焦点节点失去历史而消失。由于案例按正收益中位数选取，本文不据此推导逐样本结论。

**跨划分核验。** 在相同最终权重和抽样规则下，测试集图3的四数据集六个唯一干预点均取得最低MAE；图4的最优组合数依次为12、5、12、9，与验证集的12、8、11、9并不一致。因此，轨迹收益在测试集上得到补充支持，但布局优势具有样本和划分依赖性；两套结果不能通过逐图择优拼成一组结论。

## 5 结论

本文研究路网拓扑已知、部分节点不参与动态信号训练时，窗口内观测支持变化下的归纳时空预测，并提出 ALFNet。ALFNet 以逐步的观测角色、状态年龄和掩码构成生命周期上下文：观测支持感知历史补全（SAHC）为两个分支提供同一份历史，观测条件化图预测（ALFGraph）依据观测状态组合图传播算子，空间上下文时间条件分支（ALFTemporal）经 FiLM 调制图编码状态，输出端再施加由末步角色控制的有界修正。本文的贡献不在于提出新的基础算子，而在于以观测支持组织补全、传播与上下文的计算方式。

实验证据与引言的论点逐一对应。总体与条件评价表明，ALFNet 在四个数据集的跨条件平均误差（LifeAvg）三项指标上均优于八个比较方法，逐步加入（Join）与历史中断（Fault）两类主条件的优势在四个数据集上一致；组件消融表明，由观测状态决定的传播组合、跨节点时间上下文与时间条件分支在完整框架中起作用；在预测目标与非干预输入保持不变的配对干预中，加入或删除本地历史、改变可见读数的时间位置都伴随相应的预测响应，时间布局与可见数量同样重要。这些结果支持本文的核心主张：观测支持应同时作为输入有效性的描述和预测计算的条件，而不能仅由训练身份或掩码比例刻画。

结果的边界同样明确。未取得第一的结果集中在本地历史接近缺失的端点：两个流量数据集的零历史（Unseen）条件、PEMS-BAY 的 Cold 与 Outage 阶段，以及 PEMS04 的完全中断干预；PEMS08 的部分中断与仅保留早期读数（Early）的布局也劣于代表基线。ALFNet 的优势在本地证据部分存在时最为稳定，零历史（Unseen）条件下的泛化仍需改进。此外，实验使用单一随机种子，部分历史配置的保留曾参考测试结果，因此测试结果是既定研究流程下的最终评估，而非全程未触碰的独立确认；比较方法是共同输入与评价边界下的适配版本，SLPF 与 STBP 未执行其原始的多阶段或持续学习流程；速度与流量数据在窗口插值、解码残差倍率和中心化上使用不同的固定设置；机制分析在已知节点的验证窗口上进行，布局优势在验证集与测试集之间并不一致。后续工作可着重改进本地历史完全缺失时对空间证据的利用；当前解码阶段固定使用末步观测状态，未来也可引入对预测窗口内观测状态演变的建模。

## 参考文献

[1] Cini et al. Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks. ICLR, 2022.

[2] Liu et al. PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. ICDE, 2023.

[3] Zuo et al. Graph Convolutional Networks for Traffic Forecasting with Missing Values. Data Mining and Knowledge Discovery, 2023.

[4] Yu et al. GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing. KDD, 2024.

[5] Wu et al. Inductive Graph Neural Networks for Spatiotemporal Kriging. AAAI, 2021.

[6] Mei et al. Uncertainty-aware Traffic Prediction under Missing Data. ICDM, 2023.

[7] Zhou et al. MoGERNN: An Inductive Traffic Predictor for Unobserved Locations. Transportation Research Part C, 2025. doi:10.1016/j.trc.2025.105080.

[8] Liu et al. Spatio-temporal Partial Sensing Forecast of Long-term Traffic. TMLR, 2025.

[9] Zhao et al. Structure-inductive Meta-learning Empowers Partial-to-global Traffic State Forecasting in Dynamic Sensor Networks. Transportation Research Part C, 2026.

[10] Yu et al. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. IJCAI, 2018.

[11] Wu et al. Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI, 2019.

[12] Bai et al. Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. NeurIPS, 2020.

[13] Jiang et al. PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction. AAAI, 2023.

[14] Fang et al. Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective. KDD, 2025.

[15] Shao et al. Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting. CIKM, 2022.

[16] Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI, 2018.

[17] Liu et al. A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting. ICLR, 2026.

[18] Li, Yu, Shahabi, Liu. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR, 2018. arXiv:1707.01926.

[19] Cho et al. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP, 2014, pp. 1724–1734. doi:10.3115/v1/D14-1179.

[20] Zhang, Yan. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. ICLR, 2023.

[21] Shazeer et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR, 2017. arXiv:1701.06538.

[22] Wang et al. Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling. NeurIPS, 2023. arXiv:2304.02806.

[23] Guo et al. Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. AAAI, 2019, 33(01): 922–929. doi:10.1609/aaai.v33i01.3301922.

[24] Wang et al. Cold-Start Traffic State Forecasting at Unseen Sensor Locations via Support-Conditioned Meta-Graph Learning. Sensors, 2026.
