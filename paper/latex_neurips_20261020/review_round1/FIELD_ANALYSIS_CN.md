# Field Analysis Report（Phase 0）

说明：配置在写各席位 Phase 1 预承诺之前确定，本文件在评审完成后整理成文。因用户已要求直接进入阶段 4，配置卡未单独停下来请用户确认；如需调整审稿人身份，可按新配置重跑本轮。

## Paper Basic Information
- **Title**: ALFNet：动态观测支持下的归纳时空预测
- **Abstract length**: 中文约 600 字
- **Full text length**: 约 15,500（中文字符与英文词合计，见 `metadata.json`）
- **Number of references**: 24

## Field Analysis

| Dimension | Analysis Result |
|---|---|
| Primary Discipline | 机器学习：时空图预测 |
| Secondary Disciplines | 智能交通系统、缺失数据学习、归纳图学习 |
| Research Paradigm | 实证—方法型（提出模型并在基准上评价） |
| Methodology Type | 深度学习模型设计＋对比实验、组件消融、配对输入干预 |
| Target Journal Tier | 顶级 ML 会议（NeurIPS 风格），应用型时空学习方向 |
| Paper Maturity | 投稿前中文工作稿：实验完整，正文篇幅与表述尚未按会议格式压缩 |

## Recommended Target Journals (Top 3)
1. NeurIPS — 作者选定的样式与目标
2. ICLR / KDD — 时空图学习论文常见去处，迁移成本低
3. IEEE T-ITS / Transportation Research Part C — 若强调交通应用与协议

## Reviewer Configuration Cards

### Reviewer Configuration Card #1
**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: ML 会议应用型时空学习方向的领域主席，长期评审图神经网络与时间序列论文
**Review Focus**:
  1. 贡献是否超出已有组件的组合，是否有可迁移的原则
  2. 篇幅、图表与符号是否符合会议体例
  3. 标题、摘要与主要证据的定位是否一致
**Will particularly care about**: 方法层面的新认识，以及边界结果是否被如实呈现
**Possible blind spots**: 统计细节与基线适配细节

### Reviewer Configuration Card #2
**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1
**Identity Description**: 机器学习实验方法学与评测协议研究者，关注可复现性与选择偏差
**Review Focus**:
  1. 随机性、方差与选择效应（测试集是否参与选择）
  2. 基线调参预算与适配方式的公平性
  3. 消融能否隔离所声称的机制
**Will particularly care about**: 单次运行下计数式结论的可靠性
**Possible blind spots**: 交通领域惯例与部署问题

### Reviewer Configuration Card #3
**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2
**Identity Description**: 交通时空预测与归纳克里金方向研究者，熟悉 METR-LA、PEMS 系列基准
**Review Focus**:
  1. 比较方法是否覆盖领域标准基线
  2. 相关工作的覆盖与表述准确性
  3. 协议与领域常见设置的可比性
**Will particularly care about**: 自适应图方法与近期归纳预测工作是否被比较或讨论
**Possible blind spots**: 统计严谨性

### Reviewer Configuration Card #4
**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3
**Identity Description**: 传感网络运维与部署方向研究者，关注方法在真实系统中的可用性
**Review Focus**:
  1. 推理时所需输入（角色、状态年龄）在部署中的可得性
  2. 计算代价
  3. 向其他传感网络的迁移与适用范围
**Will particularly care about**: 条件信息有误时方法是否仍然可靠
**Possible blind spots**: 模型结构细节

第五席为固定的 Devil's Advocate，不使用动态配置卡。

## Review Strategy Recommendations
- 本文的核心是"以观测支持作为计算条件"的主张，方法学审稿人与 DA 需重点检验这一主张是否被消融隔离。
- 作者自评风险：本轮五个席位与稿件作者为同一模型、同一会话，详见 `panel-provenance.json`，评审意见的独立性有限。
