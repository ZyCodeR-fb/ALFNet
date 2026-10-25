## Contract Paraphrase

D1（methodology_rigor）：从方法严谨性的角度，这一维要求实验设计、数据划分与泄漏控制、基线公平性、随机性与不确定性的报告以及可复现信息达到领域的评审标准；由我负责评分。

D2（domain_accuracy）：要求主张与领域证据一致、对已有工作的描述准确；由领域审稿人负责，我不评分。

D3（argumentative_coherence）：要求核心论点内部一致、证据确实支撑主张；从方法学看，重点是每个机制性主张是否有能隔离该机制的实验；我与反方审稿人共同评分。

D4（cross_disciplinary_relevance）：要求框架与影响对相邻领域可理解且有依据；由跨领域审稿人负责，我不评分。

D5（writing_and_structure）：关注组织、清晰度、图表与体例；由编辑负责，我不评分。

D6（venue_fit_and_contribution）：关注与目标会议的契合度与贡献的原创性和意义；由编辑负责，我不评分。

## Scoring Plan

### D1: methodology_rigor
dimension_id: D1
what_to_look_for: 实验协议、数据划分与泄漏控制、基线公平性、随机性与不确定性报告、模型选择是否使用测试集、可复现信息
what_triggers_block: 核心比较结论只依赖单次运行且无方差估计，或测试集参与了配置选择而未能提供独立确认
what_triggers_warn: 基线调参预算或适配方式不对等，或评价指标的计算细节与局限披露不足
what_triggers_fatal: 存在训练使用测试信号等数据泄漏，或报告数值无法由所述协议产生

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: 每个机制性主张是否有对应的消融或干预支持，结论措辞是否超出证据范围
what_triggers_block: 核心机制主张没有能够隔离该机制的实验证据
what_triggers_warn: 部分结论措辞强于证据，或消融覆盖不完整而未加说明
what_triggers_fatal: 结论与文中报告的结果直接矛盾

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
