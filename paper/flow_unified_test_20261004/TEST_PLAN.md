# 统一机制候选测试：2026-10-04

用户授权本轮分析及合适候选test。10项训练结束，08三方案达到27/28/29格并且三项LifeAvg优于八基线；04新方案尚未达到27格，同一方案两flow同时达标尚未实现。

固定五项：04/08 triple配对；04/08 triple_spatial配对；08 triple_halfcenter。08triple是最低LifeAvg MAPE，halfcenter最多第一格，spatial最低LifeAvg MAE。04为检查共同机制保留triple和spatial，不能将04称为已经达标；不测试04halfcenter及其他无明确优势的组合。旧04bridge_delta1/center、08bridge_delta1/bridge_center和两原版test均只复用。

先准确使用experiments.flow_unified.adapter重建并核对方法合同、配置、checkpoint/scaler/暴露哈希；完整33validation重放一致才开放test，每项一次。最多3并发评估，不训练、不改指标阈值或掩码、不自动追加候选。全部528validation、11组363test格整理为主表/阶段表。

首选统一计算规则，允许validation所选超参数差异；如果配对方案不能两数据集达标，不声称统一成功。比较原ALFNet时披露LifeAvg取舍，不把外部基线第一等同于原模型三项全面改善。历史test已参与研发，本轮是探索性比较，不是未见test的独立确认。

正式四表暂不覆盖，speed/基线不重训；达到可保留共同机制时，再按已有speed迁移授权生成独立训练计划。PEMS03不在范围。无提交/推送/数据删除。
