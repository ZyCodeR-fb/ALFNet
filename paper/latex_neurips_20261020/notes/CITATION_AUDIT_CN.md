# 引用审计报告（ARS citation-check，2026-10-20）

对象：`references.bib` 与 `sections/*.tex`（中文 LaTeX 工作稿）。引用格式按本任务约定覆盖 ARS 默认 APA 7：natbib 数字编号（`numbers,compress,sort`）＋ `plainnat`，因此 APA 的作者—年份格式条目不适用。元数据来源见 [CITATION_LOG_CN.md](CITATION_LOG_CN.md)。

## Summary

| 指标 | 数量 |
|---|---|
| 正文引用命令中的文献键（含重复） | 60 |
| 参考文献条目 | 24 |
| 正文引用但 bib 中不存在 | 0 |
| bib 中存在但未被引用 | 0 |
| 已自动修正的格式问题 | 2（见下） |
| 需人工复核的问题 | 3（见下） |
| 缺少 DOI 的条目 | 8（均为 ICLR/NeurIPS/TMLR 等不分配论文 DOI 的出版物） |
| 自引比例 | 0%（无作者既有工作） |
| 近 5 年（2021–2026）文献占比 | 16/24 = 66.7% |

检查方式：交叉引用由静态脚本完成（键名、未用条目）；作者、标题、会议/期刊、卷期页码与 DOI 逐条联网核对（arXiv 摘要页、AAAI OJS、ACL Anthology、TRID、PMC、TMLR BibTeX 导出、ICLR 官网）。DBLP 与 OpenReview 被拦截，未使用。DOI 只写入了在出版方或检索结果中确认过的值，未逐个打开 doi.org 做解析验证。

## Corrections Made

| # | 位置 | 问题 | 修正 |
|---|---|---|---|
| 1 | v6 参考文献 [17] STBP | 会议信息为"作者仓库标注 ICLR 2026" | ICLR 2026 官方页面（iclr.cc/virtual/2026/poster/10010050）确认，改为 ICLR 2026，作者 Aoyu Liu、Yaying Zhang |
| 2 | v5/v6 参考文献 | 只有"某某 et al."与网址说明 | `references.bib` 中补全全部作者、卷期页码与已确认 DOI |

## Items Flagged for Review

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| 1 | liu2023pristi、yu2024ginar、mei2023uncertainty、yu2018spatio、wu2019graph、fang2025efficient | 已解决：作者提供 DOI，经 Crossref 记录核对标题、作者与会议一致，并补入页码（1927–1939、3989–4000、1223–1228、3634–3640、1907–1913、307–317） | — |
| 2 | cho2014learning（2014） | 超过 10 年 | 属于 GRU 的原始出处，作为基础文献保留 |
| 3 | 全部期刊条目 | 未对照 Retraction Watch 数据库做撤稿筛查（本轮无法访问） | 投稿前补做；本轮不声称已排除撤稿 |

## 引用与原文主张的一致性抽查

对照本轮读取的摘要或出版页，正文中的描述与原文一致：GinAR（插值注意力＋自适应图卷积）、PatchSTG（KD-tree 空间分块）、SC-SAMG（支持期与一阶元学习）、SIML-MSIGN（掩码门控归纳扩散与元学习）、MoGERNN（图专家混合、动态传感网络）、UIGNN（无历史位置预测与不确定性）、STBP（可扩展上下文模式库）、DCRNN（METR-LA、PEMS-BAY 数据来源）、ASTGCN（PeMS 数据来源）、Shazeer et al.（CV² 平衡项来源）。GCN-M"需要12步窗口外历史记忆"来自项目内部适配审计（`docs/comparison_model_audit.md` 2026-09-15 段），与其摘要中"注意力记忆网络利用全局历史模式"相符，但窗口长度这一具体判断出自本项目的协议核对，不是原文的表述。
