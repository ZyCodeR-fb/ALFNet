# 新增引用查证记录（阶段 2，2026-10-20）

规则：只用 v5 草稿或相关工作证据中已出现的条目，或经联网核对的条目；不凭记忆写 DOI、页码或作者。DBLP 与 OpenReview 本轮访问被拦截（429/403/浏览器验证），改用 arXiv 摘要页、出版方页面、ACL Anthology 与 ML Anthology 交叉核对。阶段 3 生成 `references.bib` 后再运行 ARS citation-check 作正式审计。

注：下表为删除 ST-FiT 之前的编号。删除原 [8] 后，[19]–[25] 依次变为 [18]–[24]；LaTeX 中使用 bib 键名（如 `li2018diffusion`），编号由 natbib 自动生成。

| v6 编号 | 条目 | 核对来源 | 结果 |
|---|---|---|---|
| [19] | Li, Yu, Shahabi, Liu. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR 2018 | arXiv:1707.01926 摘要页（注明 "Published as a conference paper at ICLR 2018"） | 已核：标题、四位作者、会议年份 |
| [20] | Cho, van Merriënboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP 2014, pp. 1724–1734 | ACL Anthology D14-1179 | 已核：标题、七位作者、页码、DOI 10.3115/v1/D14-1179 |
| [21] | Zhang, Yan. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. ICLR 2023 | ML Anthology（指向 OpenReview vSVLM2j9eie）；检索结果 | 已核：标题、两位作者、会议年份；无页码（OpenReview 会议论文） |
| [22] | Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017 | arXiv:1701.06538（七位作者）；Google Research 的 ICLR 2017 论文列表 | 已核：标题、作者；会议年份由 Google Research 页面佐证，arXiv 页未写会议 |
| [23] | Wang, Jiang, You, Han, Liu, Srinivasa, Kompella, Wang. Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling. NeurIPS 2023 | arXiv:2304.02806（注明 NeurIPS 2023） | 已核 |
| [24] | Guo, Lin, Feng, Song, Wan. Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. AAAI 2019, 33(01): 922–929 | AAAI OJS 3881 | 已核：标题、五位作者、卷期、页码、DOI 10.1609/aaai.v33i01.3301922 |
| [25] | Wang et al. Cold-Start Traffic State Forecasting at Unseen Sensor Locations via Support-Conditioned Meta-Graph Learning. Sensors 2026 | 来自 `paper/related_work_cn.md` 与 `related_work_evidence_cn.md`（出版页 mdpi.com/1424-8220/26/15/4995） | 沿用相关工作证据；本轮未重新联网核对，阶段 3 citation-check 时补核 |

v5 原有 [1]–[18] 沿用 v5 与相关工作证据中的信息，本轮未重新核对，阶段 3 一并进入 citation-check。[18] STBP 的会议信息仍为"作者仓库标注 ICLR 2026"，需在 citation-check 中确认。

## 阶段 3 补充核对（2026-10-20）

生成 `references.bib` 时对 v5 原有条目逐条联网核对了完整作者、出处、卷期页码与 DOI（来源：arXiv 摘要页、AAAI OJS、TRID、PMC、TMLR BibTeX 导出、ICLR 官网、Springer 检索结果）。STBP 经 ICLR 2026 官方页面确认；SC-SAMG 经 PMC 确认作者与 DOI 10.3390/s26154995。审计结果见 [CITATION_AUDIT_CN.md](CITATION_AUDIT_CN.md)。
