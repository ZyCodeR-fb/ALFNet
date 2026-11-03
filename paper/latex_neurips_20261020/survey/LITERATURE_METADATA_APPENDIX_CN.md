# 文献元数据附表（与 LITERATURE_MATRIX_CN.md v2 配套）

- 编写日期：2026-11-03。由 `verification_20261103/build_appendix.py` 依据 Crossref 记录（`verification_20261103/cr_all.json`）生成，无 DOI 的条目为手工填写并逐条注明来源。
- 范围：矩阵中全部编号条目。共 89 行：22 张核心卡片中 UniSTOK 分 v1、v2 两行；扩展条目 E01–E67 中 E10 已升级为 C22，不再单列。对应 **88 篇唯一论文、89 个被引用版本**。
- 作者：有 Crossref/DataCite 记录的取其完整作者表；其余取 arXiv 摘要页、官方论文集或会议页面；只有 Semantic Scholar 一路来源的已在"未核字段"注明。
- 正式 DOI 与 arXiv：本表"正式 DOI"一列只填出版方 DOI（Crossref 或 DataCite 注册）；arXiv 自动分配的 `10.48550/arXiv.*` DataCite DOI 不填入此列。

## 1. 元数据核实路径代号

| 代号 | 含义 | 是否计作独立来源 |
|---|---|---|
| CR | Crossref REST API 解析 DOI，核对题名、作者、出处、卷期页、日期 | 是 |
| DC | DataCite API 与 doi.org 跳转（用于 OpenProceedings 等 DataCite 注册的 DOI） | 是 |
| PUB | 出版方或会议的官方页面或 PDF（PMLR、ICLR/NeurIPS 论文集、会议虚拟页面、OpenProceedings） | 是 |
| AX | arXiv 摘要页或 HTML 全文（含版本历史、作者注释与 journal-ref） | 是（作者自报的出处信息另行注明） |
| LOC | 本地全文 `references/paper/*.md` | 是 |
| S2 | Semantic Scholar Graph API | 是，但它是聚合数据库；其返回的 DBLP 键属于同一条记录，**不另计为 DBLP 来源** |
| OpenAlex | OpenAlex API（只用于取摘要） | 否（只作内容来源） |
| OLD | 旧审计文件已记录、本轮未重核 | 否 |

2026-11-02 第一轮中，Semantic Scholar 对 FUNS（arXiv:2211.11596）与 IBN（arXiv:2509.07725）的查询只返回 429，当时实际依靠 arXiv 摘要页核实。v1 矩阵第 9.1 节"E01–E57 均至少两路核对"的说法不准确，已在 v2 中撤回，以本表逐条路径为准。

## 2. 逐条元数据

| 编号 | 完整题名 | 作者 | 正式出处（年份） | 正式 DOI | arXiv（版本） | 官方来源 | 元数据核实路径 | 访问日期 | 未核字段与备注 |
|---|---|---|---|---|---|---|---|---|---|
| C01 | MoGERNN: An inductive traffic predictor for unobserved locations | Qishen Zhou, Yifan Zhang, Michail A. Makridis, Anastasios Kouvelas, Yibing Wang, Simon Hu | Transportation Research Part C: Emerging Technologies, 174, 105080（2025） | 10.1016/j.trc.2025.105080 | 2501.12281（预印本题名多 "in Dynamic Sensing Networks"） | https://doi.org/10.1016/j.trc.2025.105080 | CR+S2+LOC | 2026-11-02 | arXiv 版本未逐一核对 |
| C02 | Structure-inductive meta-learning empowers partial-to-global traffic state forecasting in dynamic sensor networks | Yan Zhao, Can Wang, Yikang Rui, Wenqi Lu, Bin Ran | Transportation Research Part C: Emerging Technologies, 185, 105564（2026） | 10.1016/j.trc.2026.105564 | 无 | https://doi.org/10.1016/j.trc.2026.105564 | CR+S2+LOC | 2026-11-02 | — |
| C03 | Uncertainty-aware Traffic Prediction under Missing Data | Hao Mei, Junxian Li, Zhiming Liang, Guanjie Zheng, Bin Shi, Hua Wei | 2023 IEEE International Conference on Data Mining (ICDM), 1223-1228（2023） | 10.1109/ICDM58522.2023.00152 | 2309.06800v5（长版） | IEEE Xplore 10415744 | CR+S2+AX | 2026-11-02 | — |
| C04 | ST-FiT: Inductive Spatial-Temporal Forecasting with Limited Training Data | Zhenyu Lei, Yushun Dong, Jundong Li, Chen Chen | Proceedings of the AAAI Conference on Artificial Intelligence, 39, 12031-12039（2025） | 10.1609/aaai.v39i11.33310 | 2412.10912v2 | AAAI OJS 33310 | CR+S2+AX | 2026-11-02 | — |
| C05 | Spatial-temporal Forecasting for Regions without Observations | Xinyu Su, Jianzhong Qi, Egemen Tanin, Yanchuan Chang, Majid Sarvi | EDBT 2024, pp. 488–（首页页码） | 10.48786/edbt.2024.42 | 2401.10518v1 | https://openproceedings.org/2024/conf/edbt/paper-123.pdf | DC+PUB(PDF 首页)+S2+AX | 2026-11-03 | DOI 为 DataCite 注册（OpenProceedings），Crossref 无记录属正常；终止页码未核 |
| C06 | Generalising Traffic Forecasting to Regions Without Traffic Observations | Xinyu Su, Majid Sarvi, Feng Liu, Egemen Tanin, Jianzhong Qi | Proceedings of the AAAI Conference on Artificial Intelligence, 40, 15761-15769（2026） | 10.1609/aaai.v40i18.38607 | 2508.08947v2 | AAAI OJS 38607 | CR+S2+AX | 2026-11-02 | — |
| C07 | Cold-Start Traffic State Forecasting at Unseen Sensor Locations via Support-Conditioned Meta-Graph Learning | Can Wang, Zhiyu Wang, Weijie Wang, Jing Gan, Yanni Ju | Sensors, 26, 4995（2026） | 10.3390/s26154995 | 无 | MDPI（页面 403） | CR+S2+OpenAlex | 2026-11-02 | 全文未读 |
| C08 | Spatio-Temporal Partial Sensing Forecast for Long-term Traffic（arXiv 题名；mlanthology 记为 "…of Long-Term Traffic"） | Zibo Liu, Zhe Jiang, Zelin Xu, Tingsong Xiao, Zhengkun Xiao, Yupu Zhang, Haibo Wang, Shigang Chen | Transactions on Machine Learning Research, 2025 | 无 | 2408.02689v3（2026-09-30；v1 2024-08-02，v2 2025-08-08） | OpenReview Ff08aPjVjD（v3 正文所载） | AX(注释与 journal-ref)+S2+mlanthology | 2026-11-03 | TMLR 无 DOI；OpenReview 论坛页本轮未直接读取 |
| C09 | GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing | Chengqing Yu, Fei Wang, Zezhi Shao, Tangwen Qian, Zhao Zhang, Wei Wei, Yongjun Xu | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 3989-4000（2024） | 10.1145/3637528.3672055 | 2405.11333v1 | ACM DL | CR+S2+AX | 2026-11-02 | — |
| C10 | GinAR+: A Robust End-to-End Framework for Multivariate Time Series Forecasting With Missing Values | Chengqing Yu, Fei Wang, Zezhi Shao, Tangwen Qian, Zhao Zhang, Wei Wei, Zhulin An, Qi Wang, Yongjun Xu | IEEE Transactions on Knowledge and Data Engineering, 37, 4635-4648（2025） | 10.1109/TKDE.2025.3569649 | 无 | IEEE Xplore 11002729 | CR+S2+OpenAlex | 2026-11-02 | 全文未读 |
| C11 | Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates | Chengqing Yu, Fei Wang, Chuanguang Yang, Zezhi Shao, Tao Sun, Tangwen Qian, Wei Wei, Zhulin An, Yongjun Xu | Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 3633-3644（2025） | 10.1145/3711896.3737046 | 2506.12459v1 | ACM DL | CR+S2+AX | 2026-11-02 | — |
| C12 | Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling | Ivan Marisca, Cesare Alippi, Filippo Maria Bianchi | ICML 2024, PMLR 235, pp. 34846–34865 | 无 | 2402.10634（读 v2 附录 F、v3 正文） | https://proceedings.mlr.press/v235/marisca24a.html | PUB(PMLR)+S2+AX | 2026-11-03 | ICML 无正式 DOI |
| C13 | MAS‐Mamba: A Compact Missing‐Aware Selective State‐Space Model With Calibrated Uncertainty for Short‐Term Traffic Forecasting Under Sensor Outages | Chung‐I Huang, Wen‐Yi Chang | IET Intelligent Transport Systems, 20, e70304（2026） | 10.1049/itr2.70304 | 无 | Wiley/IET | CR+S2+OpenAlex | 2026-11-02 | 全文未读 |
| C14 | Graph convolutional networks for traffic forecasting with missing values | Jingwei Zuo, Karine Zeitouni, Yehia Taher, Sandra Garcia-Rodriguez | Data Mining and Knowledge Discovery, 37, 913-947（2022） | 10.1007/s10618-022-00903-7 | 2212.06419 | Springer | CR+S2 | 2026-11-02 | — |
| C15 | KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy | Qianxiong Xu, Cheng Long, Ziyue Li, Sijie Ruan, Rui Zhao, Zhishuai Li | Proceedings of the AAAI Conference on Artificial Intelligence, 39, 12945-12953（2025） | 10.1609/aaai.v39i12.33412 | 2311.02565v2 | AAAI OJS 33412 | CR+S2+AX | 2026-11-02 | — |
| C16a | UniSTOK: Uniform Inductive Spatio-Temporal Kriging | Lewei Xie, Haoyu Zhang, Juan Yuan, Liangjun You, Yulong Chen, Yifan Zhang | arXiv 预印本 v1 | 无 | 2603.05301v1（2026-03-05） | arXiv | AX+S2 | 2026-11-02 | 预印本；v1 与 v2 题名、作者、方法均不同，引用须注明版本 |
| C16b | Uniform Inductive Spatio-Temporal Kriging | Lewei Xie, Haoyu Zhang, Yulong Chen, Liangjun You, Zongxian Yang, Yifan Zhang | arXiv 预印本 v2 | 无 | 2603.05301v2（2026-05-09） | arXiv | AX | 2026-11-03 | 预印本；当前最新版 |
| C17 | TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts | Hyunwook Lee, Sungahn Ko | ICLR 2024 | 无 | 2403.02600（注释：ICLR 2024） | arXiv 摘要页；代码页 | AX+S2 | 2026-11-02 | ICLR 永久页未读；无正式 DOI |
| C18 | From XXLTraffic to EvoXXLTraffic: Scaling Traffic Forecasting to Sensor-Evolving Networks | Du Yin, Hao Xue, Arian Prabowo, Shuang Ao, Flora Salim | arXiv 预印本 | 无 | 2605.29768v2（2026-06） | arXiv | AX+S2 | 2026-11-02 | 预印本 |
| C19 | Frigate: Frugal Spatio-temporal Forecasting on Road Networks | Mridul Gupta, Hariprasad Kodamana, Sayan Ranu | Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 649-660（2023） | 10.1145/3580305.3599357 | 2306.08277v1 | ACM DL；KDD 2023 目录 | CR+AX | 2026-11-03 | 代码仓库 BibTeX 中 DOI 前缀有误（据 Codex 报告，未独立打开 BibTeX） |
| C20 | Traffic forecasting on new roads using spatial contrastive pre-training (SCPT) | Arian Prabowo, Hao Xue, Wei Shao, Piotr Koniusz, Flora D. Salim | Data Mining and Knowledge Discovery, 38, 913-937（2023） | 10.1007/s10618-023-00982-0 | 2305.05237（v4，2023-09-21） | Springer | CR+AX | 2026-11-03 | — |
| C21 | Robust Spatio-Temporal Centralized Interaction for OOD Learning | Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, Yang Wang | ICML 2025, PMLR 267, pp. 42165–42192 | 无 | 未核 | https://proceedings.mlr.press/v267/ma25s.html | PUB(PMLR 页与 PDF) | 2026-11-03 | ICML 无正式 DOI；arXiv 号未核 |
| C22 | Forecasting Unobserved Node States with spatio-temporal Graph Neural Networks | Andreas Roth, Thomas Liebig | 2022 IEEE International Conference on Data Mining Workshops (ICDMW), 740-747（2022） | 10.1109/ICDMW58026.2022.00101 | 2211.11596 | IEEE Xplore | CR+AX | 2026-11-03 | ICDM Workshops，不是 ICDM 主会 |
| E01 | Inductive Graph Neural Networks for Spatiotemporal Kriging | Yuankai Wu, Dingyi Zhuang, Aurelie Labbe, Lijun Sun | Proceedings of the AAAI Conference on Artificial Intelligence, 35, 4478-4485（2021） | 10.1609/aaai.v35i5.16575 | 2006.07527 | AAAI OJS | CR+S2 | 2026-11-02 | — |
| E02 | INCREASE: Inductive Graph Representation Learning for Spatio-Temporal Kriging | Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, Jianzhong Qi, Chaochao Chen, Longbiao Chen | Proceedings of the ACM Web Conference 2023, 673-683（2023） | 10.1145/3543507.3583525 | 2302.02738 | ACM DL | CR+S2 | 2026-11-02 | — |
| E03 | Graph Neural Processes for Spatio-Temporal Extrapolation | Junfeng Hu, Yuxuan Liang, Zhencheng Fan, Hongyang Chen, Yu Zheng, Roger Zimmermann | Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 752-763（2023） | 10.1145/3580305.3599372 | 2305.18719 | ACM DL | CR+S2 | 2026-11-02 | — |
| E04 | Decoupling Long- and Short-Term Patterns in Spatiotemporal Inference | Junfeng Hu, Yuxuan Liang, Zhencheng Fan, Li Liu, Yifang Yin, Roger Zimmermann | IEEE Transactions on Neural Networks and Learning Systems, 35, 16328-16340（2024） | 10.1109/TNNLS.2023.3293814 | 2109.09506 | IEEE Xplore | CR+S2 | 2026-11-02 | — |
| E05 | Inductive and adaptive graph convolution networks equipped with constraint task for spatial–temporal traffic data kriging | Tonglong Wei, Youfang Lin, Shengnan Guo, Yan Lin, Yiji Zhao, Xiyuan Jin, Zhihao Wu, Huaiyu Wan | Knowledge-Based Systems, 284, 111325（2024） | 10.1016/j.knosys.2023.111325 | 无 | Elsevier | CR+S2 | 2026-11-02 | 摘要未取得 |
| E06 | AnchorGK: Anchor-based Incremental and Stratified Graph Learning Framework for Inductive Spatio-Temporal Kriging | Xiaobin Ren, Kaiqi Zhao, Katerina Taškova, Patricia Riddle | Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, 1251-1262（2026） | 10.1145/3770854.3780189 | 2512.21569 | ACM DL | CR+S2+AX | 2026-11-02 | — |
| E07 | Spatial Aggregation and Temporal Convolution Networks for Real-time Kriging | Yuankai Wu, Dingyi Zhuang, Mengying Lei, Aurélie Labbe, Lijun Sun | arXiv 预印本 | 无 | 2109.12144 | arXiv | S2 | 2026-11-02 | 只有单一聚合来源；预印本 |
| E08 | Inductive Spatio-Temporal Kriging with Physics-Guided Increment Training Strategy for Air Quality Inference | Songlin Yang, Tao Yang, Bo Hu | arXiv 预印本 | 无 | 2503.09646 | arXiv | S2 | 2026-11-02 | 单一聚合来源；预印本 |
| E09 | A Hybrid Inductive-Transductive Network for Traffic Flow Imputation on Unsampled Locations | Mohammadmahdi Rahimiasl, Ynte Vanderhoydonc, Siegfried Mercelis | arXiv 预印本 | 无 | 2512.17984 | arXiv | S2 | 2026-11-02 | 单一聚合来源；预印本 |
| E11 | Network-Wide Freeway Traffic Estimation Using Sparse Sensor Data: A Dirichlet Graph Auto-Encoder Approach | Qishen Zhou, Yifan Zhang, Michail A. Makridis, Anastasios Kouvelas, Yibing Wang, Simon Hu | IEEE Transactions on Intelligent Transportation Systems, 26, 22161-22177（2025） | 10.1109/TITS.2025.3610911 | 2503.15845 | IEEE Xplore | CR+S2 | 2026-11-02 | — |
| E12 | SUSTeR: Sparse Unstructured Spatio Temporal Reconstruction on Traffic Prediction | Yannick Wölker, Christian Beth, Matthias Renz, Arne Biastoch | Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems, 1-10（2023） | 10.1145/3589132.3625631 | 2502.16935 | ACM DL | CR+S2 | 2026-11-02 | — |
| E13 | Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks | Andrea Cini, Ivan Marisca, Cesare Alippi | ICLR 2022 | 无 | 2108.00298 | OpenReview kOu3-S3wJ7（旧文件） | S2+OLD | 2026-11-02 | ICLR 永久页本轮未读 |
| E14 | PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation | Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu | 2023 IEEE 39th International Conference on Data Engineering (ICDE), 1927-1939（2023） | 10.1109/ICDE55515.2023.00150 | 2302.09746 | IEEE Xplore | CR+S2 | 2026-11-02 | — |
| E15 | Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations | Ivan Marisca, Andrea Cini, Cesare Alippi | Advances in Neural Information Processing Systems 35, 32069-32082（2022） | 10.52202/068431-2324 | 2205.13479 | NeurIPS 官方页 | CR+S2 | 2026-11-03 | — |
| E16 | ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation | Tong Nie, Guoyang Qin, Wei Ma, Yuewen Mei, Jian Sun | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2260-2271（2024） | 10.1145/3637528.3671751 | 2312.01728 | ACM DL | CR+S2 | 2026-11-02 | — |
| E17 | STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation | Yiming Wang, Hao Peng, Senzhang Wang, Haohua Du, Chunyang Liu, Jia Wu, Guanlin Wu | Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 3435-3443（2025） | 10.24963/ijcai.2025/382 | 2506.08054 | IJCAI 论文集 | CR+S2 | 2026-11-02 | — |
| E18 | Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation | Xiaowei Mao, Huihu Ding, Yan Lin, Tingrui Wu, Shengnan Guo, Dazhuo Qiu, Feiling Fang, Jilin Hu, Huaiyu Wan | Proceedings of the AAAI Conference on Artificial Intelligence, 40, 15528-15536（2026） | 10.1609/aaai.v40i18.38581 | 2601.04572 | AAAI OJS 38581 | CR+S2 | 2026-11-02 | — |
| E19 | Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values | Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li（S2 记录） | ICLR 2024 | 无 | 无 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html | PUB(旧审计)+S2 | 2026-11-02 | — |
| E20 | S4M: S4 for multivariate time series forecasting with Missing values | Jing Peng, Meiqi Yang, Qiong Zhang, Xiaoxiao Li | ICLR 2025 | 无 | 2503.00900 | arXiv 摘要页 | AX+S2 | 2026-11-02 | 第二作者 arXiv 为 Meiqi Yang，S2 为 Meiqing Yang，未裁定；ICLR 永久页未读 |
| E21 | GeoMAE : Masking representation learning for spatio-temporal graph forecasting with missing values | Songyu Ke, Chenyu Wu, Yuxuan Liang, Huiling Qin, Junbo Zhang, Yu Zheng | Neural Networks, 202, 108986（2026） | 10.1016/j.neunet.2026.108986 | 2508.14083 | Elsevier | CR+S2 | 2026-11-02 | 摘要未取得 |
| E22 | A Unified Diffusion Framework for Traffic Imputation and Prediction With Physical Priors | Peng Liu, Yaodong Zhu, Yang Yang, Caixia Wang, Jingfeng Jie | IEEE Transactions on Mobile Computing, 25, 341-357（2026） | 10.1109/TMC.2025.3591423 | 无 | IEEE Xplore 11087699 | CR+S2 | 2026-11-02 | — |
| E23 | IBN: An Interpretable Bidirectional-Modeling Network for Multivariate Time Series Forecasting with Variable Missing | Shusen Ma, Tianhao Zhang, Qijiu Xia, Yun-Bo Zhao | arXiv 预印本 | 无 | 2509.07725 | arXiv 摘要页 | AX | 2026-11-02 | 单一来源；预印本；正式录用未见 |
| E24 | Recurrent Neural Networks for Multivariate Time Series with Missing Values | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu | Scientific Reports, 8, 6085（2018） | 10.1038/s41598-018-24271-9 | 1606.01865 | Nature | CR+S2 | 2026-11-02 | — |
| E25 | Modeling Information Blackouts in Missing Not-At-Random Time Series Data | Aman Sunesh, Alan Ma, Siddarth Nilol | arXiv 预印本 | 无 | 2601.01480 | arXiv | S2 | 2026-11-02 | 单一聚合来源；预印本；只有检索片段 |
| E26 | Taming Local Effects in Graph-based Spatiotemporal Forecasting | Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi | Advances in Neural Information Processing Systems 36, 55375-55393（2023） | 10.52202/075280-2417 | 2302.04071 | NeurIPS 官方页 | CR+S2 | 2026-11-03 | — |
| E27 | Prompt-Based Spatio-Temporal Graph Transfer Learning | Junfeng Hu, Xu Liu, Zhencheng Fan, Yifang Yin, Shili Xiang, Savitha Ramasamy, Roger Zimmermann | Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 890-899（2024） | 10.1145/3627673.3679554 | 2405.12452 | ACM DL | CR+S2 | 2026-11-02 | — |
| E28 | Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation | Yuan Yuan, Chenyang Shao, Jingtao Ding, Depeng Jin 及另 1 位（S2 记录 5 人） | ICLR 2024 | 无 | 2402.11922 | arXiv | S2 | 2026-11-02 | 单一聚合来源；ICLR 永久页未读；作者全列表未二次核对 |
| E29 | Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank | Zhanyu Liu, Guanjie Zheng, Yanwei Yu | Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 1451-1460（2023） | 10.1145/3583780.3614829 | 2308.09727 | ACM DL | CR+S2 | 2026-11-02 | — |
| E30 | UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction | Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, Yong Li | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4095-4106（2024） | 10.1145/3637528.3671662 | 2402.11838 | ACM DL | CR+S2 | 2026-11-02 | — |
| E31 | Open Spatio-Temporal Foundation Models for Traffic Prediction | Zhonghang Li, Long Xia, Lei Shi, Yong Xu, Dawei Yin, Chao Huang | ACM Transactions on Intelligent Systems and Technology, 3773912（2025） | 10.1145/3773912 | 2408.10269 | ACM DL | CR+S2 | 2026-11-02 | — |
| E32 | FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction | Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang | ICML 2024 | 无 | 2405.17898 | arXiv | S2 | 2026-11-02 | 单一聚合来源；ICML 永久页未读 |
| E33 | TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning | Xu Chen, Junshan Wang, Kunqing Xie | Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 3620-3626（2021） | 10.24963/ijcai.2021/498 | 2106.06273 | IJCAI 论文集 | CR+S2 | 2026-11-02 | — |
| E34 | Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction | Binwu Wang, Yudong Zhang, Xu Wang, Pengkun Wang, Zhengyang Zhou, Lei Bai, Yang Wang | Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2223-2232（2023） | 10.1145/3580305.3599463 | 无 | ACM DL | CR | 2026-11-02 | — |
| E35 | Knowledge Expansion and Consolidation for Continual Traffic Prediction With Expanding Graphs | Binwu Wang, Yudong Zhang, Jiahao Shi, Pengkun Wang, Xu Wang, Lei Bai, Yang Wang | IEEE Transactions on Intelligent Transportation Systems, 24, 7190-7201（2023） | 10.1109/TITS.2023.3263904 | 无 | IEEE Xplore | CR+S2 | 2026-11-02 | — |
| E36 | A Unified Replay-Based Continuous Learning Framework for Spatio-Temporal Prediction on Streaming Data | Hao Miao, Yan Zhao, Chenjuan Guo, Bin Yang, Kai Zheng, Feiteng Huang, Jiandong Xie, Christian S. Jensen | 2024 IEEE 40th International Conference on Data Engineering (ICDE), 1050-1062（2024） | 10.1109/ICDE60146.2024.00085 | 2404.14999 | IEEE Xplore | CR+S2 | 2026-11-02 | — |
| E37 | Continual Traffic Forecasting via Mixture of Experts | Sanghyun Lee, Chanyoung Park | arXiv 预印本 | 无 | 2406.03140 | arXiv | S2 | 2026-11-02 | 单一聚合来源；预印本 |
| E38 | Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting | Wei Chen, Yuxuan Liang | ICLR 2025 | 无 | 2410.12593（注释：ICLR 2025 录用） | arXiv 摘要页 | AX+S2 | 2026-11-02 | ICLR 永久页未读 |
| E39 | A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting | Aoyu Liu, Yaying Zhang | ICLR 2026 | 无 | 未核 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/7547048d73ed056286762b17434466a7-Abstract-Conference.html | PUB | 2026-11-02 | 单一官方来源；S2 未命中 |
| E40 | STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts | Binwu Wang, Jiaming Ma, Pengkun Wang, Xu Wang, Yudong Zhang, Zhengyang Zhou, Yang Wang | Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2948-2959（2024） | 10.1145/3637528.3671680 | 未核 | ACM DL | CR+S2 | 2026-11-02 | — |
| E41 | Robust Traffic Forecasting against Spatial Shift over Years | Hongjun Wang, Jiyuan Chen, Tong Pan, Zheng Dong, Lingyu Zhang, Renhe Jiang, Xuan Song | arXiv 预印本 | 无 | 2410.00373 | arXiv | S2 | 2026-11-02 | 单一聚合来源；预印本；作者全列表未二次核对 |
| E42 | STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization | Haoyu Zhang, Wentao Zhang, Hao Miao, Xinke Jiang, Yuchen Fang, Yifan Zhang | Advances in Neural Information Processing Systems 38, 130763-130798（2025） | 10.52202/085713-3934 | 2505.19547v2 | NeurIPS 官方页 | CR+S2 | 2026-11-03 | — |
| E43 | XXLTraffic: Expanding and Extremely Long Traffic Forecasting beyond Test Adaptation | Du Yin, Hao Xue, Arian Prabowo, Shuang Ao, Flora Salim | Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems, 511-521（2025） | 10.1145/3748636.3762762 | 2406.12693（arXiv 题名不同） | ACM DL | CR+S2 | 2026-11-02 | — |
| E44 | Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling | Haotao Wang, Ziyu Jiang, Yuning You, Yan Han, Gaowen Liu, Jayanth Srinivasa, Ramana Kompella, Zhangyang "atlas" Wang | Advances in Neural Information Processing Systems 36, 50825-50837（2023） | 10.52202/075280-2211 | 2304.02806 | NeurIPS 官方页 | CR+S2 | 2026-11-03 | — |
| E45 | Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts | Sukwon Yun, Inyoung Choi, Jie Peng, Yangfan Wu, Jingxuan Bao, Qiyiwen Zhang, Jiayi Xin, Qi Long, Tianlong Chen | Advances in Neural Information Processing Systems 37, 98782-98805（2024） | 10.52202/079017-3135 | 2410.08245 | NeurIPS 官方页 | CR+S2 | 2026-11-03 | — |
| E46 | FiLM: Visual Reasoning with a General Conditioning Layer | Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, Aaron Courville | Proceedings of the AAAI Conference on Artificial Intelligence, 32（2018） | 10.1609/aaai.v32i1.11671 | 1709.07871 | AAAI OJS | CR+S2 | 2026-11-02 | — |
| E47 | Spatio-Temporal Meta-Graph Learning for Traffic Forecasting | Renhe Jiang, Zhaonan Wang, Jiawei Yong, Puneet Jeph, Quanjun Chen, Yasumasa Kobayashi, Xuan Song, Shintaro Fukushima, Toyotaro Suzumura | Proceedings of the AAAI Conference on Artificial Intelligence, 37, 8078-8086（2023） | 10.1609/aaai.v37i7.25976 | 2211.14701 | AAAI OJS | CR+S2 | 2026-11-02 | — |
| E48 | Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution | Fuxian Li, Jie Feng, Huan Yan, Guangyin Jin, Fan Yang, Funing Sun, Depeng Jin, Yong Li | ACM Transactions on Knowledge Discovery from Data, 17, 1-21（2023） | 10.1145/3532611 | 2104.14917 | ACM DL | CR+S2 | 2026-11-02 | — |
| E49 | Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective | Yuchen Fang, Yuxuan Liang, Bo Hui, Zezhi Shao, Liwei Deng, Xu Liu, Xinke Jiang, Kai Zheng | Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, 307-317（2025） | 10.1145/3690624.3709177 | 2412.09972 | ACM DL | CR+S2 | 2026-11-02 | — |
| E50 | LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting | Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, Roger Zimmermann | Advances in Neural Information Processing Systems 36, 75354-75371（2023） | 10.52202/075280-3293 | 2306.08259 | NeurIPS 官方页（数据集与基准轨道） | CR+S2 | 2026-11-03 | — |
| E51 | BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks | Jindong Han, Weijia Zhang, Hao Liu, Tao Tao, Naiqiang Tan, Hui Xiong | Proceedings of the VLDB Endowment, 17, 1081-1090（2024） | 10.14778/3641204.3641217 | 无 | PVLDB | CR+S2 | 2026-11-02 | — |
| E52 | Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting | Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Jinliang Deng, Quanjun Chen, Xuan Song | Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 4125-4129（2023） | 10.1145/3583780.3615160 | 2308.10425 | ACM DL | CR+S2 | 2026-11-02 | 正式题名无 "STAEformer:" 前缀 |
| E53 | PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction | Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, Jingyuan Wang | Proceedings of the AAAI Conference on Artificial Intelligence, 37, 4365-4373（2023） | 10.1609/aaai.v37i4.25556 | 2301.07945 | AAAI OJS | CR+S2 | 2026-11-02 | — |
| E54 | Adaptive Frequency Pathways for Spatiotemporal Forecasting | Yanjun Qin, Yuchen Fang, Xinke Jiang, Hao Miao, Xiaoming Tao | Proceedings of the AAAI Conference on Artificial Intelligence, 40, 15653-15661（2026） | 10.1609/aaai.v40i18.38595 | 无 | AAAI OJS 38595 | CR+S2 | 2026-11-02 | — |
| E55 | Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting | Zezhi Shao, Zhao Zhang, Fei Wang, Wei Wei, Yongjun Xu | Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management, 4454-4458（2022） | 10.1145/3511808.3557702 | 2208.05233 | ACM DL | CR+S2 | 2026-11-02 | — |
| E56 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling | Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang | Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 1907-1913（2019） | 10.24963/ijcai.2019/264 | 1906.00121 | IJCAI 论文集 | CR+S2 | 2026-11-02 | — |
| E57 | Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting | Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang | NeurIPS 2020 | 无 | 2007.02842 | https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html | PUB(页面可访问)+S2 | 2026-11-03 | NeurIPS 2020 无 10.52202 DOI |
| E58 | Traffic Prediction With Missing Data: A Multi-Task Learning Approach | Ao Wang, Yongchao Ye, Xiaozhuang Song, Shiyao Zhang, James J. Q. Yu | IEEE Transactions on Intelligent Transportation Systems, 24, 4189-4202（2023） | 10.1109/TITS.2022.3233890 | 无 | IEEE Xplore | CR | 2026-11-03 | — |
| E59 | RT-GCN: Gaussian-based spatiotemporal graph convolutional network for robust traffic prediction | Yutian Liu, Soora Rasouli, Melvin Wong, Tao Feng, Tianjin Huang | Information Fusion, 102, 102078（2024） | 10.1016/j.inffus.2023.102078 | 无 | Elsevier | CR | 2026-11-03 | — |
| E60 | Graph Markov network for traffic forecasting with missing data | Zhiyong Cui, Longfei Lin, Ziyuan Pu, Yinhai Wang | Transportation Research Part C: Emerging Technologies, 117, 102671（2020） | 10.1016/j.trc.2020.102671 | 1912.05457（检索结果） | Elsevier | CR | 2026-11-03 | arXiv 号只见于检索结果 |
| E61 | STA-GANN: A Valid and Generalizable Spatio-Temporal Kriging Approach | Yujie Li, Shao Zezhi, Chengqing Yu, Tangwen Qian, Zhao Zhang, Yifan Du, Shaoming He, Fei Wang, Yongjun Xu | Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 1726-1736（2025） | 10.1145/3746252.3761045 | 2508.16161 | ACM DL | CR | 2026-11-03 | — |
| E62 | Relation-diffusion augmented network-wide traffic state estimation under sparse sensor deployment | Tao Guo, Ningkang Yang, Kevin Yu, Panagiotis Angeloudis, Constantinos Antoniou | Communications in Transportation Research, 6, 9640046（2026） | 10.26599/COMMTR.2026.9640046 | 无 | SciOpen | CR | 2026-11-03 | — |
| E63 | Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting | Wei Chen, Yuxuan Liang | Advances in Neural Information Processing Systems 38, 173098-173132（2025） | 10.52202/085713-5210 | 2506.00635 | NeurIPS 官方页 | CR+AX | 2026-11-03 | — |
| E64 | Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States | Shuhao Li, Weidong Yang, Changan Liu, Wei Zhuo, Yingbo Zhou, Fan Zhang, Siqiang Luo | arXiv 预印本 | 无 | 2610.08818v2（v1 2026-09-24，v2 2026-10-08） | arXiv | AX | 2026-11-03 | 单一来源；预印本 |
| E65 | Traffic STGNNs across Sensor, City, and Time Shifts: Routing Concentration Tracks Sensitivity to Sensor Dropout | Xile Wang, Mingqi Yang（会议页面） | NeurIPS 2026（Poster） | 无 | 未见 | https://neurips.cc/virtual/2026/poster/139842；OpenReview 1DWmoUiM00 | PUB(会议页面，两次读取一致) | 2026-11-03 | 单一官方来源；轨道只显示 Poster；全文未读 |
| E66 | Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting | Zheng Fang, Qingqing Long, Guojie Song, Kunqing Xie | Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining（2021） | 10.1145/3447548.3467430 | 未核 | ACM DL | CR | 2026-11-03 | — |
| E67 | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting | Bing Yu, Haoteng Yin, Zhanxing Zhu | Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, 3634-3640（2018） | 10.24963/ijcai.2018/505 | 1709.04875 | IJCAI 论文集 | CR | 2026-11-02 | S2 作者解析有误，以 Crossref 为准 |

## 3. 旧调研 28 项的复核映射

v1 矩阵开头写"旧文件的 28 项工作全部重新核对了元数据"，与 v1 第 9.3 节"STGformer、RT-GCN、GMN/SGMN 本轮未重新核实"相矛盾。v2 收回"全部重核"的说法，按下表逐项说明：

| 旧文件条目 | 本表编号 | 2026-11-03 状态 |
|---|---|---|
| GRIN | E13 | S2 + 旧文件 OpenReview 链接；ICLR 永久页本轮未读 |
| PriSTI | E14 | CR |
| GCN-M | C14 | CR |
| GinAR | C09 | CR |
| GMN/SGMN | E60 | CR（第二轮补核） |
| RT-GCN | E59 | CR（第二轮补核） |
| IBN | E23 | 只有 AX 一路 |
| IGNNK | E01 | CR |
| INCREASE | E02 | CR |
| FUNS | C22 | CR（第二轮补核；v1 误标"未正式发表"） |
| UIGNN | C03 | CR |
| MoGERNN | C01 | CR + LOC |
| ST-FiT | C04 | CR |
| SLPF | C08 | AX + S2 + mlanthology（TMLR 无 DOI） |
| SIML-MSIGN | C02 | CR + LOC |
| SC-SAMG | C07 | CR |
| DGAE | E11 | CR（旧文件"年份未裁定"现确认为 TITS 2025） |
| STGCN | E67 | CR |
| Graph WaveNet | E56 | CR |
| AGCRN | E57 | PUB + S2 |
| STID | E55 | CR |
| PDFormer | E53 | CR |
| STAEformer | E52 | CR |
| PatchSTG | E49 | CR |
| STBP | E39 | 只有 PUB 一路（ICLR 2026 官方论文集） |
| AdaFre | E54 | CR |
| STGODE | E66 | CR（第二轮补核） |
| STGformer | 未列入 | S2 两次 429，arXiv:2410.00385 未重核；正式会议未核实 |

即：28 项中 27 项已有本轮可追溯的核实路径，其中 IBN、STBP 只有单一来源；STGformer 仍未核实，不列入矩阵。
