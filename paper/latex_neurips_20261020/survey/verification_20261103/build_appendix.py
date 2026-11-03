import json, sys

cr = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]

D1, D2 = "2026-11-02", "2026-11-03"
# id, doi(formal or None), arxiv, official source, sources, access, unverified fields / notes, manual(meta) dict
E = [
 ("C01", "10.1016/j.trc.2025.105080", "2501.12281（预印本题名多 \"in Dynamic Sensing Networks\"）", "https://doi.org/10.1016/j.trc.2025.105080", "CR+S2+LOC", D1, "arXiv 版本未逐一核对", None),
 ("C02", "10.1016/j.trc.2026.105564", "无", "https://doi.org/10.1016/j.trc.2026.105564", "CR+S2+LOC", D1, "", None),
 ("C03", "10.1109/ICDM58522.2023.00152", "2309.06800v5（长版）", "IEEE Xplore 10415744", "CR+S2+AX", D1, "", None),
 ("C04", "10.1609/aaai.v39i11.33310", "2412.10912v2", "AAAI OJS 33310", "CR+S2+AX", D1, "", None),
 ("C05", "10.48786/edbt.2024.42", "2401.10518v1", "https://openproceedings.org/2024/conf/edbt/paper-123.pdf", "DC+PUB(PDF 首页)+S2+AX", D2, "DOI 为 DataCite 注册（OpenProceedings），Crossref 无记录属正常；终止页码未核", {"title": "Spatial-temporal Forecasting for Regions without Observations", "authors": "Xinyu Su, Jianzhong Qi, Egemen Tanin, Yanchuan Chang, Majid Sarvi", "venue": "EDBT 2024, pp. 488–（首页页码）"}),
 ("C06", "10.1609/aaai.v40i18.38607", "2508.08947v2", "AAAI OJS 38607", "CR+S2+AX", D1, "", None),
 ("C07", "10.3390/s26154995", "无", "MDPI（页面 403）", "CR+S2+OpenAlex", D1, "全文未读", None),
 ("C08", None, "2408.02689v3（2026-09-30；v1 2024-08-02，v2 2025-08-08）", "OpenReview Ff08aPjVjD（v3 正文所载）", "AX(注释与 journal-ref)+S2+mlanthology", D2, "TMLR 无 DOI；OpenReview 论坛页本轮未直接读取", {"title": "Spatio-Temporal Partial Sensing Forecast for Long-term Traffic（arXiv 题名；mlanthology 记为 \"…of Long-Term Traffic\"）", "authors": "Zibo Liu, Zhe Jiang, Zelin Xu, Tingsong Xiao, Zhengkun Xiao, Yupu Zhang, Haibo Wang, Shigang Chen", "venue": "Transactions on Machine Learning Research, 2025"}),
 ("C09", "10.1145/3637528.3672055", "2405.11333v1", "ACM DL", "CR+S2+AX", D1, "", None),
 ("C10", "10.1109/TKDE.2025.3569649", "无", "IEEE Xplore 11002729", "CR+S2+OpenAlex", D1, "全文未读", None),
 ("C11", "10.1145/3711896.3737046", "2506.12459v1", "ACM DL", "CR+S2+AX", D1, "", None),
 ("C12", None, "2402.10634（读 v2 附录 F、v3 正文）", "https://proceedings.mlr.press/v235/marisca24a.html", "PUB(PMLR)+S2+AX", D2, "ICML 无正式 DOI", {"title": "Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling", "authors": "Ivan Marisca, Cesare Alippi, Filippo Maria Bianchi", "venue": "ICML 2024, PMLR 235, pp. 34846–34865"}),
 ("C13", "10.1049/itr2.70304", "无", "Wiley/IET", "CR+S2+OpenAlex", D1, "全文未读", None),
 ("C14", "10.1007/s10618-022-00903-7", "2212.06419", "Springer", "CR+S2", D1, "", None),
 ("C15", "10.1609/aaai.v39i12.33412", "2311.02565v2", "AAAI OJS 33412", "CR+S2+AX", D1, "", None),
 ("C16a", None, "2603.05301v1（2026-03-05）", "arXiv", "AX+S2", D1, "预印本；v1 与 v2 题名、作者、方法均不同，引用须注明版本", {"title": "UniSTOK: Uniform Inductive Spatio-Temporal Kriging", "authors": "Lewei Xie, Haoyu Zhang, Juan Yuan, Liangjun You, Yulong Chen, Yifan Zhang", "venue": "arXiv 预印本 v1"}),
 ("C16b", None, "2603.05301v2（2026-05-09）", "arXiv", "AX", D2, "预印本；当前最新版", {"title": "Uniform Inductive Spatio-Temporal Kriging", "authors": "Lewei Xie, Haoyu Zhang, Yulong Chen, Liangjun You, Zongxian Yang, Yifan Zhang", "venue": "arXiv 预印本 v2"}),
 ("C17", None, "2403.02600（注释：ICLR 2024）", "arXiv 摘要页；代码页", "AX+S2", D1, "ICLR 永久页未读；无正式 DOI", {"title": "TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts", "authors": "Hyunwook Lee, Sungahn Ko", "venue": "ICLR 2024"}),
 ("C18", None, "2605.29768v2（2026-06）", "arXiv", "AX+S2", D1, "预印本", {"title": "From XXLTraffic to EvoXXLTraffic: Scaling Traffic Forecasting to Sensor-Evolving Networks", "authors": "Du Yin, Hao Xue, Arian Prabowo, Shuang Ao, Flora Salim", "venue": "arXiv 预印本"}),
 ("C19", "10.1145/3580305.3599357", "2306.08277v1", "ACM DL；KDD 2023 目录", "CR+AX", D2, "代码仓库 BibTeX 中 DOI 前缀有误（据 Codex 报告，未独立打开 BibTeX）", None),
 ("C20", "10.1007/s10618-023-00982-0", "2305.05237（v4，2023-09-21）", "Springer", "CR+AX", D2, "", None),
 ("C21", None, "未核", "https://proceedings.mlr.press/v267/ma25s.html", "PUB(PMLR 页与 PDF)", D2, "ICML 无正式 DOI；arXiv 号未核", {"title": "Robust Spatio-Temporal Centralized Interaction for OOD Learning", "authors": "Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, Yang Wang", "venue": "ICML 2025, PMLR 267, pp. 42165–42192"}),
 ("C22", "10.1109/ICDMW58026.2022.00101", "2211.11596", "IEEE Xplore", "CR+AX", D2, "ICDM Workshops，不是 ICDM 主会", None),
 ("E01", "10.1609/aaai.v35i5.16575", "2006.07527", "AAAI OJS", "CR+S2", D1, "", None),
 ("E02", "10.1145/3543507.3583525", "2302.02738", "ACM DL", "CR+S2", D1, "", None),
 ("E03", "10.1145/3580305.3599372", "2305.18719", "ACM DL", "CR+S2", D1, "", None),
 ("E04", "10.1109/TNNLS.2023.3293814", "2109.09506", "IEEE Xplore", "CR+S2", D1, "", None),
 ("E05", "10.1016/j.knosys.2023.111325", "无", "Elsevier", "CR+S2", D1, "摘要未取得", None),
 ("E06", "10.1145/3770854.3780189", "2512.21569", "ACM DL", "CR+S2+AX", D1, "", None),
 ("E07", None, "2109.12144", "arXiv", "S2", D1, "只有单一聚合来源；预印本", {"title": "Spatial Aggregation and Temporal Convolution Networks for Real-time Kriging", "authors": "Yuankai Wu, Dingyi Zhuang, Mengying Lei, Aurélie Labbe, Lijun Sun", "venue": "arXiv 预印本"}),
 ("E08", None, "2503.09646", "arXiv", "S2", D1, "单一聚合来源；预印本", {"title": "Inductive Spatio-Temporal Kriging with Physics-Guided Increment Training Strategy for Air Quality Inference", "authors": "Songlin Yang, Tao Yang, Bo Hu", "venue": "arXiv 预印本"}),
 ("E09", None, "2512.17984", "arXiv", "S2", D1, "单一聚合来源；预印本", {"title": "A Hybrid Inductive-Transductive Network for Traffic Flow Imputation on Unsampled Locations", "authors": "Mohammadmahdi Rahimiasl, Ynte Vanderhoydonc, Siegfried Mercelis", "venue": "arXiv 预印本"}),
 ("E11", "10.1109/TITS.2025.3610911", "2503.15845", "IEEE Xplore", "CR+S2", D1, "", None),
 ("E12", "10.1145/3589132.3625631", "2502.16935", "ACM DL", "CR+S2", D1, "", None),
 ("E13", None, "2108.00298", "OpenReview kOu3-S3wJ7（旧文件）", "S2+OLD", D1, "ICLR 永久页本轮未读", {"title": "Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks", "authors": "Andrea Cini, Ivan Marisca, Cesare Alippi", "venue": "ICLR 2022"}),
 ("E14", "10.1109/ICDE55515.2023.00150", "2302.09746", "IEEE Xplore", "CR+S2", D1, "", None),
 ("E15", "10.52202/068431-2324", "2205.13479", "NeurIPS 官方页", "CR+S2", D2, "", None),
 ("E16", "10.1145/3637528.3671751", "2312.01728", "ACM DL", "CR+S2", D1, "", None),
 ("E17", "10.24963/ijcai.2025/382", "2506.08054", "IJCAI 论文集", "CR+S2", D1, "", None),
 ("E18", "10.1609/aaai.v40i18.38581", "2601.04572", "AAAI OJS 38581", "CR+S2", D1, "", None),
 ("E19", None, "无", "https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html", "PUB(旧审计)+S2", D1, "", {"title": "Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values", "authors": "Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li（S2 记录）", "venue": "ICLR 2024"}),
 ("E20", None, "2503.00900", "arXiv 摘要页", "AX+S2", D1, "第二作者 arXiv 为 Meiqi Yang，S2 为 Meiqing Yang，未裁定；ICLR 永久页未读", {"title": "S4M: S4 for multivariate time series forecasting with Missing values", "authors": "Jing Peng, Meiqi Yang, Qiong Zhang, Xiaoxiao Li", "venue": "ICLR 2025"}),
 ("E21", "10.1016/j.neunet.2026.108986", "2508.14083", "Elsevier", "CR+S2", D1, "摘要未取得", None),
 ("E22", "10.1109/TMC.2025.3591423", "无", "IEEE Xplore 11087699", "CR+S2", D1, "", None),
 ("E23", None, "2509.07725", "arXiv 摘要页", "AX", D1, "单一来源；预印本；正式录用未见", {"title": "IBN: An Interpretable Bidirectional-Modeling Network for Multivariate Time Series Forecasting with Variable Missing", "authors": "Shusen Ma, Tianhao Zhang, Qijiu Xia, Yun-Bo Zhao", "venue": "arXiv 预印本"}),
 ("E24", "10.1038/s41598-018-24271-9", "1606.01865", "Nature", "CR+S2", D1, "", None),
 ("E25", None, "2601.01480", "arXiv", "S2", D1, "单一聚合来源；预印本；只有检索片段", {"title": "Modeling Information Blackouts in Missing Not-At-Random Time Series Data", "authors": "Aman Sunesh, Alan Ma, Siddarth Nilol", "venue": "arXiv 预印本"}),
 ("E26", "10.52202/075280-2417", "2302.04071", "NeurIPS 官方页", "CR+S2", D2, "", None),
 ("E27", "10.1145/3627673.3679554", "2405.12452", "ACM DL", "CR+S2", D1, "", None),
 ("E28", None, "2402.11922", "arXiv", "S2", D1, "单一聚合来源；ICLR 永久页未读；作者全列表未二次核对", {"title": "Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation", "authors": "Yuan Yuan, Chenyang Shao, Jingtao Ding, Depeng Jin 及另 1 位（S2 记录 5 人）", "venue": "ICLR 2024"}),
 ("E29", "10.1145/3583780.3614829", "2308.09727", "ACM DL", "CR+S2", D1, "", None),
 ("E30", "10.1145/3637528.3671662", "2402.11838", "ACM DL", "CR+S2", D1, "", None),
 ("E31", "10.1145/3773912", "2408.10269", "ACM DL", "CR+S2", D1, "", None),
 ("E32", None, "2405.17898", "arXiv", "S2", D1, "单一聚合来源；ICML 永久页未读", {"title": "FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction", "authors": "Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang", "venue": "ICML 2024"}),
 ("E33", "10.24963/ijcai.2021/498", "2106.06273", "IJCAI 论文集", "CR+S2", D1, "", None),
 ("E34", "10.1145/3580305.3599463", "无", "ACM DL", "CR", D1, "", None),
 ("E35", "10.1109/TITS.2023.3263904", "无", "IEEE Xplore", "CR+S2", D1, "", None),
 ("E36", "10.1109/ICDE60146.2024.00085", "2404.14999", "IEEE Xplore", "CR+S2", D1, "", None),
 ("E37", None, "2406.03140", "arXiv", "S2", D1, "单一聚合来源；预印本", {"title": "Continual Traffic Forecasting via Mixture of Experts", "authors": "Sanghyun Lee, Chanyoung Park", "venue": "arXiv 预印本"}),
 ("E38", None, "2410.12593（注释：ICLR 2025 录用）", "arXiv 摘要页", "AX+S2", D1, "ICLR 永久页未读", {"title": "Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting", "authors": "Wei Chen, Yuxuan Liang", "venue": "ICLR 2025"}),
 ("E39", None, "未核", "https://proceedings.iclr.cc/paper_files/paper/2026/hash/7547048d73ed056286762b17434466a7-Abstract-Conference.html", "PUB", D1, "单一官方来源；S2 未命中", {"title": "A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting", "authors": "Aoyu Liu, Yaying Zhang", "venue": "ICLR 2026"}),
 ("E40", "10.1145/3637528.3671680", "未核", "ACM DL", "CR+S2", D1, "", None),
 ("E41", None, "2410.00373", "arXiv", "S2", D1, "单一聚合来源；预印本；作者全列表未二次核对", {"title": "Robust Traffic Forecasting against Spatial Shift over Years", "authors": "Hongjun Wang, Jiyuan Chen, Tong Pan, Zheng Dong, Lingyu Zhang, Renhe Jiang, Xuan Song", "venue": "arXiv 预印本"}),
 ("E42", "10.52202/085713-3934", "2505.19547v2", "NeurIPS 官方页", "CR+S2", D2, "", None),
 ("E43", "10.1145/3748636.3762762", "2406.12693（arXiv 题名不同）", "ACM DL", "CR+S2", D1, "", None),
 ("E44", "10.52202/075280-2211", "2304.02806", "NeurIPS 官方页", "CR+S2", D2, "", None),
 ("E45", "10.52202/079017-3135", "2410.08245", "NeurIPS 官方页", "CR+S2", D2, "", None),
 ("E46", "10.1609/aaai.v32i1.11671", "1709.07871", "AAAI OJS", "CR+S2", D1, "", None),
 ("E47", "10.1609/aaai.v37i7.25976", "2211.14701", "AAAI OJS", "CR+S2", D1, "", None),
 ("E48", "10.1145/3532611", "2104.14917", "ACM DL", "CR+S2", D1, "", None),
 ("E49", "10.1145/3690624.3709177", "2412.09972", "ACM DL", "CR+S2", D1, "", None),
 ("E50", "10.52202/075280-3293", "2306.08259", "NeurIPS 官方页（数据集与基准轨道）", "CR+S2", D2, "", None),
 ("E51", "10.14778/3641204.3641217", "无", "PVLDB", "CR+S2", D1, "", None),
 ("E52", "10.1145/3583780.3615160", "2308.10425", "ACM DL", "CR+S2", D1, "正式题名无 \"STAEformer:\" 前缀", None),
 ("E53", "10.1609/aaai.v37i4.25556", "2301.07945", "AAAI OJS", "CR+S2", D1, "", None),
 ("E54", "10.1609/aaai.v40i18.38595", "无", "AAAI OJS 38595", "CR+S2", D1, "", None),
 ("E55", "10.1145/3511808.3557702", "2208.05233", "ACM DL", "CR+S2", D1, "", None),
 ("E56", "10.24963/ijcai.2019/264", "1906.00121", "IJCAI 论文集", "CR+S2", D1, "", None),
 ("E57", None, "2007.02842", "https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html", "PUB(页面可访问)+S2", D2, "NeurIPS 2020 无 10.52202 DOI", {"title": "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting", "authors": "Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang", "venue": "NeurIPS 2020"}),
 ("E58", "10.1109/TITS.2022.3233890", "无", "IEEE Xplore", "CR", D2, "", None),
 ("E59", "10.1016/j.inffus.2023.102078", "无", "Elsevier", "CR", D2, "", None),
 ("E60", "10.1016/j.trc.2020.102671", "1912.05457（检索结果）", "Elsevier", "CR", D2, "arXiv 号只见于检索结果", None),
 ("E61", "10.1145/3746252.3761045", "2508.16161", "ACM DL", "CR", D2, "", None),
 ("E62", "10.26599/COMMTR.2026.9640046", "无", "SciOpen", "CR", D2, "", None),
 ("E63", "10.52202/085713-5210", "2506.00635", "NeurIPS 官方页", "CR+AX", D2, "", None),
 ("E64", None, "2610.08818v2（v1 2026-09-24，v2 2026-10-08）", "arXiv", "AX", D2, "单一来源；预印本", {"title": "Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States", "authors": "Shuhao Li, Weidong Yang, Changan Liu, Wei Zhuo, Yingbo Zhou, Fan Zhang, Siqiang Luo", "venue": "arXiv 预印本"}),
 ("E65", None, "未见", "https://neurips.cc/virtual/2026/poster/139842；OpenReview 1DWmoUiM00", "PUB(会议页面，两次读取一致)", D2, "单一官方来源；轨道只显示 Poster；全文未读", {"title": "Traffic STGNNs across Sensor, City, and Time Shifts: Routing Concentration Tracks Sensitivity to Sensor Dropout", "authors": "Xile Wang, Mingqi Yang（会议页面）", "venue": "NeurIPS 2026（Poster）"}),
 ("E66", "10.1145/3447548.3467430", "未核", "ACM DL", "CR", D2, "", {"title": "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting", "authors": "Zheng Fang, Qingqing Long, Guojie Song, Kunqing Xie", "venue": "KDD 2021"}),
 ("E67", "10.24963/ijcai.2018/505", "1709.04875", "IJCAI 论文集", "CR", D1, "S2 作者解析有误，以 Crossref 为准", None),
]

def fmt(m):
    c = m.get("container", ""); iss = m.get("issued") or []
    yr = str(iss[0]) if iss else ""
    vol = m.get("volume", ""); pg = m.get("page", "")
    s = c
    if vol: s += f", {vol}"
    if pg: s += f", {pg}"
    return yr, s

rows = []
for (i, doi, ax, src_url, srcs, acc, note, man) in E:
    if doi and doi.lower() in cr:
        m = cr[doi.lower()]
        title, authors = m["title"], m["authors"]
        yr, venue = fmt(m)
        venue = f"{venue}（{yr}）"
        if man and "title" in man:  # crossref absent but manual provided
            pass
    elif man:
        title, authors, venue = man["title"], man["authors"], man["venue"]
    else:
        title, authors, venue = "【缺】", "【缺】", "【缺】"
    if man and doi and doi.lower() not in cr:
        title, authors, venue = man["title"], man["authors"], man["venue"]
    rows.append(f"| {i} | {title} | {authors} | {venue} | {doi or '无'} | {ax} | {src_url} | {srcs} | {acc} | {note or '—'} |")

hdr = ("| 编号 | 完整题名 | 作者 | 正式出处（年份） | 正式 DOI | arXiv（版本） | 官方来源 | 元数据核实路径 | 访问日期 | 未核字段与备注 |\n"
       "|---|---|---|---|---|---|---|---|---|---|")
open(out, "w", encoding="utf-8").write(hdr + "\n" + "\n".join(rows) + "\n")
print(len(rows), "rows;", sum(1 for r in rows if "【缺】" in r), "missing")
