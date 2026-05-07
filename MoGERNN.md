# MoGERNN: An inductive traffic predictor for unobserved locations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/b8259834a6df2bcf2f227aa0d2a51a7aa2f3c4a4f503c2aea082a804f2bebc3d.jpg)


Qishen Zhou $\mathbf { a } , \mathbf { b } \oplus$ , Yifan Zhang $\mathbf { c } , \mathbf { b } \oplus \mathbf { \overline { { \mathbb { D } } } }$ , Michail A. Makridis b , Anastasios Kouvelas b , ： Yibing Wang a , Simon Hu d,a ,∗ 

a Institute of Intelligent Transportation Systems, College of Civil Engineering and Architecture, Zhejiang University, Hangzhou, 310058, China 

b Institute for Transport Planning and Systems, ETH Zurich, Zurich, 8093, Switzerland 

c Department of Computer Science, City University of Hong Kong (Dongguan), Dongguan, 523808, China 

d ZJU-UIUC Institute, Zhejiang University, Haining, 314400, China 

# A R T I C L E I N F O

Dataset link: Performance Measurement System (PeMS) 

Keywords: 

Spatio-temporal extrapolation 

Traffic state estimation 

Traffic prediction 

Kriging 

Inductive graph representation learning 

Mixture of experts 

# A B S T R A C T

Given a partially observed road network, how can we predict the traffic state of interested unobserved locations? Traffic prediction is crucial for advanced traffic management systems, with deep learning approaches showing exceptional performance. However, most existing approaches assume sensors are deployed at all locations of interest, which is impractical due to financial constraints. Furthermore, these methods are typically fragile to structural changes in sensing networks, which require costly retraining even for minor changes in sensor configuration. To address these challenges, we propose MoGERNN, an inductive spatio-temporal graph model with two key components: (i) a Mixture of Graph Experts (MoGE) with sparse gating mechanisms that dynamically route nodes to specialized graph aggregators, capturing heterogeneous spatial dependencies efficiently; (ii) a graph encoder-decoder architecture that leverages these embeddings to capture both spatial and temporal dependencies for comprehensive traffic state prediction. Experiments on two real-world datasets show MoGERNN consistently outperforms baseline methods for both observed and unobserved locations. MoGERNN can accurately predict congestion evolution even in areas without sensors, offering valuable information for traffic management. Moreover, MoGERNN is adaptable to the changes of sensor network, maintaining competitive performance even compared to its retrained counterpart. Tests performed with different numbers of available sensors confirm its consistent superiority, and ablation studies validate the effectiveness of its key modules. The code of this work is publicly available at: https://github.com/ZJU-TSELab/MoGERNN. 

# 1. Introduction

Large-scale traffic forecasting plays a pivotal role in advanced traffic management systems and has gained considerable attention over recent decades. Among the existing research, deep learning-based approaches dominate and achieve extraordinary performance (Shaygan et al., 2022; Yin et al., 2022). In particular, Graph Neural Networks (GNNs) have established state-of-the-art results by effectively capturing heterogeneous spatio-temporal dynamics in traffic flow (Jin et al., 2024). Despite these advances, two critical issues have received limited attention. First, prediction of traffic states at unobserved locations: nearly all existing studies presuppose the presence of sensors at the all locations of interest. However, due to financial constraints, it is impractical to deploy and maintain sufficient sensors across all target areas. Second, fragility to the structural changes of sensing networks: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/1366ee09a6377835e365c69fe4796665ce1c004940ab111e2d60a4122938368f.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/e1db06ae29e1a03161c1a90f167f08f5d2ddd4862a3e9fe8a181b979aeeb1c36.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ffaff3a03787712b6cbd2c4bd00a49056d67d7a1c83b4ee3c5cf0dccd93d15c5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/960e5704a1636e9cf01dbc2c786b13f8a204d0153ef63c7b0ffa2ca42f1ed732.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/82cb0f4adc54c8a9f64fabe715c4c75b730dce52890a31c407aa811697897e01.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/24a5aac69303e1b03585866d6390a189663f704c4f91f8eb0a9164766c01fc0c.jpg)



Available sensor


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/08d634d6773f6d0f988c95449539ba748d51d993bb0c9fbd2ffb29deededcae3.jpg)



New added sensor


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/30117f20352be9ef4829304cd5f9d885afb17f9fae282ca78604f71aedb3162f.jpg)



Failed sensor


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/7a1cd9fb4bf4a40fc640f6c15927e40187f0b5551d3493ab89eedf53e2ca3454.jpg)



Virtual sensor (Unobserved location)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ea38dafb418778f1cbc7569e5e744fd8acc373a104e34eb80404026e696e69e0.jpg)



Bidirectional connection


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/47f981266c31eb3368edcbf0ec64f5f5a9fffd7196cbd987ad5f111ff6c490d6.jpg)



Unidirectional connection


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/b0c27217c3ecc5943bd3d94cae28e8eaa9cc59155623941f2fb763273953718e.jpg)



Disused connection



Fig. 1. An illustration of inductively Forecasting unobserved nodes (iFun), for the sake of presentation, unobserved locations are considered as being equipped with virtual sensors (Liu et al., 2009; Martin et al., 2021): (a) an example of sensor deployment on the map, with locations of interest without physical sensors; (b) graph typologies for changes in sensor configuration.


most approaches are transductive, requiring complete model retraining even for minor changes in sensor configuration, such as the addition of a single sensor. In real-world applications, where sensor networks undergo changes due to inevitably sensor retirement and new installations, along with potential cross-region model deployment (Liu et al., 2023b; Li et al., 2024), this retraining requirement severely constrains their practical applicability. These challenges can be conceptualized as the problem of inductively Forecasting unobserved nodes (iFun), where the key is to develop inductive models that can generalize to unseen sensor configurations without retraining while forecasting states at unobserved locations, as illustrated in Fig. 1. 

The main challenge of iFun lies in modeling heterogeneous spatial relationships without historical data from unobserved sites while adapting to possible changes in sensor deployment configurations. Inductive GNNs, with their message passing mechanism (Gilmer et al., 2017), can generalize to unseen nodes and graph structure (Hamilton et al., 2017; Rossi et al., 2018), which aligns well with the requirements of iFun and has achieved initial success in unobserved nodes state inference (Appleby et al., 2020; Wu et al., 2021a). However, these studies primarily operate on static graph structures, making it difficult to capture complex spatiotemporal dependencies. On the other hand, while advanced GNNs based on dynamic graph have achieved remarkable success in modeling spatial dependency, they are either transductive architecture with learnable node embeddings (Cirstea et al., 2022; Li et al., 2023b; Weng et al., 2023) or heavily dependent on complete raw node features like dynamic time warping (DTW) (Berndt and Clifford, 1994; Li et al., 2022) and graph attention (Guo et al., 2019; Zheng et al., 2020; Wang et al., 2023), making them inapplicable to the iFun problem, which has no historical data on unobserved nodes. These limitations raise a crucial question: how to design an inductive spatiotemporal graph neural network that can effectively capture heterogeneous and dynamic spatial relationships. Existing research efforts addressing this question have evolved along three distinct directions: (1) augmenting static adjacency matrices through the integration of auxiliary data, such as Points of Interest (POI) and Origin-Destination (OD) data (Zheng et al., 2023), though the accessibility of such auxiliary data remains a practical constraint; (2) attempting to combine static graph structure with learnable node embeddings (Wei et al., 2024). However, since these embeddings are only optimized for observed nodes, this approach still heavily relies on static graph structures and achieves only limited improvements with the help of node embedding; and (3) exploring multiple graph message aggregators (Wu et al., 2021b), drawing inspiration from Corso et al. (2020). Nevertheless, this approach encounters substantial computational complexity and raises fundamental questions regarding the optimal integration of diverse operators. Our work builds upon and extends the third direction, proposing novel solutions to address its inherent challenges. Beyond these spatial modeling considerations, a temporal dimension further complicates the iFun problem. Existing GNN-based approaches for unobserved node inference primarily address Kriging/state estimation problems (Appleby et al., 2020; Wu et al., 2021a; Zheng et al., 2023; Nie et al., 2023b; Wei et al., 2024; Xu et al., 2023), which aim to infer current traffic states of unobserved locations using concurrent observations from monitored locations. To date, only a limited number of GNN-based methods (Roth and Liebig, 2022; Mei et al., 2023; Sun et al., 2023) directly address the challenge of forecasting future states of unobserved nodes, leaving room for a comprehensive solution that addresses both spatial and temporal aspects of the iFun problem. 

To address these challenges in both spatial and temporal dimensions of the iFun problem, we propose MoGERNN, a novel inductive spatio-temporal graph representation model. Our approach builds upon two key insights: First, to enhance static graph representation, we design a Mixture of Graph Experts (MoGE) module that integrates multiple specialized message aggregators with a sparse gating mechanism. Unlike traditional static graphs that rely on fixed aggregation patterns, our module captures diverse spatial dependencies through different graph experts, each specializing in distinct neighborhood interaction patterns. The sparse gating network performs two key functions: adaptively routing nodes to their most suitable experts and optimally coupling expert outputs. This design not only achieves more accurate node embeddings but also maintains computational efficiency through selective expert activation. Second, to enable future state prediction beyond current estimation, we develop a Spatio-Temporal Graph-based Encoder-Decoder (STGED) architecture that operates on the powerful node embeddings generated by MoGE, enabling joint spatio-temporal extrapolation and effectively addressing the iFun problem. 

The main contributions of this paper are as follows: 

• Develops a novel method that extends beyond conventional traffic prediction approaches by enabling accurate predictions for unobserved locations without historical data. 

• Designs a sparse-gated Mixture of Graph Experts (MoGE) framework that enhances model expressiveness through multiple specialized graph message aggregators and adaptive expert routing, effectively capturing diverse spatial dependencies in traffic networks. 

• Conducts comprehensive evaluations comparing MoGERNN with various pipeline and end-to-end models for addressing iFun problems from multiple perspectives. Results demonstrate our model’s superior performance over state-of-the-art methods in future traffic state prediction on two open-source real-world datasets, with notable improvements for unobserved nodes. 

• The proposed MoGERNN demonstrates robust adaptability to structural changes in sensor networks, maintaining prediction quality during sensor failures and achieving satisfactory zero-shot prediction for newly installed sensors. 

The remainder of the paper is organized as follows: Section 2 delivers a literature review related to iFun in traffic data modeling. Section 3 provides the formalization of the problem. Section 4 details the proposed MoGERNN model. Section 5 reports the performance and ablation results of the proposed method. Section 6 concludes the paper. 

# 2. Literature review

# 2.1. Traffic Kriging and state estimation

Both Kriging and TSE aim to estimate the current state of unobserved locations based on limited observations. The most widely used Kriging method is Ordinary Kriging (OKriging) (Wackernagel, 1995), which interpolates unseen location by using specific covariance and variogram functions to describe the spatial relationships of random variables at different locations. However, the method assumes the spatial field is stable, which does not correspond to the characteristics of traffic networks. When this stationarity assumption is relaxed, OKriging can be shown to be equivalent to Gaussian Process Regression (GPR), a well-known kernel machine learning method. While GPR offers more flexibility in modeling spatial relationships, it is computationally expensive and thus not applicable to large-scale networks. For large-scale applications, matrix/tensor completion with specific regularization, such as graph Laplacian (Bahadori et al., 2014; Nie et al., 2023a), local Markov structure (Xiong et al., 2010), Gaussian process (Lei et al., 2022), and temporal consistency (Nie et al., 2023a), serves as a effective solution. Notably, Lei et al. (2022) and Nie et al. (2023a) demonstrated satisfactory outcomes in traffic data. Despite this, matrix completion cannot automatically adapt to changes in the sensor network, as it is transductive which requires re-calculation for a new sensor configuration. Moreover, the above Kriging methods cannot be extended to perform temporal forecasting for unobserved nodes, as their underlying theoretical framework is fundamentally designed only for spatial interpolation. 

Recently, inductive methods based on GNNs have been introduced for spatio-temporal Kriging. Appleby et al. (2020) and Wu et al. (2021a) demonstrated the prospect of graph convolution network and diffusion graph convolution network for spatial extrapolation by means of well-designed training procedure. Specifically, Wu et al. (2021a) enhanced model generalization to varying sensor configurations through random subgraph construction during the training phase, where different combinations of observed nodes are sampled to simulate potential changes in sensor deployment. However, these two methods are all built on a predefined adjacency matrix, which constrains their ability to capture complex spatial dependencies. To address this challenge, several methods have been proposed. Zheng et al. (2023) incorporated multi-source data such as Points of Interest (POI) and Origin-Destination (OD) information to improve model accuracy. Nevertheless, such auxiliary data is not always readily available in practical applications. Nie et al. (2023b) proposed a speed pattern-adaptive graph neural network that leverages ubiquitous speed measurements from floating cars and crowd-sourcing vehicles to address traffic volume inference at locations without fixed sensors. However, data integration between government-owned fixed sensors and privately operated mobile sensors faces substantial barriers due to commercial and privacy concerns (Wang and Yang, 2024). Roth and Liebig (2022) introduced Graph Attention mechanism (GAT) that adaptively assigns different weights to neighboring nodes based on their node embeddings. Nonetheless, in the context of iFun where data for unobserved nodes (virtual sensors) is completely missing, these node-feature driven dynamic neighbor weights cannot be learned effectively by attention mechanism, resulting in inaccurate node-wise information aggregation. Wei et al. (2024) adapted learnable node embedding-based dynamic graph techniques into an inductive framework and combines them with static graph structure, while introducing an auxiliary task to mitigate information propagation loss caused by random subgraph sampling in IGNNK. However, since these embeddings are optimized through loss functions that require complete node features during training, the resulting dynamic components cannot effectively generalize to unobserved nodes where such features are absent. Inspired by Corso et al. (2020), Wu et al. (2021b) proposed an alternative approach using multiple aggregators for neighbor message passing to capture complex spatial patterns, enhancing the model’s generalization capability and accuracy. Nevertheless, this method faces the challenge of effectively integrating multiple aggregators in response to varying graph structures and data contexts. 

Compared to OKriging and tensor completion methods, GNN-based approaches offer the flexibility to map outputs to future timestamps by adjusting training labels, which provides valuable insights for our work. However, it is worth noting that this adaptation is not universally applicable to all GNN-based Kriging methods. For instance, Xu et al. (2023) identified that traditional GNN-based Kriging methods using decremental training (graph sampling) create a significant graph gap between training and inference. In response, they propose an incremental training strategy by adding fake nodes, but their approach relies fundamentally on a Node-aware Cycle Regularization mechanism that assumes inputs and outputs correspond to the same time interval, rendering these techniques ineffective for future state prediction tasks. 

As for TSE, most approaches are based on physical models combined with Kalman Filtering (KF) and its variants (Wang and Papageorgiou, 2005; Roncoli et al., 2016; Sun and Work, 2018; Wang et al., 2022; Makridis and Kouvelas, 2023). Although filtering methods achieve state-of-the-art performance and are theoretically well-established, their algorithm structure design fundamentally restricts them to one-step-ahead prediction, making them inadequate for advanced traffic control methods (Frejo and Camacho, 2012) which requires multi-step traffic state forecasting. Another emerging direction in TSE research is the use of Physics-Informed Neural Network (PINN) (Raissi et al., 2019). This approach integrates prior traffic knowledge, represented in the form of Partial Differential Equations (PDEs) (Lighthill and Whitham, 1955; Richards, 1956; Payne, 1971; Whitham, 1974; Aw and Rascle, 2000; Zhang, 2002), into the neural network framework (Shi et al., 2021, 2022; Yuan et al., 2021, 2022), allowing for accurate estimation of states with limited observational data. However, the ability of this method to generalize beyond the spatio-temporal range of the training data is quite limited (Kim et al., 2021). Despite some more advanced studies attempting to address this issue (Davini et al., 2021; Fesser et al., 2023), they have not yet been applied to the traffic domain. 

# 2.2. Traffic forecasting for unobserved locations

Traffic forecasting has seen rapid development with various sophisticated deep-learning models, including STGNN (Yu et al., 2018; Li et al., 2018; Wu et al., 2020; Song et al., 2020), Transformer (Do et al., 2019; Liu et al., 2023a), CNN (Liu et al., 2022b; Wu et al., 2023), and MLP-based approaches (Shao et al., 2022; Wang et al., 2024). However, these methods typically assume the availability of historical observations at all locations and cannot be directly applied to unobserved sites where no historical data exists. For predicting traffic states at unobserved locations, current research efforts primarily follow two technical directions: 

The first direction decomposes the problem into independent spatial and temporal extrapolation tasks. Yang et al. (2023, 2024) proposed an innovative approach that utilizes neural networks to predict future boundary conditions of the road network, which are then fed into the Extended Kalman Filter (EKF) to predict future states of unobserved nodes. However, since Kalman filtering and its variants cannot be integrated into deep learning frameworks for joint training, this decoupled strategy leads to compounded errors from two independent procedures, potentially compromising the overall performance. Similarly, Sun et al. (2023) employed Gaussian Mixture Models (GMM) for spatial interpolation of unobserved nodes, followed by a combination of Graph Convolutional Networks and Gated Recurrent Units for global prediction. This approach faces similar limitations as GMM cannot be incorporated into deep learning backpropagation. Moreover, both methods adopt transductive architectures, resulting in limited generalization capabilities. 

The second direction embraces end-to-end deep learning approaches, where the entire prediction pipeline can be jointly optimized through backpropagation. Roth and Liebig (2022) adopted GATv2, an improved variant of Graph Attention Networks (GAT), to model spatiotemporal correlations for global prediction. However, GAT’s reliance on complete node features for establishing dynamic spatial relationships significantly constrains its predictive capacity for unobserved nodes and may contaminate the feature representations of observed nodes, thereby degrading overall performance. Mei et al. (2023) employed diffusion graph convolution networks for prediction, but this approach, being dependent on static adjacency matrices and lacking comprehensive temporal dynamics modeling, demonstrates limited capability in capturing the complex spatiotemporal characteristics of traffic systems. 

# 3. Problem formulation

In this section, we formally introduce the problem of inductively Forecasting unobserved nodes (iFun). Given a set of $N$ locations of interest in a traffic network, represented as nodes $\mathfrak { V } = \{ \nu _ { 1 } , \nu _ { 2 } , \dots , \nu _ { N } \}$ , with $\widetilde { N }$ nodes $( \widetilde N < N )$ observed by traffic sensors. We model the traffic network as a graph $\mathcal { G } = ( \mathcal { V } , \mathbf { A } )$ , where ?? denotes the weighted adjacency matrix. Considering the traffic flow characteristics, ?? is defined based on travel distance: 

$$
\mathbf {A} _ {i j} = \left\{ \begin{array}{l l} \exp \left(- \frac {\operatorname {d i s t} \left(\nu_ {i} , \nu_ {j}\right) ^ {2}}{\sigma^ {2}}\right), & \text {i f} \operatorname {d i s t} \left(\nu_ {i}, \nu_ {j}\right) \leq \kappa \\ 0, & \text {o t h e r w i s e} \end{array} \right. \tag {1}
$$

where $\mathrm { d i s t } ( \nu _ { i } , \nu _ { j } )$ denotes the travel distance from sensor $\nu _ { i }$ to $\nu _ { j }$ . $\sigma$ and $\kappa$ denote the standard deviation of distance samples and threshold, respectively. 

The primary target for iFun is to infer the future states of $\overline { { N } }$ unobserved locations $\sqrt { N } = N - \widetilde { N } )$ using the historical data from the $\widetilde { N }$ sensors and graph topology $\mathscr { G } \mathrm { : }$ : 

$$
f _ {\theta} (\widetilde {\mathbf {X}} ^ {\widetilde {N} \times P \times I}, \mathcal {G}) = \mathbf {X} ^ {N \times F \times O} \tag {2}
$$

where $\widetilde { \mathbf { X } }$ and $\overline { { \mathbf { X } } }$ denote the observed node and unobserved node features, respectively; $P$ and $F$ denote the historical and future time horizon, respectively. 

It is essential to highlight that for nodes without sensors, both their number and positions will dynamically change based on the specific requirements during the model’s application. Consequently, the complete graph $\mathcal { G }$ is only available during the application phase, also referred to as the test phase in this paper. In addition, considering possible structural changes happens in sensor network, the model must be able to accommodate any addition of new sensors $\widetilde { M }$ and failures of existing sensors $\overline { { M } }$ in real-time, i.e., no need for model retraining. The mathematical representation is as follows: 

$$
f _ {\theta} \left(\widetilde {\mathbf {X}} ^ {\left(\widetilde {N} + \widetilde {M} - \overline {{M}}\right) \times P}, \mathcal {G}\right) = \overline {{\mathbf {X}}} ^ {\left(\overline {{N}} - \widetilde {M} + \overline {{M}}\right) \times F} \tag {3}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/7f127482453644a4f40d266f94540e4810a1b0dd3d1df0cc4a7fd80f0ab5017c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/06167d57c007fdd6faef811c51c1f558f2d3584994cb647c6ccc5c9df5c5acb5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/a4a6e14cd21bfef80a33673929fb01ef0d2be340fab4a194ec45f2a0673fc5af.jpg)



Fig. 2. The overview of the proposed MoGERNN.


# 4. Methodology

In this section, we introduce the architecture of the proposed MoGERNN. Fig. 2 demonstrates the architecture of MoGERNN, it begins with an Mixture of Graph Experts (MoGE) block to model the heterogeneous spatio-temporal characteristic, and initially construct the embedding for unobserved nodes. Then, a Spatio-Temporal Graph Encoder-Decoder (STGED) framework is used to obtain multi-step predictions. Details of these two blocks and model training and testing are introduced in the following subsection. 

# 4.1. Mixture of graph experts for unobserved node embedding

To model the heterogeneous spatio-temporal characteristic, the following aggregators that are commonly used in spatio-temporal applications are introduced (Gilmer et al., 2017; Kipf and Welling, 2017; Xu et al., 2019; Corso et al., 2020). 

• Distance-based weighted aggregator: ??(??+1) = ∑??∈ (??) ??(??)?? ⋅?????? ∑ $\begin{array} { r } { \mathbf { X } _ { j } ^ { ( l + 1 ) } = \frac { \sum _ { i \in \mathcal { N } ( j ) } \mathbf { X } _ { i } ^ { ( l ) } \cdot \mathbf { A } _ { i j } } { \sum _ { i \in \mathcal { N } ( j ) } \mathbf { A } _ { i j } } } \end{array}$ 

• Mean aggregator: $\mathbf { X } _ { j } ^ { ( l + 1 ) } = \mathrm { m e a n } _ { i \in \mathcal { N } ( j ) } \mathbf { X } _ { i } ^ { ( l ) }$ 

• Max pooling aggregator: $\mathbf { X } _ { j } ^ { ( l + 1 ) } = \operatorname* { m a x } _ { i \in \mathcal { N } ( j ) } \mathbf { X } _ { i } ^ { ( l ) }$ 

• Min pooling aggregator: $\begin{array} { r } { \mathbf { X } _ { j } ^ { ( l + 1 ) } = \operatorname* { m i n } _ { i \in \mathcal { N } ( j ) } \mathbf { X } _ { i } ^ { ( l ) } } \end{array}$ 

• Diffusion convolution aggregator: 

$$
\mathbf {X} _ {j} ^ {(l + 1)} = \sum_ {k = 0} ^ {K - 1} \left(\sum_ {i} \left(D _ {O} ^ {- 1} \mathbf {A}\right) _ {i j} ^ {k} \mathbf {X} _ {i} ^ {(l)} \mathbf {W} _ {k, O} ^ {(l)} + \sum_ {i} \left(D _ {I} ^ {- 1} \mathbf {A} ^ {\top}\right) _ {i j} ^ {k} \mathbf {X} _ {i} ^ {(l)} \mathbf {W} _ {k, I} ^ {(l)}\right)
$$

where $\mathbf { X } ^ { ( l ) } \in \mathbb { R } ^ { N \times H _ { 1 } }$ denotes the input graph signal at layer ??.  (??) denotes the set of all nodes ?? that are neighbors of node $j$ , i.e., all nodes ?? such that $\mathbf { A } _ { i j } > 0$ . $\scriptstyle { { \mathcal { D } } _ { O } }$ and $\scriptstyle { \mathcal { D } } _ { I }$ denote out-degree and in-degree of adjacency matrix ??, respectively. $\mathbf { W } _ { k , O } ^ { ( l ) }$ ??(?? ),?? and ??(??)??,?? $\mathbf { W } _ { k , I } ^ { ( l ) }$ represent learnable parameters in $\mathbb { R } ^ { H _ { 1 } \times H _ { 1 } }$ at step $k$ , layer $l$ , respectively. In the following text, the graph aggregation operation is denoted as Agg. 

The selection of these aggregators is motivated by both theoretical foundations and empirical evidence in traffic modeling. The distance-weighted and mean aggregators serve complementary purposes: the former aligns with Tobler’s First Law of Geography (Tobler, 1970) by assigning higher weights to spatially closer nodes, while the latter treats all neighbors equally to capture potential long-range dependencies. The max-pooling and min-pooling aggregators are specifically designed to capture distinct traffic states - max-pooling for free-flow conditions characterized by higher speeds, and min-pooling for congested states marked by lower speeds. Additionally, the diffusion convolution aggregator is included based on its proven effectiveness in modeling traffic flow dynamics (Li et al., 2018). 

In order to avoid the effective of unobserved nodes to observed ones and further direct the network to focus on acquiring information from neighboring nodes, we make the following adjustments to the adjacency matrix used in the aggregators within first layer: 

$$
\mathbf {A} _ {i j} ^ {\prime} = \left\{ \begin{array}{l l} 0, & \text {i f} i = = j, \text {o r n o d e} j \text {i s n o t o b s e r v e d} \\ \mathbf {A} _ {i j}, & \text {o t h e r w i s e} \end{array} \right. \tag {4}
$$

Inspired by the Mixture of Experts (MoE) proposed in Large Language Model (LLM), we design a Mixture of Graph Experts (MoGE) to efficiently integrate these aggregators. MoGE assigns weights to different experts through a sparse gating network, which reduces the complexity of the model by activating only some of the experts at a time compared to dense gating. Existing research has shown that the MoE architecture not only enhances the model capacity without adding additional computational resources, but also integrates the strengths of different experts to improve the performance of the overall task (Li et al., 2023a; Lee and Ko, 2024). 

Graph Experts: Multiple graph experts are constructed based on different types of aggregators. Specifically, in each graph expert, the input features $\mathbf { X } \in \mathbb { R } ^ { N \times P }$ are first embedded into the latent space $\mathbb { R } ^ { N \times H _ { 2 } }$ with a learnable parameter $W _ { 1 } \in \mathbb { R } ^ { P \times H _ { 2 } }$ , then the features from different locations are mixed using the graph aggregator Agg defined above, and finally mapping the features back to the space $\mathbb { R } ^ { N \times P }$ through a linear layer and activation function Act: 

$$
\mathbf {X} _ {e} = \operatorname {A c t} \left(\operatorname {L i n e a r} \left(\operatorname {A g g} _ {e} \left(\mathbf {X} W _ {1}, \mathbf {A}\right)\right)\right) \tag {5}
$$

where $\mathbf { X } _ { e }$ denotes the output of ??th graph expert based on $e$ -type graph aggregator. 

Sparse Gating Network: To obtain scores $S$ for each expert, input features ?? that are fed into the graph experts are also processed through a scoring network composed of two linear layers and an activation function: 

$$
\mathbf {S} = \operatorname {L i n e a r} (\operatorname {A c t} (\operatorname {L i n e a r} (\mathbf {X}))) \tag {6}
$$

where ${ \bf S } \in \mathbb { R } ^ { N \times N _ { e } }$ , and $N _ { e }$ denotes the number of graph experts. 

Softmax function is used to convert the scores into weights ?? for different experts. To reduce the complexity, only top $K$ experts are activated: 

$$
\mathbf {w} = \operatorname {S o f t m a x} (\operatorname {K e e p T o p K} (\mathbf {S}, K)) \tag {7}
$$

$$
\operatorname {K e e p T o p K} (\mathbf {S}, K) = \left\{ \begin{array}{l l} \mathbf {S} _ {e}, & \text {i f} \mathbf {S} _ {e} \text {i s a m o n g t h e t o p} K \text {e l e m e n t s} \\ - \infty , & \text {o t h e r w i s e} \end{array} \right. \tag {8}
$$

Based on Eqs. (7) and (8), the sparse gating network can zero out certain expert weights, enabling selective expert activation during inference. As shown in Fig. 2, dashed lines represent inactive pathways. The sparse gating network plays a crucial role in our model by serving multiple purposes. Theoretically, it creates an information bottleneck that serves as implicit regularization, which is particularly important as the combination of different graph neural experts results in a complex model prone to overfitting. From a practical perspective, by activating only a subset of experts, it significantly reduces computational overhead while maintaining model effectiveness through optimal expert selection and weight assignment. 

Given graph experts and sparse gating network, the output of MoGE will be $\begin{array} { r } { \mathbf { X } _ { \mathrm { m o g e } } \ = \ \sum _ { e } \mathbf { w } _ { e } \odot \mathbf { X } _ { e } } \end{array}$ . In this work, MoGE is designed specifically for unobserved node embedding construction, the states of observed nodes remain unchanged before entering the encoder-decoder.: 

$$
\mathbf {X} = \mathbf {X} \odot \mathbf {M} + \mathbf {X} _ {\text {m o g e}} \odot (1 - \mathbf {M}) \tag {9}
$$

where $\mathbf { M } \in \mathbb { R } ^ { N \times P }$ is a binary matrix, with zeros indicating unobserved entities. Moreover, $\mathbf { X } \in \mathbb { R } ^ { N \times P }$ is reorganized as $\mathbf { X } \in \mathbb { R } ^ { N \times P \times 1 }$ , for the sake of operation in encoder-decoder. 

# 4.2. Graph encoder-decoder for spatio-temporal extrapolation

An encoder-decoder incorporating Recurrent Neural Networks (RNNs) is a widely used multi-step prediction framework that has achieved satisfactory results in traffic prediction (Wang et al., 2021; Kong et al., 2024). In this paper, we use the GRU-based Encoder-Decoder framework. Compared to standard RNNs and Long-Short Term Memory (LSTM), GRU requires less memory and fewer parameters while maintaining comparable performance, and has been demonstrated to couple well with GNN (Li et al., 2018; Cini et al., 2022). Inspired by Li et al. (2018), we replace the linear transformation in GRU with graph aggregators: 

$$
\mathbf {r} _ {t} = \operatorname {s i g m o i d} \left(\operatorname {A g g} _ {r} \left(\mathbf {X} _ {t} \| \mathbf {h} _ {t - 1}\right) + \mathbf {b} _ {r}\right) \tag {10}
$$

$$
\mathbf {u} _ {t} = \operatorname {s i g m o i d} \left(\operatorname {A g g} _ {u} \left(\mathbf {X} _ {t} \| \mathbf {h} _ {t - 1}\right) + \mathbf {b} _ {u}\right) \tag {11}
$$

$$
\mathbf {c} _ {t} = \tanh  \left(\operatorname {A g g} _ {c} \left(\mathbf {X} _ {t} \| \left(\mathbf {r} _ {t} \odot \mathbf {h} _ {t - 1}\right)\right) + \mathbf {b} _ {c}\right) \tag {12}
$$

$$
\mathbf {h} _ {t} = \mathbf {u} _ {t} \odot \mathbf {h} _ {t - 1} + (1 - \mathbf {u} _ {t}) \odot \mathbf {c} _ {t} \tag {13}
$$

where $\parallel$ denotes the concatenation operator, $\mathbf { h } _ { t } \in \mathbb { R } ^ { N \times H _ { 2 } }$ is the hidden value that contains compressed information about the sequence before time instance ??. In our design, $\mathbf { A } \mathbf { g } \mathbf { g } _ { r }$ , $\mathrm { A g g } _ { u }$ , and $\mathrm { A g g } _ { c }$ , which represent the graph aggregators for the reset gate, update gate, and candidate hidden state respectively, share the same type of graph aggregator. 

Two GRUs are initialized to act as encoder and decoder respectively. In the encoder, the historical sequence $\mathbf { X } _ { t } , \mathbf { X } _ { t + 1 } , \ldots , \mathbf { X } _ { t + P }$ is compressed to a hidden variable $\mathbf { h } _ { t + P }$ . This variable is used to initialize the hidden variable in the decoder. Different from encoder, decoder additionally contains a linear output layer, where the output of each step of decoder will be used as input for the next step. 


Algorithm 1 Inductive Training Procedure


Require: Data $\widetilde{\mathbf{X}}\in \mathbb{R}^{(\widetilde{N},L)}$ graph $\widetilde{G}$ , historical length $P$ , prediction length $F$ , stride $s$ , batch size $B$ , epochs $E$ , learning rate $\eta$ , mask rate $m$ , maximum epoch for training $E_{m}$ Ensure: Model parameters $\theta$ 1: Create training pairs $(\widetilde{\mathbf{X}}_{\mathrm{train}},\widetilde{\mathbf{y}}_{\mathrm{train}})$ using sliding window with stride $s$ 2: Initialize $\theta$ 3: for epoch $= 1$ to $E$ do   
4: for each mini-batch $(\widetilde{\mathbf{X}}_b,\widetilde{\mathbf{y}}_b)$ after random shuffle do   
5: Mask random nodes in $\widetilde{\mathbf{X}}_b$ with rate $m$ 6: $r = \max (1 - \mathrm{epoch} / E_m,0)$ $\triangleright$ Teacher forcing rate   
7: ${\bf y}^{-} = f_{\theta}({\widetilde{\bf X}}_{b},\widetilde{G},r,{\widetilde{\bf y}}_{b})$ 8: Update $\theta$ using $\nabla_{\theta}\mathcal{L}(\mathbf{y}^{-},\widetilde{\mathbf{y}}_{b})$ 9: end for   
10: end for   
11: return $\theta$ 


Table 1 Summary of datasets.


<table><tr><td>Dataset</td><td>Unit</td><td>Sensors number</td><td>Time period</td><td>Sampling frequency</td><td>Missing rate</td></tr><tr><td>METR-LA</td><td>mph</td><td>207</td><td>01/03/2012 - 30/06/2012</td><td>5 min</td><td>8.11%</td></tr><tr><td>PEMS-BAY</td><td>mph</td><td>325</td><td>01/01/2017 - 31/05/2017</td><td>5 min</td><td>0.00%</td></tr></table>

# 4.3. Model training and testing

To effectively forecast the unobserved nodes, an inductive training algorithm is developed. This method leverages random masking to simulate the test environment based on observed training data. By creating diverse input–output pairs, this approach enhances the model’s robustness and ensures it can generalize to new, unseen nodes during application. 

Suppose we have collected training data $\widetilde { \mathbf { X } } \in \mathbb { R } ^ { ( \widetilde { N } , L ) }$ from $\widetilde { N }$ observed locations over a time horizon of length ??. The sensor network can be represented as a graph $\widetilde { \mathcal { G } } = ( \widetilde { \mathcal { V } } , \widetilde { \mathbf { A } } )$ , where $\widetilde { \mathcal { V } } = \{ \widetilde { \nu } _ { 1 } , \widetilde { \nu } _ { 2 } , \dots , \widetilde { \nu } _ { \widetilde { N } } \}$ denotes the set of nodes corresponding to the sensors. The inductive training algorithm is detailed in Algorithm 1. The core distinction of our training algorithm compared to conventional traffic prediction training algorithms lies in the use of random masking for each batch of data (lines 9–11 in Algorithm 1). Inspired by Wu et al. (2021a), we randomly select a subset of nodes and set their features to zero in each batch. This operation not only simulates the data conditions during model application for predicting unobserved nodes, but also, by masking different nodes in different batches, replicates sensor failures and additions, forming the cornerstone of the model’s successful application to iFun. In addition, to enhance the model training, scheduled sampling (Bengio et al., 2015) with linear decay strategy is employed. Specifically, we initially feed the true observation sequence within the prediction horizon into the decoder, i.e., teacher forcing, with a high probability, then decrease this probability linearly, and finally switch to using the decoder’s prediction as the next step input, as shown in the third panel of Fig. 2. 

During inference, random masking is no longer required. For unobserved nodes of interest, we assume the presence of virtual sensors (Liu et al., 2009; Martin et al., 2021). The graph $\widetilde { G }$ is replaced by the complete graph $G$ , which includes both real and virtual sensors. The input $\widetilde { \mathbf { X } }$ is replaced by the features of the currently observed and unobserved nodes, denoted as $\mathbf { X }$ , with the features of the unobserved nodes set to zero. The mathematical formulation for the inference is as follows: 

$$
\mathbf {y} ^ {-} = f _ {\theta *} (\mathbf {X}, \mathcal {G}) \tag {14}
$$

where $\theta ^ { * }$ represents the parameters of the trained model. 

# 5. Numerical experiments

In this section, several experiments are set to answer the following question: 

Q1: Prediction performance (Section 5.2). Does the prediction performance of MoGERNN outperform other benchmark methods considering unobserved nodes? 

Q2: Adaptability to the structural changes of sensing networks (Section 5.3). Is MoGERNN capable of adapting to the possible structural changes in sensor networks, and how does its performance differ from the re-trained model? 

Q3: Impact of the VS-to-AAS ratio (Section 5.4). How does the prediction performance of MoGERNN at unobserved locations vary with the ratio of virtual to available sensors? 

Q4: Ablation study (Section 5.5). Are MoGE and Encoder-Decoder components effective? 

Q5: Parameters study (Section 5.6. How the number of activated experts affects performance and computation time? 

# 5.1. Experimental setup

Dataset. Two widely used open-source traffic speed datasets are evaluated, including METR-LA and PEMS-BAY from Li et al. (2018). Descriptions for the datasets are as shown in Table 1. 

Baselines. We selected eight methods as benchmarks. These baselines can be categorized into four groups: the combination of Kriging and local traffic prediction methods (OKriging+ED, KNN+ED), state-of-the-art local traffic prediction STGNNs adapted for the iFun scenario (TGCN, GMSDR), modified inductive GNN-based Kriging approaches (IGNNK, INCREASE), and methods specifically designed for the iFun problem (FUNS, UIGNN). The detailed is as follows: 

(1) OKriging1 (Saito et al., 2005) with ED: This method combines Ordinary Kriging with encoder-decoder architecture in MoGERNN. The approach first employs Ordinary Kriging to interpolate values at unobserved locations using Gaussian variogram functions to model spatial relationships. After training ED inductively, we feed both the Kriging-interpolated values and observed measurements into ED to predict future states for both observed and unobserved nodes. 

(2) KNN2 with ED: Similar to the first baseline, OKriging is replace by K-nearest neighbors. KNN performs interpolation based on the average of k-nearest neighbors. Neighbors are defined as different samples that exhibit ‘‘feature similarity’’. Since the speed feature of unobserved nodes are completely missing, we utilize auxiliary features such as latitude, longitude, and timestamps to facilitate interpolation for these unobserved nodes. 

(3) TGCN3 (Zhao et al., 2020): TGCN integrates graph convolution network and temporal convolution network to model the spatio-temporal dynamics of traffic. As it adopts static-graph, we can modified it towards iFun scenario by using the proposed inductive training procedure. 

(4) GMSDR4 (Liu et al., 2022a): GMSDR is a spatial–temporal forecasting framework that captures long-range dependencies by explicitly incorporating multiple historical states in its graph-based architecture. Since GMSDR employs node embeddings as part of its architecture, we train it in a transductive manner to learn node-specific representations. 

(5) IGNNK5 (Wu et al., 2021a): IGNNK develops an inductive training procedure to estimate the unobserved locations state, the main block is consisted by three diffusion graph convolution layers using bidirectional random walk diffusion kernel. Instead of training the model to perform traditional Kriging interpolation, we modify IGNNK learning objective to directly predict future states at both observed and unobserved nodes, making it compatible with the iFun problem setting. 

(6) INCREASE6 (Zheng et al., 2023) : Inductive graph representation learning model for spatio-temporal Kriging, featuring GNNbased spatial aggregation and GRU-based temporal modeling to capture heterogeneous spatial relations with various temporal patterns for interpolating state from unseen nodes. We modify it as same as baseline (5). 

(7) FUNS7 (Roth and Liebig, 2022): An inductive spatio-temporal graph neural network (STGNN) that combines a Message Passing Neural Network (MPNN) (Gilmer et al., 2017) framework with a Gated Recurrent Unit (GRU), where the message aggregation function employs the variant of graph attention network (GAT), GATv2 (Brody et al., 2022). 

(8) UIGNN8 (Mei et al., 2023): An uncertainty-aware inductive graph neural network framework that utilizes Diffusion Graph Convolution Networks (DGCN), which is an extension method of IGNNK proposed by Wu et al. (2021a). To ensure a fair comparison, the independent uncertainty quantification module in UIGNN is toggled off. 

Metrics. All the methods are evaluated by three commonly adopted metrics in time series prediction: Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). It is important to note that original missing values in the dataset are excluded from the evaluation. 

Implementation. The proposed MoGERNN is implemented with PyTorch 2.2.1 on an NVIDIA RTX 4090 GPU. The historical $P$ and prediction horizons $F$ are both fixed in 12 steps (1 h). Mean square error is adopted as the loss function. The dataset is divided into training and testing sets in a 7:3 ratio. The validation set is kept the same as the training set and used for early stopping with patience set to 10. We collect data using a sliding window with a stride of 12. In the training procedure, the random mask rate is set as $2 5 \%$ . Following Li et al. (2018), the diffusion convolution aggregator is chosen to be the Agg in Encoder-Decoder. To validate model performance for the iFun problem, different ratios of Virtual Sensors (VS), Failed Sensors (FS), Newly Added Sensors (NAS) and Always Available Sensors (AAS) are set. For clarity, we only give the formal definition of each type of sensors here, the ratio of them will be illustrated in each case study. The code for MoGERNN implementation is publicly available at: https://github.com/ZJU-TSELab/MoGERNN. 

• VS: Sensors installed in never unobserved nodes and used solely for evaluation during the test phase. 

• FS: Sensors that function normally during the training phase but are excluded in input during the testing phase. 


Table 2 Performance at three prediction horizons on METR-LA and PEMS-BAY. The best result is highlighted in bold, while the second-best result is underlined. The ‘‘Improve’’ illustrates the percentage of relative performance improvement of MOGERNN compared to the most competitive baseline.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Sensor</td><td rowspan="2">Model</td><td colspan="3">15 min</td><td colspan="3">30 min</td><td colspan="3">60 min</td></tr><tr><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="20">METR-LA</td><td rowspan="10">Virtual sensor</td><td>OKriging+ED</td><td>21.71</td><td>8.22</td><td>11.21</td><td>22.33</td><td>8.43</td><td>11.47</td><td>22.68</td><td>8.83</td><td>11.91</td></tr><tr><td>KNN+ED</td><td>21.05</td><td>7.97</td><td>13.24</td><td>20.72</td><td>7.91</td><td>12.84</td><td>20.06</td><td>7.88</td><td>12.49</td></tr><tr><td>GMSDR</td><td>25.10</td><td>9.25</td><td>11.95</td><td>25.49</td><td>9.49</td><td>12.15</td><td>24.72</td><td>9.54</td><td>12.18</td></tr><tr><td>TGCN</td><td>21.43</td><td>7.60</td><td>10.76</td><td>22.17</td><td>7.82</td><td>11.05</td><td>22.43</td><td>8.06</td><td>11.32</td></tr><tr><td>IGNNK</td><td>17.58</td><td>6.64</td><td>9.82</td><td>18.57</td><td>6.90</td><td>10.21</td><td>19.41</td><td>7.31</td><td>10.66</td></tr><tr><td>INCREASE</td><td>20.20</td><td>8.50</td><td>12.30</td><td>19.91</td><td>8.43</td><td>12.31</td><td>19.15</td><td>8.26</td><td>12.19</td></tr><tr><td>UIGNN</td><td>16.70</td><td>6.62</td><td>9.89</td><td>17.88</td><td>6.93</td><td>10.29</td><td>18.92</td><td>7.33</td><td>10.76</td></tr><tr><td>FUNS</td><td>21.64</td><td>8.12</td><td>11.37</td><td>21.50</td><td>8.04</td><td>11.30</td><td>20.51</td><td>7.74</td><td>10.93</td></tr><tr><td>MoGERNN (ours)</td><td>14.08</td><td>5.83</td><td>8.64</td><td>15.06</td><td>5.97</td><td>8.88</td><td>15.91</td><td>6.19</td><td>9.38</td></tr><tr><td>Improve (%)</td><td>+15.69</td><td>+11.93</td><td>+12.02</td><td>+15.77</td><td>+13.48</td><td>+13.03</td><td>+15.91</td><td>+15.32</td><td>+12.01</td></tr><tr><td rowspan="10">Always available sensor</td><td>OKriging+ED</td><td>8.38</td><td>3.10</td><td>5.37</td><td>11.44</td><td>3.86</td><td>6.75</td><td>14.56</td><td>4.85</td><td>8.31</td></tr><tr><td>KNN+ED</td><td>8.30</td><td>3.09</td><td>5.35</td><td>10.89</td><td>3.84</td><td>6.54</td><td>13.11</td><td>4.52</td><td>7.87</td></tr><tr><td>GMSDR</td><td>9.44</td><td>3.55</td><td>6.09</td><td>11.14</td><td>4.01</td><td>6.90</td><td>13.86</td><td>4.78</td><td>8.00</td></tr><tr><td>TGCN</td><td>13.81</td><td>5.32</td><td>7.74</td><td>15.94</td><td>5.84</td><td>8.66</td><td>18.48</td><td>6.61</td><td>9.85</td></tr><tr><td>IGNNK</td><td>8.50</td><td>3.34</td><td>5.76</td><td>10.96</td><td>3.98</td><td>7.06</td><td>14.11</td><td>4.95</td><td>8.58</td></tr><tr><td>INCREASE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UIGNN</td><td>8.42</td><td>3.31</td><td>5.72</td><td>10.91</td><td>3.95</td><td>6.95</td><td>13.98</td><td>4.85</td><td>8.45</td></tr><tr><td>FUNS</td><td>16.27</td><td>5.66</td><td>9.11</td><td>16.52</td><td>5.73</td><td>9.18</td><td>15.84</td><td>5.53</td><td>8.88</td></tr><tr><td>MoGERNN (ours)</td><td>8.07</td><td>3.04</td><td>5.31</td><td>10.45</td><td>3.61</td><td>6.45</td><td>12.74</td><td>4.24</td><td>7.66</td></tr><tr><td>Improve (%)</td><td>+2.77</td><td>+1.62</td><td>+0.75</td><td>+4.04</td><td>+5.99</td><td>+0.16</td><td>+2.82</td><td>+6.19</td><td>+2.67</td></tr><tr><td rowspan="20">PEMS-BAY</td><td rowspan="10">Virtual sensor</td><td>OKriging+ED</td><td>11.19</td><td>4.83</td><td>7.86</td><td>11.41</td><td>4.88</td><td>8.03</td><td>11.55</td><td>4.98</td><td>8.39</td></tr><tr><td>KNN+ED</td><td>13.91</td><td>5.89</td><td>11.04</td><td>13.53</td><td>5.55</td><td>10.33</td><td>12.95</td><td>5.18</td><td>9.67</td></tr><tr><td>GMSDR</td><td>12.82</td><td>5.67</td><td>8.27</td><td>13.11</td><td>5.80</td><td>8.45</td><td>12.54</td><td>5.49</td><td>8.31</td></tr><tr><td>TGCN</td><td>9.51</td><td>3.94</td><td>6.78</td><td>10.12</td><td>4.09</td><td>7.11</td><td>10.87</td><td>4.36</td><td>7.61</td></tr><tr><td>IGNNK</td><td>8.05</td><td>3.59</td><td>6.12</td><td>8.77</td><td>3.81</td><td>6.29</td><td>9.78</td><td>4.18</td><td>6.98</td></tr><tr><td>INCREASE</td><td>9.46</td><td>4.02</td><td>6.85</td><td>9.53</td><td>4.04</td><td>6.87</td><td>9.57</td><td>4.13</td><td>6.97</td></tr><tr><td>UIGNN</td><td>8.24</td><td>3.67</td><td>5.92</td><td>8.92</td><td>3.88</td><td>6.31</td><td>9.83</td><td>4.22</td><td>6.94</td></tr><tr><td>FUNS</td><td>10.98</td><td>4.72</td><td>7.67</td><td>10.90</td><td>4.75</td><td>7.65</td><td>10.75</td><td>4.72</td><td>7.69</td></tr><tr><td>MoGERNN (ours)</td><td>7.58</td><td>3.56</td><td>5.95</td><td>8.10</td><td>3.68</td><td>6.20</td><td>8.88</td><td>3.92</td><td>6.72</td></tr><tr><td>Improve (%)</td><td>+5.84</td><td>+0.84</td><td>-0.51</td><td>+7.64</td><td>+3.41</td><td>+1.53</td><td>+7.21</td><td>+5.08</td><td>+3.17</td></tr><tr><td rowspan="10">Always available sensor</td><td>OKriging+ED</td><td>3.43</td><td>1.57</td><td>3.13</td><td>5.10</td><td>2.06</td><td>4.62</td><td>6.99</td><td>2.67</td><td>6.08</td></tr><tr><td>KNN+ED</td><td>3.52</td><td>1.61</td><td>3.18</td><td>5.21</td><td>2.11</td><td>4.73</td><td>7.28</td><td>2.75</td><td>6.23</td></tr><tr><td>GMSDR</td><td>4.86</td><td>2.18</td><td>4.10</td><td>5.59</td><td>2.46</td><td>4.66</td><td>6.88</td><td>2.89</td><td>5.47</td></tr><tr><td>TGCN</td><td>6.50</td><td>2.91</td><td>4.91</td><td>7.62</td><td>3.24</td><td>5.69</td><td>9.50</td><td>3.83</td><td>6.91</td></tr><tr><td>IGNNK</td><td>3.40</td><td>1.63</td><td>3.12</td><td>5.11</td><td>2.19</td><td>4.44</td><td>7.52</td><td>3.02</td><td>6.00</td></tr><tr><td>INCREASE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UIGNN</td><td>3.37</td><td>1.63</td><td>3.06</td><td>4.92</td><td>2.16</td><td>4.34</td><td>7.09</td><td>2.94</td><td>5.83</td></tr><tr><td>FUNS</td><td>8.06</td><td>3.24</td><td>6.37</td><td>7.82</td><td>3.21</td><td>6.30</td><td>7.56</td><td>3.11</td><td>6.30</td></tr><tr><td>MoGERNN (ours)</td><td>3.20</td><td>1.54</td><td>2.93</td><td>4.55</td><td>1.96</td><td>4.06</td><td>6.22</td><td>2.50</td><td>5.29</td></tr><tr><td>Improve (%)</td><td>+5.04</td><td>+1.91</td><td>+4.25</td><td>+7.52</td><td>+4.85</td><td>+6.45</td><td>+9.59</td><td>+6.37</td><td>+3.29</td></tr></table>


• NAS: Sensors that are newly added in the test phase. 



• AAS: Sensors that are always available during train and test. 


# 5.2. Prediction performance on unobserved and observed locations (Q1)

In this experiment, $2 5 \%$ sensors are excluded randomly during training so as to play a VS role in the test, with the remaining sensors serving as AAS. Table 2 summarizes the performance over 15, 30, and 60-minute prediction horizons. While our study focuses more on the predictions for VS, we also present the performance of AAS to comprehensively show the performance of the proposed method. As shown in Table 2, our MoGERNN outperforms baseline models across all prediction horizons on both datasets for unobserved locations, with particularly significant improvements on METR-LA. Additionally, our model maintains comprehensive advantages in most cases for future state prediction at observed stations. 

Among the baseline methods, the concatenated approaches combining kriging with deep learning models (OKriging+ED, ${ \bf K N N + E D } _ { \it 4 } ^ { \prime }$ ) show inferior performance. While previous research has shown that KNN and OKriging perform competitively in spatial extrapolation tasks (Wu et al., 2021a), these concatenated approaches struggle with spatiotemporal extrapolation due to error propagation between multiple independent models. For conventional traffic prediction models, despite GMSDR’s state-of-the-art performance in standard traffic prediction through dynamic graph construction, its performance deteriorates significantly in the iFun task, as it fails when unobserved nodes lack historical data for node embedding training. TGCN, while using a static graph structure, performs better than GMSDR at unobserved locations due to its preserved spatial connectivity information, though its 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/21e2f4d195d8e7240851e0e38dfc0471871b018a2098e3126ffe89c74f0922fa.jpg)



Fig. 3. Map presentation of prediction performance in an evening-peak time point (2012-05-24 17:30) of METR-LA.


overall performance is limited by the over-smoothing issue. In terms of methods extending inductively kriging approaches, IGNNK shows strong generalization capability to unseen nodes through its diverse subgraph construction strategy. However, its performance is constrained by inadequate temporal dynamics modeling. INCREASE, despite considering both spatial and temporal dynamics, exhibits inferior performance due to its limited diversity in subgraph samples during training. Furthermore, the method is restricted to predictions for unobserved nodes only. For methods specifically designed for iFun, FUNS’s performance is hampered by its ineffective attention mechanism when dealing with unobserved nodes lacking historical data. As an extension of IGNNK, UIGNN incorporates both Kriging and prediction errors to guide model training for forecasting future states of both observed and unobserved nodes. While this approach demonstrates superior performance compared to IGNNK and achieves the second-best results, it shares IGNNK’s limitation of inadequate temporal dynamics modeling, thus performing worse than our proposed MoGERNN. 

Focusing on the critical task of predicting unobserved nodes within the METR-LA dataset, our method achieves significant reductions in key error metrics relative to the most competitive baseline, with decreases of $1 6 \%$ in MAPE, $1 4 \%$ in MAE, and $^ { 1 3 \% }$ in RMSE for all prediction horizon in average. Moreover, the prediction accuracy gap between AAS and VS reveals that predicting unobserved locations poses significant challenges. Based on the predictions generated by our model, the error for unobserved locations is 1.2–1.9 times that of observed locations. This substantial difference underscores the complexity inherent in extrapolating to unseen spatial points and highlights the need for advanced methodologies in addressing such prediction tasks. 

Fig. 3 illustrates the prediction performance of various models during an evening rush hour (May 24, 2012, 17:30) for the METR-LA dataset. The first column depicts the sensor measurements at this time point, while the remaining three columns represent the speed predictions of each model based on historical data. For clarity, prediction results for different sensors are presented in different rows. It is evident that FUNS consistently underperforms compared to the other two models, especially when the speed is lower than 50 mph. UIGNN and MoGERNN showed comparable strong prediction performance at locations where physical sensors are deployed. Our approach, however, shows improved accuracy in both high-speed and low-speed scenarios at unobserved locations. This superiority in performance is notably visible in the results highlighted within the red frames in the third-row subplot. 

Given the paramount importance of congestion evolution in traffic management tasks, the data from METR-LA that exhibit significant congestion characteristics are selected for further analysis. As shown in Fig. 4, it illustrates a comparison between observed and predicted time series for selected sensors. Specifically, Fig. 4(a) displays results for unobserved locations, while Fig. 4(b) shows results for observed locations. We can find that our MoGERNN demonstrates superior performance in predicting congestion evolution, with this advantage being particularly pronounced at unobserved locations compared to other baselines. 

As previously described, MoGERNN constructs diverse spatial aggregation patterns through multiple graph experts and employs a sparse gating network to dynamically assign and fuse the most suitable experts, thus enhancing the exiting works. To provide insights into this dynamic mechanism, we visualized the expert activation probabilities (aggregated across all nodes) at different time points during the test period from June 26 to June 28, 2012, on the METR-LA dataset, as shown in Fig. 5. The results reveal significant variations in expert activation probabilities across different time points, demonstrating MoGERNN’s ability to dynamically adjust spatial modeling strategies based on traffic conditions. In the long-term statistics, the weighted average graph expert exhibits the highest activation probability, which aligns with the fundamental understanding of spatial continuity in traffic flow. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ed66c41664641ea3e547ad7e4d5077a9a9ec097ecedcf280f4efe51f6174c602.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/c714de5b2a54cfb7b0612557833fe7e56ff809c24ba125a9e00dfe2c498d4d33.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/9696be3f9041bba15c7e5b61e7ec7b3e018b24bd2711eab98b765f436a392ddd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/6060dbecaaf0fbf07f3a56ff8c218d4c98dd7c63b5c766554aae417fff9e5289.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/a4c616277ae02e0a1bb791370b8fd7908043586ad0af223a55cd01f0ba8ef60c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/27f421d6d294d0c9f2686b36da88a66fe7c477fe507b27d9768194453dbf3f5b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/eca8ce6d4dda3ec7c3435d9a1d6e408250de9f6badfd174a0030c660afb69558.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/21202542da753f53372179e3e0d8a231ed4dd9737c8b57319677ccaa4fec82a7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/8f1f9710d4fe3ae19d2c48850077deab71711e05fcf833ca7ca75eb3f3f23c87.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/33d482edb9876617bf4e1dfe4c8bba69a8b3fca36a5bc1748e83ebb34f226eb0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/c84524417f67798c7cb90d30c1507440b8f30dc78e6a3abd4e2b32c2349bda7c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/d5c594e065a8011a02af456af59226bd793ea7000f65a393bc071cac86721510.jpg)



(a) Unobserved locations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ac1e6db8f7b71c049059d6577f8e074ff13e8c4802dac093a95097a044534d43.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/5dedb60ab3a9919d26aae1be53bef400a3da7519defaa418068489e29304a0fd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/154be23c97ac4821b637011d007842aa1041ac54d898468724f658419f428be0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/6fa7cae6262c0a25298d55d939296f230e3a3d27652bfa3362abeb0e6a8257c4.jpg)



(b) Observed location.



Fig. 4. Predicting results for METR-LA. (a) show the results of unobserved locations, including three virtual sensors (Sensor $\# 1 9 3$ , $\# 9 8$ , #70). (b) show the results of observed locations, including one physical sensor (Sensor $\# 6 8 \mathrm { . }$ ).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ff07f97126c724bb9f6650b399269c79843aa74f6ea73a823b10101a20217ea9.jpg)



Fig. 5. Average expert activation probability across all nodes over time for test dataset in METR-LA.


To further illustrate the mechanism of MoGE, we analysis expert activation probabilities in four typical scenarios, as shown in Fig. 6. Fig. 6a and Fig. 6b display the expert activation probability distributions in two local regions, while Fig. 6c and Fig. 6d focus on a single location marked by a red box. In the region shown in Fig. 6a, traffic states exhibit high spatial smoothness, consistent with the ‘‘stronger correlation at closer distances’’ characteristic, leading to the highest activation probability for the weighted average expert. In contrast, the region in Fig. 6b shows uniformly smooth traffic flow with minimal speed variations, resulting in significantly reduced activation of the weighted average expert while maximum value, minimum value experts, and diffusion models play more crucial roles. In Fig. 6c, where surrounding nodes experience congestion while the boxed location remains smooth, the activation probabilities of mean and weighted mean experts decrease significantly. The model relies mainly on maximum and minimum experts instead, as the maximum expert captures related smooth traffic states from more distant locations, while the minimum expert provides maximum information gain by capturing the most distinct states. A similar scenario is observed in Fig. 6d, where the boxed location experiences low-speed conditions, and the model primarily activates the minimum expert to capture the relevant congested nodes in the surrounding area. These observations demonstrate how MoGERNN flexibly adjusts its spatial modeling strategy across different traffic scenarios for accurate prediction. 

# 5.3. Adaptability to the structural changes of sensing networks (Q2)

To simulate the structural changes in sensor network, we extend the sensor configuration from Section 5.2 by introducing NS and FS. Table 3 presents the quantities of all four sensor types under this new configuration. Given the aims to evaluate the model’s adaptability to changes in sensor configurations, we directly apply the model trained on the sensor configuration from Q1 to the current sensing environment for testing. In addition, to comprehensively assess the model’s adaptability to the changes of sensor 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/ec81ac013c8ffe0cce4345b5d2ee98c5b9f6d5d4818e6cd78d4984063cf894bc.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/2aba8d31d86698a6eb2d07febc4447d00ffee1b789ab8ca0b674f674fe01d38a.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/39d82e8a2a4608475d883f4bf77e2adcab6ec6b5013dbb7c225fbf1fa1cb04a4.jpg)



(c)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/29d5d05f57e322cb62b2074a052614d33259ae165032b6bddc43f67d8f6eefda.jpg)



(d)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/b3179b6eb7bb6c05911cecfbbac30307db35edac82b0f8966c7ab00b7fef6012.jpg)



Fig. 6. Analysis of expert activation patterns on METR-LA dataset. (a) and (b) Expert activation probabilities in a local region. (c) and (d) Expert activation probability at a single location (red box).



Table 3 Sensor configuration to simulate the structural changes of sensing network.


<table><tr><td>Dataset</td><td>Always Available Sensors (AAS)</td><td>Virtual Sensors (VS)</td><td>New Added Sensors (NAS)</td><td>Failed Sensors (FS)</td></tr><tr><td>METR-LA</td><td>137</td><td>30</td><td>20</td><td>20</td></tr><tr><td>PEMS-BAY</td><td>196</td><td>65</td><td>32</td><td>32</td></tr></table>

networks, we also compare its performance with that of a model retrained on the new sensor network configuration. The average prediction performance of all prediction horizons is presented in Table 4. For brevity, we present only the results of UIGNN and FUNS for comparison. The results demonstrate that our model outperforms these baselines in the majority of scenarios across all four settings. This advantage is particularly pronounced in VS and FS scenarios, highlighting MoGERNN’s superior spatial extrapolation capabilities. Moreover, despite the absence of NAS training data, MoGERNN achieves comparable performance between NAS and AAS, indicating satisfactory zero-shot prediction capability. 

In comparing the performance of MoGERNN with its retrained counterpart, shown in Table 5, we observe that the retrained version achieves slightly better predictive accuracy. However, the moderate performance gap suggests that MoGERNN maintains robust predictive capability even without retraining. This adaptability to different sensor configurations offers practitioners the flexibility to make retraining decisions based on operational requirements and resource constraints, thereby enhancing the model’s practical utility in real-world applications. 

# 5.4. Impact of the VS-to-AAS ratio (Q3)

To examine the impact of the VS-to-AAS ratio on model performance, we randomly sample 50 to 175 VS from the METR-LA dataset. For simplicity, NAS and FS are not considered in this analysis. The performance trends of the model as the VS-to-AAS ratio changes are illustrated in Fig. 7. In addition to the VS-to-AAS ratio, we introduce an additional metric, Average VS-to-AAS Distance $D _ { v 2 a }$ , to better characterize the spatial extrapolation capability of the model, as shown in purple ticks in Fig. 7. The definition of $D _ { v 2 a }$ is as follows: 

$$
d _ {\min } ^ {i} = \min  _ {j: j \in \mathrm {A A S}, i \neq j} d _ {i, j} \tag {15}
$$

$$
D _ {v 2 a} = \operatorname {A v g} _ {i \in \mathrm {V S}} d _ {\text {m i n}} ^ {i} \tag {16}
$$

According to Fig. 7, it is evident that our method consistently outperforms our main competitor, UIGNN, across various VSto-AAS ratios. Given the same level of predictive performance, our approach typically requires at least 25 fewer sensors than UIGNN. Furthermore, based on the MAE and $D _ { v 2 a }$ values, we can observe that the spatial extrapolation capability of our model is approximately 1.4 times that of UIGNN. 


Table 4 Average prediction performance of all prediction horizons under the structural changes in sensing network.


<table><tr><td rowspan="2">Sensor</td><td rowspan="2">Models</td><td colspan="3">METR-LA</td><td colspan="3">PEMS-Bay</td></tr><tr><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="4">Virtual sensor</td><td>UIGNN</td><td>16.13</td><td>6.45</td><td>9.74</td><td>8.71</td><td>3.90</td><td>6.28</td></tr><tr><td>FUNS</td><td>20.53</td><td>7.60</td><td>10.88</td><td>10.60</td><td>4.67</td><td>7.61</td></tr><tr><td>MoGERNN (ours)</td><td>14.77</td><td>6.01</td><td>8.92</td><td>7.73</td><td>3.59</td><td>5.94</td></tr><tr><td>Improve (%)</td><td>+8.43</td><td>+6.82</td><td>+8.42</td><td>+11.25</td><td>+7.95</td><td>+5.41</td></tr><tr><td rowspan="4">Always available sensor</td><td>UIGNN</td><td>10.80</td><td>3.92</td><td>6.96</td><td>4.98</td><td>2.17</td><td>4.44</td></tr><tr><td>FUNS</td><td>16.17</td><td>5.60</td><td>9.08</td><td>7.71</td><td>3.18</td><td>6.39</td></tr><tr><td>MoGERNN (ours)</td><td>10.56</td><td>3.64</td><td>6.59</td><td>4.60</td><td>1.94</td><td>4.15</td></tr><tr><td>Improve (%)</td><td>+2.22</td><td>+7.14</td><td>+5.32</td><td>+7.63</td><td>+10.60</td><td>+6.53</td></tr><tr><td rowspan="4">New added sensor</td><td>UIGNN</td><td>11.16</td><td>4.00</td><td>7.12</td><td>5.11</td><td>2.12</td><td>4.50</td></tr><tr><td>FUNS</td><td>16.13</td><td>5.66</td><td>9.91</td><td>7.74</td><td>3.05</td><td>6.31</td></tr><tr><td>MoGERNN (ours)</td><td>11.38</td><td>4.04</td><td>6.94</td><td>5.02</td><td>2.08</td><td>4.54</td></tr><tr><td>Improve (%)</td><td>-1.97</td><td>-1.00</td><td>+2.53</td><td>+1.76</td><td>1.87</td><td>-0.09</td></tr><tr><td rowspan="4">Failed sensor</td><td>UIGNN</td><td>18.50</td><td>6.40</td><td>9.82</td><td>9.27</td><td>3.80</td><td>6.40</td></tr><tr><td>FUNS</td><td>22.31</td><td>8.02</td><td>11.28</td><td>11.38</td><td>4.52</td><td>7.70</td></tr><tr><td>MoGERNN (ours)</td><td>13.24</td><td>5.00</td><td>7.78</td><td>7.27</td><td>3.12</td><td>5.49</td></tr><tr><td>Improve (%)</td><td>+28.43</td><td>+21.88</td><td>+20.77</td><td>+21.57</td><td>+17.89</td><td>+14.22</td></tr></table>


Table 5 Performance comparison of MoGERNN with its retrained counterpart after sensor change.


<table><tr><td rowspan="2">Sensor</td><td rowspan="2">Models</td><td colspan="3">METR-LA</td><td colspan="3">PEMS-Bay</td></tr><tr><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="2">Virtual sensor</td><td>MoGERNN</td><td>14.77</td><td>6.01</td><td>8.92</td><td>7.73</td><td>3.59</td><td>5.94</td></tr><tr><td>MoGERNN-retrained</td><td>14.51</td><td>5.92</td><td>9.74</td><td>7.93</td><td>3.60</td><td>6.05</td></tr><tr><td rowspan="2">Always available sensor</td><td>MoGERNN</td><td>10.56</td><td>3.64</td><td>6.59</td><td>4.60</td><td>1.94</td><td>4.15</td></tr><tr><td>MoGERNN-retrained</td><td>10.02</td><td>3.43</td><td>6.26</td><td>4.44</td><td>1.90</td><td>4.00</td></tr><tr><td rowspan="2">New added sensor</td><td>MoGERNN</td><td>11.01</td><td>4.04</td><td>6.94</td><td>5.02</td><td>2.08</td><td>4.54</td></tr><tr><td>MoGERNN-retrained</td><td>10.81</td><td>3.83</td><td>6.30</td><td>4.34</td><td>1.84</td><td>4.02</td></tr><tr><td rowspan="2">Failed sensor</td><td>MoGERNN</td><td>13.24</td><td>5.00</td><td>7.78</td><td>7.27</td><td>3.12</td><td>5.49</td></tr><tr><td>MoGERNN-retrained</td><td>12.69</td><td>4.45</td><td>7.24</td><td>6.79</td><td>2.93</td><td>5.14</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/88563cf13261de3293a02dbabe560792672dad90632ff77edfa5bd50f38a65d3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/eac917ee51c227e3d1b71e9751e449025098a5c6946b0f4a48820ed5aa11a303.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/3cb41b6d184831d2465af596fe837a908d50f935ce97c1d37c7cca7bf86f2775.jpg)



Fig. 7. Model performance under different ratios of VS to AAS. The first row of $\mathbf { X } \cdot$ -ticks labels indicates the number of VS, while the second row shows the number of AAS.


# 5.5. Ablation study (Q4)

To evaluate the effectiveness of the Encoder-Decoder and MoGE in our model, we conduct the following ablation studies, with results consolidated in Table 6. For simplicity, we present only the average results across all prediction horizons on the METR-LA dataset, without consideration of NAS and FS. The sensor type configuration remains consistent with that in Section 5.2. 

(1) Removal of the Encoder-Decoder Module: The first experiment involves removing the encoder-decoder module. Results indicate that this module positively influences predictions for both AAS and VS. 

(2) Removal of the MoGE Module: In the second experiment, we removed the MoGE module. while AAS performance shows modest improvement with a $3 . 2 \%$ reduction in MAPE, the VS prediction accuracy substantially deteriorates, exhibiting increases of $6 . 9 \%$ in MAPE, $1 0 . 2 \%$ in MAE, and $8 . 9 6 \%$ in RMSE. These results demonstrate that the MoGE module plays a crucial role in maintaining the model’s capability to accurately predict at unobserved locations, despite a slight trade-off in performance at observed nodes. This trade-off stems from the removal of self-loops in MoGE’s graph expert operations (reflected in Eq. (4)), which forces nodes to construct their representations primarily through neighboring nodes’ features rather than their own historical data. While 


Table 6 Effect of Encoder-Decoder and MoGE module.


<table><tr><td rowspan="2">Models</td><td colspan="3">Always available sensor (AAS)</td><td colspan="3">Virtual sensor (VS)</td></tr><tr><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td></tr><tr><td>w/o ED</td><td>11.63</td><td>4.15</td><td>7.32</td><td>18.13</td><td>6.66</td><td>9.82</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+11.95</td><td>+13.98</td><td>+12.02</td><td>+17.10</td><td>+10.21</td><td>+8.96</td></tr><tr><td>w/o MoGE</td><td>9.93</td><td>3.58</td><td>6.42</td><td>16.14</td><td>6.77</td><td>9.98</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>-3.12</td><td>+0.28</td><td>-0.31</td><td>+6.88</td><td>+11.67</td><td>+10.42</td></tr><tr><td>replace MoGE with mean</td><td>10.47</td><td>3.61</td><td>6.55</td><td>16.31</td><td>6.50</td><td>9.62</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+2.20</td><td>+1.11</td><td>+1.68</td><td>+7.85</td><td>+8.00</td><td>+7.07</td></tr><tr><td>replace MoGE with weighted-mean</td><td>10.31</td><td>3.60</td><td>6.48</td><td>15.98</td><td>6.28</td><td>9.33</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+0.68</td><td>+0.83</td><td>+0.62</td><td>+5.94</td><td>+4.78</td><td>+4.18</td></tr><tr><td>replace MoGE with max-pooling</td><td>10.79</td><td>3.63</td><td>6.61</td><td>15.75</td><td>6.11</td><td>9.32</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+5.10</td><td>+1.65</td><td>+2.57</td><td>+4.57</td><td>+2.13</td><td>+4.08</td></tr><tr><td>replace MoGE with min-pooling</td><td>10.89</td><td>3.67</td><td>6.62</td><td>16.11</td><td>6.17</td><td>9.42</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+5.97</td><td>+2.72</td><td>+2.72</td><td>+6.70</td><td>+3.08</td><td>+5.10</td></tr><tr><td>replace MoGE with DGCN</td><td>10.43</td><td>3.63</td><td>6.49</td><td>16.04</td><td>6.41</td><td>9.43</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>+1.82</td><td>+1.65</td><td>+0.77</td><td>+6.30</td><td>+6.71</td><td>+5.20</td></tr><tr><td>replace sparse gating with average sums</td><td>9.96</td><td>3.58</td><td>6.45</td><td>15.97</td><td>6.81</td><td>9.82</td></tr><tr><td>MoGERNN (ours)</td><td>10.24</td><td>3.57</td><td>6.44</td><td>15.03</td><td>5.98</td><td>8.94</td></tr><tr><td>Improve (%)</td><td>-2.81</td><td>+0.28</td><td>+0.16</td><td>+5.89</td><td>+12.19</td><td>+8.96</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/c373ade1f8775bed286d848545f3e2c2c8ef8c6b7329bf3b85d565a2cf52b4d3.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-07/deea5b74-d368-4ce5-b0e3-4743b0d32d3d/96c8839ad195fa3a8ba2d27fbe6ba3fe657300e5ca36b30c5f3df56a5472394e.jpg)



(b)



Fig. 8. Impact of number of activated experts:(a) performance of state prediction on unobserved nodes in METR-LA dataset, (b) inference time the model.


this design choice is beneficial for unobserved nodes that lack historical information, it slightly compromises the performance at observed nodes where self-historical information would naturally be most relevant for prediction. A potential improvement could be adopting a multitask learning framework that treats observed and unobserved node prediction as separate but related tasks, allowing the model to better balance the performance between these two scenarios. 

(3) Replace MoGE with Single Graph Expert: To further investigate the role of multiple graph experts in the MoGE module, we conducted experiments retaining only one expert at a time. In this ablation setting, the model degenerates to a static graph structure without the ability to dynamically combine different spatial aggregation patterns. As shown in Table 6, the weighted-mean expert delivers the best performance for AAS predictions, whereas the max-pooling expert excels in VS predictions. MoGERNN, leveraging the strengths of multiple graph experts, achieves superior performance in both AAS and VS than a single graph expert. This confirms the importance of multi-graph message aggregators in the iFun problem. 

(4) Replace Sparse Gating With Average Sums: To validate the efficacy of using sparse gating networks, we replaced them with simple averaging. While average sum provides some improvement in AAS, it resulted in a noticeable decline in VS predictions compared to the max-pooling expert. In contrast, MoGERNN with sparse gating improved performance in both metrics. Although the MAPE for AAS with sparse gating was slightly higher than the average sum approach $( - 2 . 8 \%$ in MAPE), the VS prediction error decreased by $5 . 9 \%$ , $1 2 . 2 \%$ and $9 . 0 \%$ in MAPE, MAE and RMSE, respectively. This experiment confirms that there is value in designing advanced methods for integrating multi-graph experts. 

# 5.6. Parameters study (Q5)

In this experiment, we investigate the impact of varying numbers of active experts on both model performance and inference time. We conduct experiments on the METR-LA dataset, maintaining the same sensor type configuration as in Section 5.2. As shown in Fig. 8, subplot (a) presents the RMSE on unobserved nodes across all prediction horizon, while subplot (b) illustrates the inference time. The results demonstrate a U-shaped performance pattern as the number of active experts increases, with optimal performance achieved when activating three experts. Notably, the performance with just one activated expert is comparable to that with three experts. Referring to the results in Table 2, this single-expert performance also surpasses the baseline methods, demonstrating the effectiveness of our expert design. Meanwhile, the runtime increases linearly with the number of activated experts, aligning with expectations. Also, the inference time increase caused by multiple experts is acceptable- from activating a single expert to all five experts, the total inference time only increases from 0.014 s to 0.026 s, which is well within the requirements for real-time practice. 

# 6. Conclusion

In this paper, we propose MoGERNN, an inductive spatio-temporal graph learning framework that enables traffic state forecasting at unobserved locations. Our experiments demonstrate that the proposed method consistently outperforms four baseline models for both observed and unobserved nodes. For congestion states, our model accurately predicts congestion evolution based on historical data, even in areas without sensors. This provides richer and more effective information for traffic management and control compared to traditional traffic predictors. Additionally, our method adapts more effectively to the structural changes of sensor network, such as the addition or failure of sensors compared to baselines. Even when compared to MoGERNN retrained with new sensor configurations, the model trained on historical sensor configuration maintains competitive performance. Notably, the traffic state prediction accuracy for locations with newly added sensors closely matches that of locations with continuous data, highlighting our method’s superior zero-shot prediction capability. 

Several avenues for future research merit exploration. First, uncertainty estimation: the quantification of uncertainty is of paramount importance (Wen et al., 2023), particularly given that these predictions inform critical traffic management strategies. The need for this quantification becomes even more pronounced when dealing with unobserved nodes. Given that the spatial extrapolation capabilities of models are limited, arbitrary application to unobserved nodes may lead to erroneous decisions in traffic management. Different from nodes with observed data, model errors at unobserved nodes cannot be evaluated, so uncertainty quantification can assist us in understanding model performance at these locations. Second, sensor deployment analysis: identifying the relationship between sensor deployment patterns and model performance would help establish clear guidelines for practical applications. This includes understanding what types of sensor distributions allow reliable predictions and what patterns lead to degraded performance. Third, multi-source data fusion: the iFun problem can be ill-posed when the existing sensor deployment is too sparse, in which case we would like the model to be able to incorporate other available data sources to solve the iFun, e.g., connected vehicle data (Xu et al., 2024). With the support of multi-source data, we can also attempt to address more valuable and challenging problems, such as traffic flow estimation and prediction at unobserved locations (Nie et al., 2023b). Third, multivariate data processing capability: typically, fixed traffic sensors simultaneously measure three key variables: speed, flow, and occupancy. By extending the model to accommodate multivariate data, we could not only fully utilize the available data but also potentially enhance the model by introducing regularization constraints based on the prior dependencies among the multivariate variables, such as the fundamental diagram. 

# CRediT authorship contribution statement

Qishen Zhou: Writing – original draft, Conceptualization, Investigation, Methodology, Formal analysis, Data curation. Yifan Zhang: Writing – review & editing, Methodology, Investigation, Formal analysis. Michail A. Makridis: Writing – review & editing, Supervision, Project administration, Investigation, Conceptualization. Anastasios Kouvelas: Writing – review & editing, Supervision, Investigation. Yibing Wang: Writing – review & editing, Supervision, Investigation, Funding acquisition. Simon Hu: Writing – review & editing, Supervision, Resources, Project administration, Investigation, Funding acquisition, Conceptualization. 

# Acknowledgments

The authors would like to thank the anonymous reviewers for their valuable comments and suggestions that have significantly improved this paper. This work was supported in part by ‘‘Pioneer’’ and ‘‘Leading Goose’’ R & D Program of Zhejiang, China (2023C03155), State Key Laboratory (SKL) of Biobased Transportation Fuel Technology, China, the National Key R&D Program of China (2023YFB4302600), the National Natural Science Foundation of China (52131202, 72350710798), the Smart Urban Future (SURF) Laboratory, Zhejiang Province, China, Zhejiang University Global Partnership Fund, China, Zhejiang University Sustainable Smart Livable Cities Alliance (SSLCA), China, the Provincial Key R&D Program of Zhejiang, China (2024C01180; 2022C01129), the National Natural Science Foundation of China (52272315), Ningbo International Science and Technology Cooperation Program, China (2023H020). Qishen Zhou acknowledges support from the China Scholarship Council for his one-year visiting at ETH Zurich. 

# Data availability

The data that support the findings of this study are publicly available. They can be accessed at PerformanceMeasurementSystem(PeMS). 

# References



Appleby, G., Liu, L., Liu, L.P., 2020. Kriging convolutional networks. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, (04), pp. 3187–3194. http://dx.doi.org/10.1609/aaai.v34i04.5716. 





Aw, A., Rascle, M., 2000. Resurrection of ‘second order’ models of traffic flow. SIAM J. Appl. Math. 60 (3), 916–938. 





Bahadori, M.T., Yu, Q.R., Liu, Y., 2014. Fast multivariate spatio-temporal analysis via low rank tensor learning. In: Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N., Weinberger, K. (Eds.), In: Advances in Neural Information Processing Systems, vol. 27, Curran Associates, Inc.. 





Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N., 2015. Scheduled sampling for sequence prediction with recurrent neural networks. Adv. Neural Inf. Process. Syst. 28. 





Berndt, D.J., Clifford, J., 1994. Using dynamic time warping to find patterns in time series. In: Proceedings of the 3rd International Conference on Knowledge Discovery and Data Mining. pp. 359–370. 





Brody, S., Alon, U., Yahav, E., 2022. How attentive are graph attention networks? In: International Conference on Learning Representations. URL https: //openreview.net/forum?id=F72ximsx7C1. 





Cini, A., Marisca, I., Alippi, C., 2022. Filling the g_ap_s: Multivariate time series imputation by graph neural networks. In: International Conference on Learning Representations. URL https://openreview.net/forum?id=kOu3-S3wJ7. 





Cirstea, R.G., Yang, B., Guo, C., Kieu, T., Pan, S., 2022. Towards spatio- temporal aware traffic time series forecasting. In: 2022 IEEE 38th International Conference on Data Engineering. ICDE, pp. 2900–2913. http://dx.doi.org/10.1109/ICDE53745.2022.00262. 





Corso, G., Cavalleri, L., Beaini, D., Liò, P., Veličković, P., 2020. Principal neighbourhood aggregation for graph nets. Adv. Neural Inf. Process. Syst. 33, 13260–13271. 





Davini, D., Samineni, B., Thomas, B., Tran, A.H., Zhu, C., Ha, K., Dasika, G., White, L., 2021. Using physics-informed regularization to improve extrapolation capabilities of neural networks. In: Fourth Workshop on Machine Learning and the Physical Sciences. NeurIPS 2021. 





Do, L.N., Vu, H.L., Vo, B.Q., Liu, Z., Phung, D., 2019. An effective spatial-temporal attention based neural network for traffic flow prediction. Transp. Res. C 108, 12–28. http://dx.doi.org/10.1016/j.trc.2019.09.008. 





Fesser, L., D’Amico-Wong, L., Qiu, R., 2023. Understanding and mitigating extrapolation failures in physics-informed neural networks. arXiv preprint arXiv: 2306.09478. 





Frejo, J.R.D., Camacho, E.F., 2012. Global versus local MPC algorithms in freeway traffic control with ramp metering and variable speed limits. IEEE Trans. Intell. Transp. Syst. 13 (4), 1556–1565. http://dx.doi.org/10.1109/TITS.2012.2195493. 





Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., Dahl, G.E., 2017. Neural message passing for quantum chemistry. In: International Conference on Machine Learning. PMLR, pp. 1263–1272. 





Guo, S., Lin, Y., Feng, N., Song, C., Wan, H., 2019. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, (01), pp. 922–929. 





Hamilton, W., Ying, Z., Leskovec, J., 2017. Inductive representation learning on large graphs. Adv. Neural Inf. Process. Syst. 30. 





Jin, G., Liang, Y., Fang, Y., Shao, Z., Huang, J., Zhang, J., Zheng, Y., 2024. Spatio-temporal graph neural networks for predictive learning in urban computing: A survey. IEEE Trans. Knowl. Data Eng. 36 (10), 5388–5408. http://dx.doi.org/10.1109/TKDE.2023.3333824. 





Kim, J., Lee, K., Lee, D., Jhin, S.Y., Park, N., 2021. DPM: A novel training method for physics-informed neural networks in extrapolation. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, (9), pp. 8146–8154. http://dx.doi.org/10.1609/aaai.v35i9.16992. 





Kipf, T.N., Welling, M., 2017. Semi-supervised classification with graph convolutional networks. In: International Conference on Learning Representations. URL https://openreview.net/forum?id=SJU4ayYgl. 





Kong, J., Fan, X., Jin, X., Lin, S., Zuo, M., 2024. A variational Bayesian inference-based en-decoder framework for traffic flow prediction. IEEE Trans. Intell. Transp. Syst. 25 (3), 2966–2975. http://dx.doi.org/10.1109/TITS.2023.3276216. 





Lee, H., Ko, S., 2024. TESTAM: A time-enhanced spatio-temporal attention model with mixture of experts. In: The Twelfth International Conference on Learning Representations. URL https://openreview.net/forum?id=N0nTk5BSvO. 





Lei, M., Labbe, A., Wu, Y., Sun, L., 2022. Bayesian kernelized matrix factorization for spatiotemporal traffic data imputation and kriging. IEEE Trans. Intell. Transp. Syst. 23 (10), 18962–18974. http://dx.doi.org/10.1109/TITS.2022.3161792. 





Li, S., Cui, Y., Zhao, Y., Yang, W., Zhang, R., Zhou, X., 2023a. ST-MoE: Spatio-temporal mixture-of-experts for debiasing in traffic prediction. In: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. CIKM ’23, Association for Computing Machinery, New York, NY, USA, pp. 1208–1217. http://dx.doi.org/10.1145/3583780.3615068. 





Li, F., Feng, J., Yan, H., Jin, G., Yang, F., Sun, F., Jin, D., Li, Y., 2023b. Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution. ACM Trans. Knowl. Discov. Data 17 (1), 9:1–9:21. http://dx.doi.org/10.1145/3532611. 





Li, J., Liao, C., Hu, S., Chen, X., Lee, D.H., 2024. Physics-guided multi-source transfer learning for network-scale traffic flow prediction. IEEE Trans. Intell. Transp. Syst. 25 (11), 17533–17546. http://dx.doi.org/10.1109/TITS.2024.3405970. 





Li, F., Yan, H., Jin, G., Liu, Y., Li, Y., Jin, D., 2022. Automated spatio-temporal synchronous modeling with multiple graphs for traffic prediction. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. CIKM ’22, Association for Computing Machinery, New York, NY, USA, pp. 1084–1093. http://dx.doi.org/10.1145/3511808.3557243. 





Li, Y., Yu, R., Shahabi, C., Liu, Y., 2018. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In: International Conference on Learning Representations. ICLR’18. 





Lighthill, M.J., Whitham, G.B., 1955. On kinematic waves II. A theory of traffic flow on long crowded roads. Proc. R. Soc. Lond. A 229 (1178), 317–345. 





Liu, H., Dong, Z., Jiang, R., Deng, J., Deng, J., Chen, Q., Song, X., 2023a. Spatio-temporal adaptive embedding makes vanilla transformer SOTA for traffic forecasting. In: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York, NY, USA, pp. 4125–4129. http://dx.doi.org/10.1145/3583780.3615160. 





Liu, L., Kuo, S.M., Zhou, M., 2009. Virtual sensing techniques and their applications. In: 2009 International Conference on Networking, Sensing and Control. pp. 31–36. http://dx.doi.org/10.1109/ICNSC.2009.4919241. 





Liu, D., Wang, J., Shang, S., Han, P., 2022a. MSDR: Multi-step dependency relation networks for spatial temporal forecasting. In: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. KDD ’22, Association for Computing Machinery, New York, NY, USA, pp. 1042–1050. http://dx.doi.org/10.1145/3534678.3539397. 





Liu, M., Zeng, A., Chen, M., Xu, Z., Lai, Q., Ma, L., Xu, Q., 2022b. SCINet: Time series modeling and forecasting with sample convolution and interaction. In: Advances in Neural Information Processing Systems, Curran Associates, Inc., pp. 5816–5828. 





Liu, Z., Zheng, G., Yu, Y., 2023b. Cross-city few-shot traffic forecasting via traffic pattern bank. In: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. pp. 1451–1460. 





Makridis, M.A., Kouvelas, A., 2023. An adaptive framework for real-time freeway traffic estimation in the presence of CAVs. Transp. Res. C 149, 104066. http://dx.doi.org/10.1016/j.trc.2023.104066. 





Martin, D., Kühl, N., Satzger, G., 2021. Virtual sensors. Bus. Inf. Syst. Eng. 63 (3), 315–323. http://dx.doi.org/10.1007/s12599-021-00689-w, URL DOI: 10.1007/s12599-021-00689-w. 





Mei, H., Li, J., Liang, Z., Zheng, G., Shi, B., Wei, H., 2023. Uncertainty-aware traffic prediction under missing data. In: 2023 IEEE International Conference on Data Mining. ICDM, pp. 1223–1228. http://dx.doi.org/10.1109/ICDM58522.2023.00152. 





Nie, T., Qin, G., Wang, Y., Sun, J., 2023a. Correlating sparse sensing for large-scale traffic speed estimation: A Laplacian-enhanced low-rank tensor kriging approach. Transp. Res. C 152, 104190. http://dx.doi.org/10.1016/j.trc.2023.104190. 





Nie, T., Qin, G., Wang, Y., Sun, J., 2023b. Towards better traffic volume estimation: Jointly addressing the underdetermination and nonequilibrium problems with correlation-adaptive GNNs. Transp. Res. C 157, 104402. 





Payne, H.J., 1971. Model of freeway traffic and control. Math. Model. Public Syst. 51–61. 





Raissi, M., Perdikaris, P., Karniadakis, G., 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys. 378, 686–707. http://dx.doi.org/10.1016/j.jcp.2018.10.045. 





Richards, P.I., 1956. Shock waves on the highway. Oper. Res. 4 (1), 42–51. 





Roncoli, C., Bekiaris-Liberis, N., Papageorgiou, M., 2016. Use of speed measurements for highway traffic state estimation: Case studies on NGSIM data and highway A20, Netherlands. Transp. Res. Rec. 2559 (1), 90–100. http://dx.doi.org/10.3141/2559-11. 





Rossi, R.A., Zhou, R., Ahmed, N.K., 2018. Deep inductive graph representation learning. IEEE Trans. Knowl. Data Eng. 32 (3), 438–452. 





Roth, A., Liebig, T., 2022. Forecasting unobserved node states with spatio-temporal graph neural networks. In: 2022 IEEE International Conference on Data Mining Workshops. ICDMW, pp. 740–747. http://dx.doi.org/10.1109/ICDMW58026.2022.00101. 





Saito, H., McKenna, S.A., Zimmerman, D., Coburn, T.C., 2005. Geostatistical interpolation of object counts collected from multiple strip transects: Ordinary kriging versus finite domain kriging. Stoch. Environ. Res. Risk Assess. 19, 71–85. 





Shao, Z., Zhang, Z., Wang, F., Wei, W., Xu, Y., 2022. Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. CIKM ’22, Association for Computing Machinery, New York, NY, USA, pp. 4454–4458. http://dx.doi.org/10.1145/3511808.3557702. 





Shaygan, M., Meese, C., Li, W., Zhao, X.G., Nejad, M., 2022. Traffic prediction using artificial intelligence: Review of recent advances and emerging opportunities. Transp. Res. C 145, 103921. http://dx.doi.org/10.1016/j.trc.2022.103921. 





Shi, R., Mo, Z., Di, X., 2021. Physics-informed deep learning for traffic state estimation: A hybrid paradigm informed by second-order traffic models. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, (1), pp. 540–547. http://dx.doi.org/10.1609/aaai.v35i1.16132. 





Shi, R., Mo, Z., Huang, K., Di, X., Du, Q., 2022. A physics-informed deep learning paradigm for traffic state and fundamental diagram estimation. IEEE Trans. Intell. Transp. Syst. 23 (8), 11688–11698. http://dx.doi.org/10.1109/TITS.2021.3106259. 





Song, C., Lin, Y., Guo, S., Wan, H., 2020. Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, (01), pp. 914–921. http://dx.doi.org/10.1609/aaai.v34i01.5438. 





Sun, Z., Dai, J., Zhao, Y., Zhang, C., Ji, A., 2023. A traffic flow forecasting model using graph convolutional recurrent neural networks with incomplete data. In: 2023 IEEE 26th International Conference on Intelligent Transportation Systems. ITSC, IEEE, pp. 4669–4675. 





Sun, Y., Work, D.B., 2018. Scaling the Kalman filter for large-scale traffic estimation. IEEE Trans. Control. Netw. Syst. 5 (3), 968–980. http://dx.doi.org/10. 1109/TCNS.2017.2668898. 





Tobler, W.R., 1970. A computer movie simulating urban growth in the detroit region. Econ. Geogr. 46, 234–240. http://dx.doi.org/10.2307/143141. 





Wackernagel, H., 1995. Ordinary kriging. In: Multivariate Geostatistics: An Introduction with Applications. Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 74–81. 





Wang, Y., Papageorgiou, M., 2005. Real-time freeway traffic state estimation based on extended Kalman filter: a general approach. Transp. Res. B 39 (2), 141–167. http://dx.doi.org/10.1016/j.trb.2004.03.003. 





Wang, Z., Su, X., Ding, Z., 2021. Long-term traffic prediction based on LSTM encoder-decoder architecture. IEEE Trans. Intell. Transp. Syst. 22 (10), 6561–6571. http://dx.doi.org/10.1109/TITS.2020.2995546. 





Wang, C., Tian, R., Hu, J., Ma, Z., 2023. A trend graph attention network for traffic prediction. Inform. Sci. 623, 275–292. http://dx.doi.org/10.1016/j.ins.2022. 12.048. 





Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., ZHOU, J., 2024. TimeMixer: Decomposable multiscale mixing for time series forecasting. In: The Twelfth International Conference on Learning Representations. URL https://openreview.net/forum?id=7oLshfEIC2. 





Wang, Q., Yang, K., 2024. Privacy-preserving data fusion for traffic state estimation: A vertical federated learning approach. Transp. Res. C 104743. http://dx.doi.org/10.1016/j.trc.2024.104743. 





Wang, Y., Zhao, M., Yu, X., Hu, Y., Zheng, P., Hua, W., Zhang, L., Hu, S., Guo, J., 2022. Real-time joint traffic state and model parameter estimation on freeways with fixed sensors and connected vehicles: State-of-the-art overview, methods, and case studies. Transp. Res. C 134, 103444. http://dx.doi.org/10.1016/j. trc.2021.103444. 





Wei, T., Lin, Y., Guo, S., Lin, Y., Zhao, Y., Jin, X., Wu, Z., Wan, H., 2024. Inductive and adaptive graph convolution networks equipped with constraint task for spatial–temporal traffic data kriging. Knowl.-Based Syst. 284, 111325. http://dx.doi.org/10.1016/j.knosys.2023.111325, URL https://www.sciencedirect. com/science/article/pii/S0950705123010730. 





Wen, H., Lin, Y., Xia, Y., Wan, H., Wen, Q., Zimmermann, R., Liang, Y., 2023. DiffSTG: Probabilistic spatio-temporal graph forecasting with denoising diffusion models. In: Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems. SIGSPATIAL ’23, Association for Computing Machinery, New York, NY, USA, http://dx.doi.org/10.1145/3589132.3625614. 





Weng, W., Fan, J., Wu, H., Hu, Y., Tian, H., Zhu, F., Wu, J., 2023. A decomposition dynamic graph convolutional recurrent network for traffic forecasting. Pattern Recognit. 142, 109670. http://dx.doi.org/10.1016/j.patcog.2023.109670. 





Whitham, G.B., 1974. Linear and Nonlinear Waves. Wiley, Hoboken, NJ, USA. 





Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., Long, M., 2023. TimesNet: Temporal 2D-variation modeling for general time series analysis. In: The Eleventh International Conference on Learning Representations. 





Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., Zhang, C., 2020. Connecting the dots: Multivariate time series forecasting with graph neural networks. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. KDD ’20, Association for Computing Machinery, New York, NY, USA, pp. 753–763. http://dx.doi.org/10.1145/3394486.3403118. 





Wu, Y., Zhuang, D., Labbe, A., Sun, L., 2021a. Inductive graph neural networks for spatiotemporal kriging. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, (5), pp. 4478–4485. 





Wu, Y., Zhuang, D., Lei, M., Labbe, A., Sun, L., 2021b. Spatial aggregation and temporal convolution networks for real-time kriging. arXiv preprint arXiv: 2109.12144. 





Xiong, L., Chen, X., Huang, T.K., Schneider, J., Carbonell, J.G., 2010. Temporal collaborative filtering with bayesian probabilistic tensor factorization. In: Proceedings of the 2010 SIAM International Conference on Data Mining. SIAM, pp. 211–222. 





Xu, K., Hu, W., Leskovec, J., Jegelka, S., 2019. How powerful are graph neural networks? In: International Conference on Learning Representations. URL https://openreview.net/forum?id=ryGs6iA5Km. 





Xu, Q., Long, C., Li, Z., Ruan, S., Zhao, R., Li, Z., 2023. KITS: Inductive spatio-temporal kriging with increment training strategy. arXiv preprint arXiv:2311.02565. 





Xu, D., Tang, Y., Peng, H., Guo, H., Xuan, Q., 2024. Traffic state estimation of road sections without detectors based on multi-source causal interpretation graph. IEEE Internet Things J. http://dx.doi.org/10.1109/JIOT.2024.3427429, 1–1. 





Yang, H., Du, L., Zhang, G., Ma, T., 2023. A traffic flow dependency and dynamics based deep learning aided approach for network-wide traffic speed propagation prediction. Transp. Res. B 167, 99–117. http://dx.doi.org/10.1016/j.trb.2022.11.009. 





Yang, H., Yu, W., Zhang, G., Du, L., 2024. Network-wide traffic flow dynamics prediction leveraging macroscopic traffic flow model and deep neural networks. IEEE Trans. Intell. Transp. Syst.. 





Yin, X., Wu, G., Wei, J., Shen, Y., Qi, H., Yin, B., 2022. Deep learning on traffic prediction: Methods, analysis, and future directions. IEEE Trans. Intell. Transp. Syst. 23 (6), 4927–4943. http://dx.doi.org/10.1109/TITS.2021.3054840. 





Yu, B., Yin, H., Zhu, Z., 2018. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In: Proceedings of the 27th International Joint Conference on Artificial Intelligence. IJCAI. 





Yuan, Y., Wang, Q., Yang, X.T., 2022. Traffic flow modeling with gradual physics regularized learning. IEEE Trans. Intell. Transp. Syst. 23 (9), 14649–14660. http://dx.doi.org/10.1109/TITS.2021.3131333. 





Yuan, Y., Zhang, Z., Yang, X.T., Zhe, S., 2021. Macroscopic traffic flow modeling with physics regularized Gaussian process: A new insight into machine learning applications in transportation. Transp. Res. B 146, 88–110. http://dx.doi.org/10.1016/j.trb.2021.02.007. 





Zhang, H.M., 2002. A non-equilibrium traffic model devoid of gas-like behavior. Transp. Res. B 36 (3), 275–290. 





Zhao, L., Song, Y., Zhang, C., Liu, Y., Wang, P., Lin, T., Deng, M., Li, H., 2020. T-GCN: A temporal graph convolutional network for traffic prediction. IEEE Trans. Intell. Transp. Syst. 21 (9), 3848–3858. http://dx.doi.org/10.1109/TITS.2019.2935152. 





Zheng, C., Fan, X., Wang, C., Qi, J., 2020. Gman: A graph multi-attention network for traffic prediction. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, (01), pp. 1234–1241. 





Zheng, C., Fan, X., Wang, C., Qi, J., Chen, C., Chen, L., 2023. INCREASE: Inductive graph representation learning for spatio-temporal kriging. In: Proceedings of the ACM Web Conference 2023. WWW ’23, Association for Computing Machinery, New York, NY, USA, pp. 673–683. http://dx.doi.org/10.1145/3543507.3583525. 

