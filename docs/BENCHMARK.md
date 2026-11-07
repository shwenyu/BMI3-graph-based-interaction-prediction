# Benchmark protocol — version 1

## 图与标签的所有权

原图 `A` 是已记录互作，不是真正完整的生物网络。设隐藏集合 `H ⊆ E`，
可见图 `G_vis=(D ∪ P,E\H)` 保留所有节点，是原图的 spanning subgraph。
统一矩阵方向为 drug × protein。删边改变矩阵中的 1，不删除行或列。

预测函数只接收 `visible`、外部相似矩阵和随机种子，不能接收原始 A、hidden 或正确标签。
评估模块持有这些答案。这是 hiding/recovery benchmark，没有参数拟合意义上的训练。

## 生成一次实验

1. 从原始 A 的 1 中选隐藏边；random 模式隐藏数为 `max(1,floor(fraction*|E|))`。
2. 将隐藏位置在 A 的副本上改成 0，得到 visible。
3. random 模式在所有 drug–protein 对上评估。
4. 候选集 = 隐藏边 + 原始未记录对；已可见的边必须排除。
5. 相似性是外部结构/序列信息，保持可用；本框架不生成 interaction-profile similarity。
6. 评分、排序，评估 hidden 是否被找回。所有方法共享同一 split。

`drug` 模式随机选择一部分药物，将其全部互作隐藏，只在选中药物的候选对中评估；
`protein` 模式同理。此时 fraction 指节点比例，不是边比例；输出另记实际隐藏边比例。
不强制保留每节点一条边，也不静默移除 cold-start 节点；记录可见图的零度节点数。
若所选实体没有任何已知互作，明确失败，不悄悄改种子重抽。

默认只运行 random；按实体实验可将配置 `modes` 改为 `["random","drug","protein"]`。
四类数据全部通过后再比较跨数据集结果，不将不同难度的数据混成一个准确率。

## 指标与标签边界

- pooled AP 和 AUROC：同一个数据集/同一次隐藏实验中汇总所有候选对。
- AP 采用相同分数整组加入的 precision–recall 加权定义，不等同于梯形 PR 面积。
- 曲线阈值遍历不同预测分数；它不同于隐藏比例和邻居数量 k。
- 前十名结果按每药物计算，候选少于十个时全部使用。
- top-N 分数相同时按 protein ID 排序，不能用 hidden 标签解并列。
- 汇总的 macro precision 和 recall 均只含至少一个隐藏正例的药物；逐药物文件保留全部行。
- 没有隐藏正例的药物 recall 为 null，不记为 0；缺少正/负类时 AUROC 为 null。
- 对未记录候选仅使用“provisional/background negative”这一操作性标签。
  指标描述已记录边的恢复，不能证明新候选的真实正确率或生物学无互作。

所有背景候选均参与，默认不采样负例，避免改变 prevalence 使结果难以比较。
无信息节点保留，输出其个数；如进一步按 degree 分层，应只使用 visible degree。

## 参数与最终测试

主方法固定 q=1、双侧各占 1/2；先全邻居，因此没有调参步骤，也无须额外 validation 集。
改变 k 的分析可以逐一报告预设设置，不把在最终隐藏边上挑出的最好结果当作独立测试表现。
若决定选择最佳 k/权重，必须另外保存开发用掩码，并使最终隐藏标签在选择过程中不可见。
开发、最终测试都不能从原图构造会泄露隐藏边的 degree 或 profile 特征。

## 输出与效率

每次运行保存配置、Python/NumPy 版本、Git commit 和 dirty 状态、原始文件哈希、ID 顺序。
每个 split 保存 visible/hidden/candidates/selected_entities，每方法保存 scores、曲线点、逐药物表、指标 JSON。
`metrics.csv` 汇总；按 dataset/mode/fraction/method 对 seeds 汇总均值和标准差，保留原始结果。

计时分开记录读取、生成 split、评分、评估；评分时间包括为接口创建输入副本的成本。
`score_traced_peak_bytes` 是 tracemalloc 跟踪到的评分期分配，**不是整个进程峰值 RSS**。
当前单次计时仅用于打通输出；正式效率实验需同一设备多次运行、报告中位数，记录输入规模与密度。
先比较相同公式的简单/优化实现并确认分数一致，再单独比较 top-k 引入的信息与成本变化。
