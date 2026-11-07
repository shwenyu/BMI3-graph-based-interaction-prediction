# Student scoring implementation contract

## 最小主算法

用药物 i 的相似药物是否作用于 j 来定义 drug-based score：

`s_D(i,j) = sum_u S_D[i,u] * A_vis[u,j] / sum_u S_D[i,u]`。

用蛋白质 j 的相似蛋白质是否被 i 作用来定义 protein-based score：

`s_P(i,j) = sum_v S_P[j,v] * A_vis[i,v] / sum_v S_P[j,v]`。

相似矩阵已取 `(S+S.T)/2` 且对角线清零，所以求和不会使用自身。
分母零时该侧评分定为 0；无信息情况在分析中另行记录。主方法固定 `s=(s_D+s_P)/2`，
不采用自适应权重，不拟合参数。所有值是 ranking score，不是互作概率。

## 接口与责任

在 `src/bmi3_dti/scoring.py` 完成三种待实现方法，输出与 visible 相同的 n×m 浮点矩阵。
不能读取磁盘上的原始 A、split.npz、hidden 或评价结果决定评分。
实现后移除 `cli.py` 对这三种方法的未实现保护，并在配置中加入方法名。
增加手算例子：两侧贡献分别非零、某侧零分母、已知边隐藏前后、可见零度药物。
用测试确认预测只依赖可见图，且输入不被修改。

## 课堂图逻辑

drug-based score 累加 i→相似药物 u→蛋白质 j 的局部支持；protein-based score
累加 i→蛋白质 v→相似蛋白质 j 的局部支持。
DTI 主图保持二部图；相似性作为评分的侧信息，不要求拼装异质转移矩阵。
邻居权重求和不等于 BFS visited 规则，也不意味着沿路径发生药理作用。

## 后续 top-k

全邻居版本可靠后，才将每行权重限制到最多 k 个正相似邻居，排除自己；并列按 ID。
矩阵大小不变，只将非选中边权设零。评分不要求 union 对称化；两方向各自选择邻居即可。
若采用其它对称化方式，须作为另一个方法明确说明，不与不同规则的基线混为一谈。
top-k 会改变预测公式和可用信息；相同公式的数据结构优化才是单纯效率比较。

## 不在本阶段实现

RWR、Gaussian-distance kernel、SVM/embeddings、幂次权重搜索、自适应融合。
