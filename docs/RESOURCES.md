# Verified resources

## 原始数据

来源：http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

2026-11-07 下载核验。网页声明 gold-standard 来自 2007-11 的 KEGG BRITE、BRENDA、
SuperTarget、DrugBank，且该数据快照不会更新。结果应解释为历史记录恢复。
网页未见明确的独立数据再分发许可证；仓库保存下载程序和哈希，不提交原始 TXT。

| prefix | 类别 | 药物 | 蛋白质 | 已知边 |
|---|---|---:|---:|---:|
| e | Enzyme | 445 | 664 | 2926 |
| ic | Ion channel | 210 | 204 | 1476 |
| gpcr | GPCR | 223 | 95 | 635 |
| nr | Nuclear receptor | 54 | 26 | 90 |

每类下载四个文件：

- `{prefix}_admat_dgc.txt`：已知互作矩阵，原始方向为 protein × drug。
- `{prefix}_simmat_dc.txt`：SIMCOMP 化学结构相似矩阵。
- `{prefix}_simmat_dg.txt`：normalized Smith–Waterman 蛋白质序列相似矩阵。
- `bind_orfhsa_drug_{prefix}.txt`：protein、drug 两列互作表，用于交叉核验。

`*_predpairmat_definition_expdist.txt` 是论文预测结果，不是 gold-standard 标签，不作为实验输入下载。
原始文件与本地哈希见 `data/source_manifest.json`；网页快照见 `data/source_index.html`。
核验结果见 `data/dataset_audit.json`。边列表的 `hsa:2099` 在矩阵中为 `hsa2099`，仅交叉核验时规范化。
相似矩阵有非对称及非单位对角的情况；原始值不改，加载器记录异常后生成对称、零对角副本。
未强制修改为 PSD，因为邻居权重求和不要求核矩阵半正定。

## 论文阅读顺序

1. Yamanishi et al. (2008). Prediction of drug–target interaction networks from the integration of chemical and genomic spaces.
   Bioinformatics 24(13), i232–i240. https://doi.org/10.1093/bioinformatics/btn162
   先读数据与 weighted profile，暂不实现 Gaussian/kernel mapping。
2. Cheng et al. (2012). Prediction of drug-target interactions and drug repositioning via network-based inference.
   PLoS Computational Biology 8(5), e1002503. https://doi.org/10.1371/journal.pcbi.1002503
   区分 DBSI/TBSI 的加权邻居与 NBI 的资源扩散，不把三者写成同一算法。
3. Bleakley and Yamanishi (2009). Supervised prediction of drug–target interactions using bipartite local models.
   Bioinformatics 25(18), 2397–2403. https://doi.org/10.1093/bioinformatics/btp433
   借鉴同一配对的双侧推理；local 是节点专属分类器，不等于 top-k；不实现 SVM。

## 课程与反馈

课程要求以小组现有的 Project 10 描述、ICA Guidance 和老师批注 PDF 为准；它们不作为公开数据集附带上传。
老师强调：隐藏已知边验证恢复；明确是否训练/验证；解释零分母、阈值及每项比较；实测效率。

W07 Graph Algorithms I：第 10 页子图、49–58 页矩阵/邻接表/边列表与 degree。
W07 Graph Algorithms II：第 5–9 页路径问题、12–16 页 DFS，连接到可达性诊断，不强加最短路预测。
W08 Dynamic Programming I：第 8 页复用已计算值，24–30 页 DAG 递推；邻居评分不因此成为 DP。
W08 Dynamic Programming II：第 19–24 页局部比对，帮助解释蛋白质相似性，仍不能由序列相似直接证明互作。
