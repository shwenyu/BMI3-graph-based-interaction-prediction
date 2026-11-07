# Work plan after proposal feedback

## Milestone 1 — 一次可手算的闭环

- 组员共同用 3–4 个药物和蛋白质画出二部图，选一条已知边隐藏。
- 列出候选集，手算两侧得分和平均分。
- 写清楚 visible/hidden、原始未记录对、零分母和并列分数。
- 验收：每个人都能解释为什么这不是模型训练，以及算法如何被公平评价。

## Milestone 2 — 核心方法（建议 11 月 8–14 日）

- 数据/benchmark：审查加载器、split 保存与逐药物输出。
- 算法：组员实现 drug-only、protein-only、two-sided 并核对手算结果。
- 评估：独立核对 AP、AUROC、前十名恢复与 provisional-negative 边界。
- 文档：保留每个设计的原因、失败例子与个人贡献记录。

## Milestone 3 — 实验（建议 11 月 15–24 日）

- 四类数据的随机隐藏实验，所有方法共享掩码；再做 drug/protein-wise hiding。
- 比较三种评分与 random/degree 对照，报告 seeds 间波动。
- 核心稳定后才加入固定 top-k；进行相同公式的效率优化和一致性核验。
- 分开记录预处理、评分、排序，不能将不同实现的时间直接解释为算法复杂度。

## Milestone 4 — 解释与提交准备（建议 11 月 25 日–12 月 6 日）

- 选择可追溯成功与失败互作案例，核查身份与文献，展示评分来源。
- 在干净环境复现实验，完善软件说明与个人反思。
- 课程已给出的 code/report 截止为 12 月 10 日中午，slides 为 12 月 17 日中午；提交前再核对 Blackboard。

默认配置中的两个 baseline 只用于验收基础设施，不是最终方法成果。成绩或预测改善不作保证。
