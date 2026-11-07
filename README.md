# BMI3 Group 10 — graph-based interaction prediction

本仓库目前是**可运行的 benchmark 基础框架**，不是已经完成的项目算法。
研究主线：隐藏已知 drug–target 边 → 在可见图上评分 → 检查隐藏边恢复 → 比较效率与生物学案例。
不拟合机器学习模型，因此默认不使用 training/validation 的表述；RWR 暂不纳入范围。

## 快速开始

Python 3.10 或以上，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/download_data.py
python -m bmi3_dti inspect --dataset e
python -m unittest discover -s tests -v
python -m bmi3_dti benchmark --config configs/benchmark.json --output results/baseline-01
```

不安装包时也可以使用 `PYTHONPATH=src python -m bmi3_dti ...`。
输出目录必须尚不存在；重复实验请换目录名。默认配置运行四个数据集、三种隐藏比例、五个种子和两个基线，共 120 次评估。
完整输出含分数矩阵，磁盘占用会高于汇总表；`data/raw/` 与 `results/` 不提交 Git。

## 已完成与待完成

| 状态 | 内容 |
|---|---|
| 已实现 | 原始 TXT 下载、来源/哈希记录、ID 对齐、矩阵转置、数值检查及边列表交叉核验 |
| 已实现 | 随机删边、按药物删边、按蛋白质删边；保存可见图、隐藏标签与候选集 |
| 已实现 | random 和 target-degree 两个基础对照 |
| 已实现 | AP、AUROC、每药物前十名命中/恢复统计、曲线点、耗时与 traced-memory 记录 |
| 待小组实现 | drug-only、protein-only、固定等权 two-sided 评分，接口在 `scoring.py` |
| 后续实验 | top-k、相同评分规则下的效率优化、真实生物学案例核查 |

`drug-only` 等接口会明确报 `NotImplementedError`，不会用占位分数伪装成完成结果。
开始实现前阅读 [算法约定](docs/METHOD.md) 与 [benchmark 方案](docs/BENCHMARK.md)。

## 目录

```text
configs/                 可复现实验参数
data/                    来源清单、下载哈希和数据检查结果
  raw/                   下载的原始 TXT（忽略提交）
  processed/             预留的衍生数据目录
docs/                    方法、实验协议、课程对应与小组任务
paper-work/              仓库原有 LaTeX 模板（保留）
scripts/download_data.py 数据下载工具
src/bmi3_dti/            数据接口、删边、评分接口、评估、命令行
tests/                   防泄漏、对齐、评分指标的独立小例子
results/                 每次实验输出（忽略提交）
```

## 下一步

1. 每人手算同一个小图，确认两侧评分、归一化、零分母规则。
2. 实现并核验 `drug-only`、`protein-only`、`two-sided`，再启用配置。
3. 对相同隐藏边比较三个方法；随后完成按药物/蛋白质删边实验。
4. 再评估 top-k 和效率优化，避免一次引入多个变化。

资源与文献见 [RESOURCES](docs/RESOURCES.md)，开发与课程使用说明见 [CONTRIBUTING](CONTRIBUTING.md)。
