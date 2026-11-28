# BMI3 drug–target prediction

项目只按三件事存放文件：`input` 是原始数据，`code` 是程序和说明，`output` 是实验结果。
代码使用 Python、NumPy 和 Matplotlib，没有调用现成的 DTI 预测模型。

## 怎么读代码

按下面顺序读即可，不需要在很多脚本之间来回跳转。

| 文件 | 负责什么 |
| --- | --- |
| `data.py` | 读矩阵、按 ID 对齐、检查数据，再划分 train/validation/test |
| `models.py` | 七种打分方法，输入可见图，输出药物–蛋白质分数 |
| `benchmark.py` | 主入口：验证选参、测试、汇总、全图候选预测 |
| `plot.py` | 读取汇总表，画四项指标和评分耗时图 |
| `check.py` | 一次运行全部小数据和手算测试 |

主程序里的函数按实验顺序排列：评价一个排名、验证选参、测试方法、保存结果。
不再保留单独的下载入口、预测入口、旧格式兼容或每个种子一个结果文件夹。

## 运行

从项目根目录运行，使用 Python 3.10 或更新版本：

```sh
python -m pip install numpy matplotlib
python code/check.py
python code/benchmark.py
```

本地仓库已保留隐藏的 `.venv` 环境，可以用 `.venv/bin/python` 代替 `python`。
`.git` 保存版本历史；它们是隐藏的基础文件夹，不属于三个工作文件夹。

路径统一相对于包含 `input`、`code`、`output` 的项目根目录，和启动命令时所在位置无关。
默认读取 `input`、写入 `output`，验证模式默认写入 `output/validation`。例如
`--input input --output output/nr-demo` 在根目录或 code 目录启动都指向同一位置。
也可以进入 code 目录运行 `python benchmark.py`，或从其他位置启动这个脚本。
程序根据脚本位置寻找项目根目录，没有写死个人电脑的目录。

新生成的 JSON 设置使用 `input`、`output/nr-demo` 这样的相对路径，并标注
`path_base: project_root`。显式指定项目外部的绝对路径仍受支持，但外部数据也要随迁移
重新提供；Windows 跨盘符路径无法表示为相对路径时，会保留该显式外部路径。
已有实验 JSON 只转换了路径的表示方式，原来的指标、运行提交、时间和源码哈希保留。
换电脑时复制这三个工作目录并安装 NumPy、Matplotlib 即可；本机 `.venv` 应重新创建。

默认运行四个数据集、种子 0–4、七种方法，并自动生成图和未标注候选预测。
已有结果不会被覆盖。复跑时请使用一个新的输出位置，例如：

```sh
python code/benchmark.py --dataset nr --seed 42 --output output/nr-demo --compare-backends
```

`--compare-backends` 会比较循环版和向量化版的完整 Top-k 评分，先检查结果一致再记录耗时。
其他参数可以用 `python code/benchmark.py --help` 查看。`--no-plots` 跳过绘图，
`--no-predictions` 跳过全图候选预测。只重画已有结果可以运行：

```sh
python code/plot.py --output output
```

## 输入

| 前缀 | 数据集 | 药物 | 蛋白质 | 已知关系 |
| --- | --- | ---: | ---: | ---: |
| e | 酶 | 445 | 664 | 2926 |
| ic | 离子通道 | 210 | 204 | 1476 |
| gpcr | G 蛋白偶联受体 | 223 | 95 | 635 |
| nr | 核受体 | 54 | 26 | 90 |

每个前缀对应四个文件，以 nr 为例：

* `nr_admat_dgc.txt`：已知相互作用矩阵，1 为有记录，0 为未标注。
* `nr_simmat_dc.txt`：药物之间的化学相似性。
* `nr_simmat_dg.txt`：蛋白质之间的相似性。
* `bind_orfhsa_drug_nr.txt`：蛋白质–药物边列表，用来核对矩阵，可省略。

矩阵首行是列 ID，后面每行是行 ID 和对应数字。程序按 ID 对齐，自动识别相互作用
矩阵方向，统一成“药物为行、蛋白质为列”。自定义数据用 `--input` 和 `--dataset` 指定，
保持同样命名即可。药物和蛋白质应使用不同的 ID 命名空间，避免方向歧义。

相似性取原矩阵与转置的平均，对角线置零；原始文件不会修改。检查结果和哈希写入
`experiment.json`。原论文参考预测不参与实验，已从工作目录移到旧版本备份中。

## 方法

可见相互作用矩阵记作 A，药物、蛋白质相似性分别记作 SD、SP，二者对角线均为零。

| 方法名 | 规则 |
| --- | --- |
| random | 固定随机种子的均匀随机分数 |
| target-degree | 用可见图中蛋白质的度打分 |
| dti-only | 统计药物→蛋白质→药物→蛋白质三步路径，去掉立即回头的路径 |
| drug-only | 相似药物的作用对象加权投票：SD A，再除以对应行的权重和 |
| protein-only | 将同样的投票规则用于蛋白质侧 |
| two-sided | alpha × 药物侧分数 + (1-alpha) × 蛋白质侧分数 |
| two-sided-topk | 活跃邻居筛选、Top-k、幂加权、可用性融合 |

DTI-only 的表达式是 `A A.T A - A*(drug_degree + protein_degree - 1)`，修正项中的
乘法逐元素进行。未见边的 A 值是零，所以修正不改变候选关系的三步路径数。

增强版排除自己和可见度为零的邻居，再按相似性选前 k 个；同分按 ID 排序。
权重取相似性的 power 次方并归一化。若某一侧没有可用证据，就只在有正权重的
可用侧之间归一化；两侧均不可用时给零。alpha 为 0 或 1 时，零权重侧不作为兜底。
单侧基线保留所有非自身邻居的权重作为分母，包括可见度为零的邻居。

默认 alpha 是 0.5/0.25/0.75，k 是 5/10/20/50，power 是 1/2。固定融合有 3 个试验，
增强融合有 24 个试验。相同验证 AP 时，使用参数列表中先出现的组合。
RWR 属于此前的可选扩展，本版本没有实现。

## 划分与评价

只划分已知正例，默认约 80%/10%/10%，验证和测试数量向下取整。
验证时算法只能看到训练边；选好参数后恢复验证边，测试边仍然隐藏。
两阶段都用原始零项作为未标注背景，其他阶段的隐藏正例不混入背景。

AP/AUROC 对整个候选集合计算，同分作为一组处理。前 N 名按分数降序、蛋白质 ID
升序排列。逐药物 precision 的分母是 min(N, 候选数)，recall 的分母是该药物的隐藏正例数。
macro 指标只平均至少有一条隐藏正例的药物；没有定义的指标留空，不填成零。
汇总表报告跨种子的均值和样本标准差，单次运行的标准差留空。

评分耗时是重复运行的中位数，包含该方法的邻居准备和两侧打分，不含加载、调参和评价。
循环/向量化对比测量完整 Top-k 融合，而非分别计时每一侧。Top-k 仍扫描并排序密集相似性
矩阵，不保证更快；本次数据上它比密集矩阵基线慢。空间为 O(D²+P²+DP)，不适合超大网络。

这是随机藏边的已知关系恢复实验，不能解释成已确认的阴性分类或独立药物/蛋白质泛化。
增强版同时改变多个规则，不能将性能差异单独归因于 k。

## 输出

所有数据集、种子的结果合在同一层表格中，每条实验记录带 dataset 和 seed。

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 每个数据集、种子、方法的最终测试指标，默认 140 行 |
| `summary.csv` | 跨种子均值、样本标准差和有效次数，默认 28 行 |
| `parameters.csv` | 验证阶段全部参数试验，selected 标出选中的组合 |
| `splits.csv` | 每条已知关系在哪个划分中，可按 dataset/seed 查询 |
| `per_drug.csv` | 每种方法对每个药物的命中数、precision、recall 和冷启动标记 |
| `timing.csv` | 可选的完整 Top-k 循环/向量化计时，每次实验两行 |
| `predictions.csv` | 每个药物前 N 个原始未标注候选 |
| `experiment.json` | 参数、数据/源码哈希、版本、Git 状态、各次实验的计数及选参 |
| `benchmark.png` / `.pdf` | 四项评价指标图 |
| `runtime.png` / `.pdf` | 评分耗时图 |
| `checks.txt` | 原有完整 benchmark 的测试日志；新增验证的日志单独存放 |

候选预测预先固定使用“第一个请求种子”的验证参数，再用全部已知关系打分。
`predictions.csv` 中的 seed 表示参数来源，不表示全图再次藏边。score 是排序分数，
不是概率；status 为 unlabeled。全图已经恢复测试边，因此不能用它再算同一批藏边的成绩。

## 可选的补充验证

原有七项比较、评分公式、默认命令和输出文件保持不变。需要补充实验时运行：

```sh
python code/benchmark.py --study
```

结果单独存入 `output/validation`，不会覆盖已有 benchmark。复跑需要新的 `--output`。
这项检查不生成新的全图候选预测，只检验已有方法：

1. **对齐图**：四个数据集、seed 42，只展示与同事共同的五个方法。
   评分耗时换成毫秒和对数坐标，数值仍受硬件、线程和计时范围影响。
2. **稳定性**：核受体数据集使用预先固定的种子 0–19，比较两个单侧、固定融合和增强融合。
   `--stability-seeds` 可改变重复次数。各次仍是同一份数据的重叠划分，并没有增加独立样本。
3. **组件比较**：四个数据集、默认种子 0–4，按“固定融合 → 活跃邻居 → Top-k → 幂加权 →
   可用性融合”逐步增加组件。每一步只用验证 AP 选参数，再在相同测试候选上比较。

这些是重新验证选参后的配置比较，差异同时包含参数选择影响，不是固定参数下的纯因果效应。
幂加权消融会在 `--powers` 中补入 1 作为对照，并记录实际网格；默认网格仍是 1/2。
自定义 powers 若不包含 1，这项消融的网格会与原普通 benchmark 不同。
seed 42 只用于对齐展示，不用它挑选最好种子或修改算法。

新增文件均在 validation 目录内：

| 文件 | 内容 |
| --- | --- |
| `seed42.csv`、`seed42.png/.pdf` | 五个共同方法的单种子结果和五面板图 |
| `stability.csv`、`stability_summary.csv` | 核受体逐种子结果及均值、样本标准差 |
| `parameter_frequency.csv` | 核受体重复划分中选中各组参数的次数 |
| `stability.png/.pdf` | 逐种子 AP 和增强版相对固定融合的配对差值 |
| `ablation.csv`、`ablation_parameters.csv` | 五阶段测试结果及其验证选参记录 |
| `paired.csv` | 每一步在相同种子上的四指标差值，含均值、范围和胜/平/负次数 |
| `ablation.png/.pdf` | 逐步组件比较的均值和样本标准差 |
| `study.json` | 新验证的设置、数据/源码哈希、版本和运行记录 |
| `checks.txt` | 新增验证交付时的 26 项测试日志 |

差值均按“后一步减前一步”计算，绝对值不超过 1e-12 计为平。它们是描述性比较，
不提供基于独立样本假设的显著性结论。可用性融合在这些数据上可能完全没有增益；
`unsupported_drugs/proteins` 记录没有可用邻居的数量，零变化也应保留和解释。

## 引用、课程关联与来源

数据和原有原型：

1. Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M (2008).
   Prediction of drug–target interaction networks from the integration of chemical and genomic spaces.
   Bioinformatics 24(13), i232–i240. https://doi.org/10.1093/bioinformatics/btn162
2. 原始补充数据：http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/
3. 继承的项目：https://github.com/shwenyu/BMI3-graph-based-interaction-prediction/tree/Hanyu-Shou
   最初原型提交 7c0fed98786352e498161f1ec429dd3251936958；重构前完整实现提交 26427d1。

实现使用课内的基本构件：Week 1 的数组/NumPy，Week 2 的数据结构和排序相关操作，
Week 2.2 的穷举思路，Week 3 的随机化，Week 4 的清晰伪代码与循环，Week 7 的图表示与路径。
验证参数网格采用直接枚举；核心投票和指标自行写出，没有导入现成预测算法。
这些基础构件的课程关联不等于整个融合设计是课内原题或原创算法。

依赖文档：Python https://docs.python.org/3/library/ ，NumPy https://numpy.org/doc/stable/ ，
Matplotlib https://matplotlib.org/stable/ 。新重构没有复制外部网站的算法代码；新增代码没有
需要单独归属的外部代码片段。上面的论文用于数据来源说明，本实验不是复现其原模型。

原型、完整实现及本次重构均有 AI 辅助，保留这一真实来源记录。“humanize”在这里指
减少重复、清楚命名、直接流程和适量注释，不表示隐去来源或声称为个人独立编写。
数据和原仓库的既有使用条款继续适用，本次重构没有为它们另行授予许可证。
