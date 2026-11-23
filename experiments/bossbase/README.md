# BOSSbase 全库实验

本实验比较三种使用相同残差特征、保留相同参数量的 Lite 模型。固定决策阈值时，完整迭代版在 WOW 上比闭式版多判对 280 张，在 HILL 上多判对 168 张。两种算法分别训练和测试，各有 6,000 张测试图。

| 检测器 | WOW 0.4 正确数 | 准确率 | ROC AUC | HILL 0.4 正确数 | 准确率 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Closed · Fisher 筛选与闭式拟合 | 4,705 | 78.42% | 0.8865 | 4,441 | 74.02% | 0.8399 |
| Hybrid · Fisher 坐标＋迭代分类器 | 4,975 | 82.92% | 0.9244 | 4,602 | 76.70% | 0.8723 |
| **CoTraceNet-Lite · 迭代筛选与分类器** | **4,985** | **83.08%** | **0.9252** | **4,609** | **76.82%** | **0.8728** |
| ShuffleBANet 独立复现 · 逐轮按验证 AUC 选优 | 5,131 | 85.52% | 0.9519 | 4,597 | 76.62% | 0.8804 |
| ShuffleBANet 独立复现 · 四档轮数内按验证 AUC 选优 | — | — | — | 4,508 | 75.13% | 0.8697 |

原论文第 IV-E 节表 X 的 BOSS_256→BOSS_256、WOW 0.4 bpp 为 **91.62% / AUC 0.9813**。本项目沿用论文给出的训练、验证、测试比例，自行划分源图、模拟载密图并实现网络。上表中的本地模型逐图配对，论文成绩用于同规模参照。[原论文](https://doi.org/10.1109/TDSC.2025.3596710)

## 数据与模型选择

- **划分。** BOSSbase 1.01 的 10,000 个源文件经双三次插值缩放至 256 × 256，每张图分别生成一张 WOW 和一张 HILL 载密图，目标载荷均为 0.4 bpp。按来源分为 6,000 对训练、1,000 对验证、3,000 对测试。四组内容重复的原图成组留在训练集。WOW 和 HILL 共用原图及划分，模型权重分别拟合。
- **迭代训练。** 仅由训练图估计各特征的均值与总体标准差。先以固定 L2 系数 0.1 在 34,398 维上训练逻辑回归，最多 200 次全批量更新。按标准化权重绝对值取前 16,384 个坐标，同值时优先较小的特征编号。随后在选出的坐标上从零拟合分类器，优化平均二元交叉熵加 L2 惩罚，偏置不参与惩罚。
- **选择与迁移。** WOW 在 L2 系数 1、0.1、0.01 及第 5、15、40、100、200 次可达更新和各自终止点上比较验证 AUC。完整迭代版和 Hybrid 均选中 0.01、第 100 次更新。对 HILL 固定这套超参数，用 HILL 训练图重新拟合。闭式版和 Hybrid 在同一种藏法下共用 Fisher 特征坐标。[WOW 验证曲线数据](wow_validation.csv)
- **CNN 对照。** 两条件均从种子 73 初始化，三张 GPU 同步批归一化，全局每步 15 对，每轮遍历 6,000 对。训练用 BF16，验证用 FP32，优化器为 Adamax。逐轮选优分别取 WOW 第 30 轮（共跑 42 轮）、HILL 第 13 轮（共跑 34 轮）。另一组 HILL 对照完整跑到 200 轮，只在 20、50、100、200 轮中选，验证 AUC 在第 20 轮最高。[HILL 四档轮数的验证和测试结果](hill_epochs.json)
- **测试范围。** 每种藏法的 3,000 个测试来源中，512 个来源曾用于早期实验，另有预先标记的 2,488 个新来源。只看这 2,488 对，完整迭代版 WOW 判对 4,136 / 4,976（83.12%），HILL 判对 3,831 / 4,976（76.99%）。当前结果来自一个图库和一个训练种子。

## 收益来自哪一步

在相同的源照片上成对比较两种推理结果，连续重采样时保留同一源图的普通图、载密图与两个模型的结果。下表的 95% 区间来自 10,000 次源图重采样（PCG64 种子 7319）。

| 准确率差与配对 95% 区间（百分点） | WOW | HILL |
|---|---:|---:|
| Hybrid − Closed | +4.50 [3.68, 5.33] | +2.68 [1.82, 3.55] |
| CoTraceNet-Lite − Closed | +4.67 [3.85, 5.52] | +2.80 [1.92, 3.65] |
| CoTraceNet-Lite − Hybrid | +0.17 [−0.28, 0.63] | +0.12 [−0.40, 0.63] |

主要收益出现在换成正则化逻辑回归分类器时。这一步同时改变了判别目标、正则化和拟合方式。进一步学习特征坐标仅多判对 10 张 WOW、7 张 HILL。按普通图和载密图拆开，完整迭代版相对 Closed 在 WOW 上少误报 220 张、多检出 60 张；在 HILL 上少误报 290 张、多漏检 122 张。

## 验证集约束误报率

对每个模型，把 1,000 张验证普通图的分数从小到大排序，取第 950 个为阈值。严格大于阈值才判载密。阈值冻结后，两个模型在测试集的实际误报率可以不同。

| 模型 | WOW 测试误报率 | WOW 测试检出率 | HILL 测试误报率 | HILL 测试检出率 |
|---|---:|---:|---:|---:|
| ShuffleBANet · WOW 第 30 轮 / HILL 第 20 轮 | 4.57% | 74.07% | 5.77% | 53.50% |
| CoTraceNet-Lite | 4.40% | 61.57% | 4.53% | 48.30% |

WOW 测试中，CNN 误报 137 张、Lite 误报 132 张，CNN 多检出 375 张载密图。默认零阈值的混淆矩阵、ROC AUC、校准阈值和配对差值均可在 JSON 中查到。JSON 中的准确率和检出率为 0–1，差值单位为百分点。[WOW 汇总](wow.json) · [HILL 汇总](hill.json)

## 使用自己的预测分数复算

在仓库根目录运行。每个 CSV 至少有 `source,label` 和一个分数列，每个来源含一张普通图（0）和一张载密图（1）。各模型须使用相同来源，验证和测试来源彼此独立。主 README 的命令生成 WOW 下的 Lite 测试分数和 Hybrid 权重。以下先用这些权重生成其余三份分数，再做配对比较。

```sh
python -m cotracenet.iterative score --fit work/lite-wow --manifest work/boss/WOW_04/manifest.csv --split val --out outputs/wow/lite-val.csv
python -m cotracenet.iterative score --fit work/hybrid-wow --manifest work/boss/WOW_04/manifest.csv --split val --out outputs/wow/hybrid-val.csv
python -m cotracenet.iterative score --fit work/hybrid-wow --manifest work/boss/WOW_04/manifest.csv --split test --out outputs/wow/hybrid-test.csv

python experiments/evaluate_boss.py \
  --model CoTraceNet-Lite outputs/wow/lite-val.csv outputs/wow/lite-test.csv cotracenet_lite \
  --model Hybrid outputs/wow/hybrid-val.csv outputs/wow/hybrid-test.csv cotracenet_lite \
  --compare CoTraceNet-Lite Hybrid \
  --out outputs/wow-comparison.json
```

## 单核推理开销

在 AMD EPYC 7H12 的同一 CPU 核上，从读取 256 × 256 灰度图计到输出单图分数。两个模型权重常驻内存，PyTorch 和 BLAS 各使用一个线程，图像文件位于操作系统缓存中。按源编号排序取 12 对 HILL 测试图，每图测三次，交替两个模型的顺序，每个模型共 72 次。复现 ShuffleBANet（第 20 轮）的中位耗时为 **291.78 ms**，完整迭代 Lite 为 **64.44 ms**。Lite 使用 16,384 个浮点系数和 1 个偏置，另存 16,384 个整型坐标。CNN 可训练参数为 651,876。[原始计时汇总、四分位数与权重文件大小](resources.json)

```sh
python experiments/benchmark.py \
  --manifest work/boss/HILL_04/manifest.csv --backbone work/cnn-hill-fixed/epoch-020.pt \
  --lite-fit work/lite-hill --out outputs/hill-timing.json
```
