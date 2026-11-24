# BOSSbase 全库实验

BOSSbase 1.01 的 10,000 张源照片按 6,000 / 1,000 / 3,000 对划分为训练、验证和测试集。普通图缩放到 256 × 256，每张分别生成 WOW 和 HILL 载密图，目标载荷均为 0.4 bpp。两种藏法共用源照片划分。每个测试条件包含 3,000 张普通图和 3,000 张载密图。[机器可读实验结果](results_summary.json)

## 同一藏法训练与测试

| 模型 | WOW 验证 AUC | WOW 测试准确率 / AUC | HILL 测试准确率 / AUC |
|---|---:|---:|---:|
| ShuffleBANet，论文表 X | — | 91.62% / 0.9813 | — |
| ShuffleBANet，独立复现 | 0.9569（第 30 轮） | 89.52% / 0.9719 | 76.62% / 0.8804（第 13 轮） |
| **CoTraceNet-Lite** | **0.9406**（第 100 次更新） | **84.08% / 0.9252** | **76.82% / 0.8728** |

原论文第 IV-E 节表 X 给出 WOW 的 BOSS→BOSS 成绩。复现的 CNN 在该条件下比论文低 2.10 个百分点。Lite 在 WOW 验证集上选定 L2 正则强度 0.01 和第 100 次全批量更新，然后沿用这组超参数，在 HILL 训练集重新学习特征坐标和分类权重。[原论文](https://doi.org/10.1109/TDSC.2025.3596710)

CNN 按每一轮的验证 AUC 选权重，WOW 选中第 30 轮，HILL 选中第 13 轮。如果 HILL 只在预设的 20、50、100、200 轮中选优，则选中第 20 轮，测试准确率为 75.13%，AUC 为 0.8697。[HILL 四档选轮对照](hill_epochs.json)

## 特征筛选和分类器训练的逐步消融

三种配置使用相同的残差特征表示和 16,385 个推理参数。每一步的正确数变化都以相同测试规模比较。

| 特征筛选与分类器 | WOW 正确数 / 6,000 | WOW 准确率 | HILL 正确数 / 6,000 | HILL 准确率 |
|---|---:|---:|---:|---:|
| Fisher 筛选，闭式分类器 | 4,705 | 78.42% | 4,441 | 74.02% |
| 保留 Fisher 坐标，改为迭代逻辑回归分类器 | 4,975 | 82.92% | 4,602 | 76.70% |
| **CoTraceNet-Lite，迭代筛选与分类器** | **5,045** | **84.08%** | **4,609** | **76.82%** |

在 WOW 上，更换分类器多判对 270 张（+4.50 个百分点），再学习筛选坐标多判对 70 张（+1.17 个百分点）。在 HILL 上，两步分别多判对 161 张（+2.68 个百分点）和 7 张（+0.12 个百分点）。完整模型相对闭式基线在 WOW 上净增 340 张，在 HILL 上净增 168 张。

## 冻结权重的跨藏法实验

在训练藏法的验证集上选择 CNN 权重。测试另一种藏法时，CNN 和 Lite 都冻结权重与零阈值，使用目标藏法的同一批测试图。下表中的不同 HILL 轮数仅影响 CNN。两行共用同一个 HILL 训练的 Lite 模型。[跨藏法测试结果](results_summary.json)

| 训练 → 测试 | CNN 权重 | ShuffleBANet 准确率 / AUC | Lite 准确率 / AUC | Lite 多判对 |
|---|---|---:|---:|---:|
| WOW → HILL | 逐轮选优，第 30 轮 | 62.67% / 0.7582 | 70.68% / 0.8170 | 481 张 |
| HILL → WOW | 逐轮选优，第 13 轮 | 75.70% / 0.8727 | 78.00% / 0.8807 | 138 张 |
| HILL → WOW | 四档选优，第 20 轮 | 73.78% / 0.8540 | 78.00% / 0.8807 | 253 张 |

## 低误报下的 WOW 检测

用 1,000 张验证普通图设定各模型的 5% 误报阈值。冻结阈值后，分别测试 3,000 张普通图和 3,000 张载密图。

| 模型 | 误报普通图 | 误报率 | 检出载密图 | 检出率 |
|---|---:|---:|---:|---:|
| ShuffleBANet | 137 | 4.57% | 2,222 | 74.07% |
| CoTraceNet-Lite | 132 | 4.40% | 2,084 | 69.47% |

相对于 Lite，CNN 多误报 5 张普通图，同时多检出 138 张载密图。这里的判定阈值来自验证集，和同域主表的固定零阈值分别评估。

## 逐图评测与版本归档

版本 `f8e47d6` 保留了当时的 WOW、HILL 评测与逐图配对复算说明，包含混淆矩阵、配对自助区间、验证曲线和数值复核记录。各文件可从该版本的目录读取。[评测归档与复算说明](https://github.com/DuanQiChao888/CoTraceNet/tree/f8e47d6a81c31a808d26fa0443e293d323054e15/experiments/bossbase)

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
