# 双视角奖励分类

双视角模型从同一张图像提取三路统计。全图三元共现、纹理三区四元共现和代价五区四元共现共同产生 Lite 基础分数。纹理和代价两路另外按滤波算子、量化步长与计数方向匹配 16 对残差族。全局注意力对两路的 32 项证据逐图加权，同族门控依据证据差、方向一致性与区域词频的 Hellinger 距离作有界修正。[特征和配对](../cotracenet/binocular/prepare.py) · [两级注意力](../cotracenet/binocular/attention.py)

候选分类头读取全图统计，给双视角分数一个幅度不超过 2 的修正。两个分数判为同一类别时保留原判；类别冲突时由一层动作策略决定是否采纳候选。训练时相对原判纠错记 +1、引入错误记 −1.5，动作阈值取自源域验证集。推理输入是一张普通灰度图。[奖励分类器](../cotracenet/binocular/readout.py) · [单图推理实现](../cotracenet/binocular/predict.py)

## 注意力、奖励分类与局部复查

在同组权重的逐级消融中，WOW 同域起点为 83.08%，加入全局注意力、同族配对注意力后分别为 83.17%、83.15%，再加奖励分类达到 84.08%，最后一步净多判对 56 张。HILL 同域依次为 76.82%、76.92%、77.10%、77.23%，奖励分类净多判对 8 张。

对低置信度图像强制进行分歧选块复查，WOW 同域从 84.08% 降到 83.13%，净少判对 57 张。HILL 同域从 77.23% 降到 77.07%，净少判对 10 张。源域验证选出的策略在两域都保留全图模型判决。[四个训练与测试方向的汇总结果](detection-results.csv)

## 训练与预测

从仓库根目录运行。复用基础 Lite 的 Python 依赖、数据准备和训练入口。下例使用数据工具创建的 WOW、HILL 0.4 bpp 数据清单，以及 `cotracenet.iterative fit` 训练出的两个完整目录。[生成数据及训练基础 Lite](../README.md#复现基础-lite)

```bash
python -m pip install -r requirements.txt
python -m cotracenet.binocular fit --domain WOW --manifest work/boss/WOW_04/manifest.csv --lite-fit work/lite-wow --out work/binocular
python -m cotracenet.binocular fit --domain HILL --manifest work/boss/HILL_04/manifest.csv --lite-fit work/lite-hill --out work/binocular
python -m cotracenet.binocular predict --domain WOW --out work/binocular sample.pgm
```

`--domain` 选择训练权重。训练阶段从源域训练和验证清单提取特征，在训练源照片之间做三折交叉拟合；每个外折重新拟合特征筛选、基础分类器及注意力模块，供奖励分类器学习。最终输出分为 `work/binocular/attention/<域>` 和 `work/binocular/policy/<域>`。单图预测输出双视角分数、备选分数、策略动作及最终类别。完整模型有 **16,828** 个推理权重与偏置。[训练流程](../cotracenet/binocular/pipeline.py)

## 同域和跨藏法测试

源域拟合完成后冻结权重。以下两条命令都使用 WOW 权重，第二条更换测试清单。将源域换为 HILL 可测另两个方向。测试输出依照 `--target` 分开保存，并返回各模型的准确率与 AUC。主表分别列出基础 Lite 与双视角奖励模型。做逐级消融时，使用同一次运行产生的基础分数与注意力、策略分数配对比较。

```bash
python -m cotracenet.binocular score --domain WOW --target WOW --manifest work/boss/WOW_04/manifest.csv --out work/binocular
python -m cotracenet.binocular score --domain WOW --target HILL --manifest work/boss/HILL_04/manifest.csv --out work/binocular
```

用同一张图的两种预测做配对比较，可输出改对和改错的张数，以及准确率差的置信区间。下面只读刚生成的 WOW 同域分数。可以再传入其他方向的 `--case`，每个方向保留一组冻结的源域模型。

```bash
python -m cotracenet.binocular.evaluation --case WOW WOW work/binocular/policy/WOW/WOW-to-WOW.npz --models lite brain policy --compare policy brain --out work/reward-vs-attention.json
```

## 可选的局部复查消融

局部复查实验使用同一冻结分类器，比较保留原判、取固定位置、取纹理与高频响应排名分歧最大的位置。两种取块方法均提取一块 64 × 64 区域的 156 维残差描述。奖励门控与按低置信度设阈值的对照可分别运行。默认双视角模型的单图接口只执行上一节的全图奖励分类。[选块与局部描述](../cotracenet/local_review/patch_features.py) · [复查头与门控](../cotracenet/local_review/active_review.py)

完整消融命令使用两种藏法的 BOSSbase 全库清单，每域各有 6,000 对训练、1,000 对验证、3,000 对测试图。先按上文拟合两个源域，再运行下面的命令。它先拟合两个域的局部头并锁定阈值，再评估四种训练/测试方向。输出目录内的 `performance.csv` 含七种策略的准确率、AUC、查询数、改对和改错数；`performance.json` 另含按源图配对重采样的区间。

```bash
python -m cotracenet.local_review --models work/binocular --wow-manifest work/boss/WOW_04/manifest.csv --hill-manifest work/boss/HILL_04/manifest.csv --out work/review
python -m cotracenet.local_review.deploy --binocular work/binocular/attention --previous work/binocular/policy --root work/review predict --train WOW --method gate_selected sample.pgm
```

可用 `gate_fixed`、`uncertainty_selected`、`uncertainty_fixed`、`always_selected` 或 `always_fixed` 比较相应实验臂。算子审计从同一套自训权重和一张灰度样本生成快速、固定复查和主动复查三条路径的参数及计算量记录。

```bash
python -m cotracenet.local_review.cost --attention work/binocular/attention --policy work/binocular/policy --current work/review --image sample.pgm --train WOW --out work/review/cost-WOW.json
```
