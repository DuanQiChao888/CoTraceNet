# CoTraceNet

CoTraceNet（共现残差门控网络）是一个灰度图像隐写检测器。输入一张 256 × 256 的图像，模型估计其中是否嵌入了消息。它在五类纹理区域分别统计残差共现，用两个 Fisher 判别器给出初始分数，再根据图像内容对接近分类边界的判断做小幅校正。

## 模型结构

![CoTraceNet 架构。从一张灰度图提取残差共现和 ShuffleBANet 参考分数，经全维与子空间 Fisher 融合和三动作门控得到检测分数。](docs/architecture.svg)

1. **从残差读出局部变化。** 模型从待测图自身估计修改代价，用代价排序与局部残差幅值划出五个区域。把区域内的四阶共现频率与已有残差特征拼接，得到 34,398 维描述。
2. **从两个视角打分。** 全维判别器 E 使用全部特征，子空间判别器 B 平均 32 个 512 维判别器的输出。种子 73 的验证集选中了等权组合，初始分数为 `Q = (E + B) / 2`。
3. **控制改判幅度。** 门控读取纹理、专家分歧和冻结的 ShuffleBANet 响应，为保留、上调和下调分配概率。训练时，纠正原判断得 +1，推翻正确判断得 −1，并加入同源普通图与载密图的排序损失。推理只需一张图，最终分数大于零即判为载密。

## 单种子盲测

三个检测器均按本项目的小样本协议训练。训练集为 160 对 HILL 0.4 图像，验证集为 48 对。锁定模型后，用与此前所有实验无重合的 256 张源照片构造五个测试条件，每列分别测试 256 张普通图和 256 张载密图。下表为准确率（%）。

| 模型 | WOW 0.1 | WOW 0.4 | HILL 0.1 | HILL 0.4 | J-UNIWARD 0.3 |
|---|---:|---:|---:|---:|---:|
| SRNet | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| ShuffleBANet | 51.17 | 53.13 | 50.59 | 52.15 | 49.22 |
| **CoTraceNet** | **51.56** | **70.12** | **51.17** | **69.73** | **49.80** |

HILL 与 WOW 的载荷单位为 bpp。J-UNIWARD 使用 QF85 的 JPEG，载荷为每个非零交流系数 0.3 比特。三个模型仅在 HILL 0.4 上训练。

在预先指定的 HILL 0.4 条件下，CoTraceNet 判对 **357/512** 张，比上一版四阶残差专家多判对 **22 张**，比上一版固定融合多判对 **27 张**。保留本版两个 Fisher 判别器并固定其权重，也判对 357 张。逐图门控救回 1 张、改错 1 张。低载荷和 JPEG 条件下的准确率均接近 50%。[实验协议和消融](experiments/README.md) · [汇总指标](experiments/seed73/metrics.csv) · [配对比较](experiments/seed73/paired.csv)

发布前使用同一组历史冻结权重，让此仓库的推理入口重新处理 HILL 0.4 的全部 512 张图。与原实验逐图对照，512 个分类判断全部一致，最终分数最大绝对误差小于 `4.7e-8`。另一次复算仅沿用原 CNN 骨干，使用公开代码重新拟合全部残差专家与门控，仍得到相同的 512 个判断、357 个正确分类和 0.7722168 的 AUC。

## 训练与推理

在仓库根目录安装 Python 依赖。训练支持 CUDA，单图推理也可在 CPU 上运行。若使用 GPU，请先安装与当前 CUDA 环境匹配的 PyTorch。

```sh
python -m pip install -r requirements.txt
python -m cotracenet.srm work/SRM_Kernels.npy
```

数据和中间产物由运行者保存在本地。训练清单使用 `source,split,cover,stego` 四列，两个图像路径相对于清单所在目录。`source` 标识原照片，同源普通图和载密图应在同一行、同一划分。训练清单包含 `train`、`val` 两组；测试清单使用 `test`。图像为 8 位、单通道、256 × 256。[BOSSbase 数据准备](docs/data.md)

```sh
python -m pip install -r tools/requirements-data.txt
python tools/prepare_pairs.py --archive path/to/BOSSbase_1.01.zip --out work/data
python -m cotracenet.train_backbone --manifest work/data/train/manifest.csv --srm work/SRM_Kernels.npy --out work/backbone --device cuda
python -m cotracenet fit --manifest work/data/train/manifest.csv --backbone work/backbone/best.pt --out work/model --device cuda
python -m cotracenet predict --image path/to/image.pgm --backbone work/backbone/best.pt --fit work/model --device cpu
python -m cotracenet score --manifest work/data/test/HILL_04/manifest.csv --backbone work/backbone/best.pt --fit work/model --out outputs/cotracenet/HILL_04/scores.csv --device cuda
```

对同一组源照片的其余四个条件分别执行 `score`，即可从输出的逐图分数重算汇总表。ShuffleBANet 的分数保存在该文件的 `baseline` 列。SRNet 对照可单独训练和评分。[五条件汇总命令及输出格式](experiments/README.md#recompute-the-summaries)

```sh
python -m cotracenet.srnet train --manifest work/data/train/manifest.csv --out work/srnet --device cuda
python -m cotracenet.srnet score --manifest work/data/test/HILL_04/manifest.csv --fit work/srnet --out outputs/srnet/HILL_04 --device cuda
```

## 原论文与外部资源

- ShuffleBANet 骨干按 Hao Li 等人的 *A Convolutional Neural Network Steganalysis Method Based on ShuffleBottleneck and Attention Mechanism* 独立实现。CoTraceNet 是在此基础上设计的课程研究方法。[原论文](https://doi.org/10.1109/TDSC.2025.3596710)
- SRNet 对照对应 Boroumand、Chen 和 Fridrich 的 *Deep Residual Network for Steganalysis of Digital Images*。[作者发布页](https://dde.binghamton.edu/download/feature_extractors/)
- 骨干预处理使用 SiaStegNet 提供的 30 个参考 SRM 滤波核。`cotracenet.srm` 从固定版本获取文件并校验 SHA-256。[上游文件](https://github.com/SiaStg/SiaStegNet/blob/592d9e13f39f287d9c675f1b89272bcfb6a9627c/src/models/SRM_Kernels.npy)

本仓库的原创代码与架构图按 MIT 许可证发布。外部数据集、滤波核和依赖分别遵循其上游使用条款。[MIT 许可证](LICENSE)
