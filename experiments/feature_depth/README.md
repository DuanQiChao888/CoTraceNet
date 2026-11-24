# CoTraceNet-Hier 多层特征提取实验

CoTraceNet-Hier 直接读取单张图像的 34,398 维残差共现频数。它分别编码全局、纹理和代价统计，在同一残差族内组合不同分区，再融合三路特征。新增分支含全词表的线性直连，最终与冻结的双视角奖励模型按源域验证集选出的比例融合。推理只需一张 256×256 的灰度图。[分层编码实现](../../cotracenet/feature_depth/hierarchy.py) · [单图入口](../../cotracenet/feature_depth/predict.py)

## 结果与对照

BOSSbase 1.01 的 10,000 张原图双三次缩放至 256×256，按 WOW 或 HILL 的 0.4 bpp 改动概率模拟载密图。每种条件各有 6,000 对训练、1,000 对验证、3,000 对测试。箭头从训练藏法指向测试藏法，各格按 6,000 张图计算准确率。

| 方法 | 融合模型参数 | WOW → WOW | HILL → HILL | WOW → HILL | HILL → WOW |
|---|---:|---:|---:|---:|---:|
| 同轮冻结的双视角奖励分类器 | 16,828 | 83.15% | 77.23% | 70.25% | 78.00% |
| **CoTraceNet-Hier，种子 73** | **82,461** | **84.83%** | **79.38%** | **71.30%** | **79.07%** |
| CoTraceNet-Hier，种子 29 | 82,461 | 84.85% | 79.18% | 71.02% | 78.98% |
| 三路残差 MLP，种子 73 | 83,857 | 84.17% | 77.57% | 70.88% | 78.48% |
| 三路浅层 MLP，种子 73 | 83,135 | 84.07% | 77.67% | 71.65% | 78.90% |
| 空间卷积分支，种子 73 | 78,525 | 83.15% | 77.23% | 69.92% | 78.00% |

同一组数据划分和冻结底座用于两次初始化。源验证集确定每个候选的训练轮数及融合权重，两个源域的验证正确数相加决定共用架构。首轮选择分层共现编码器。WOW 在第 24 轮选择新分支权重 0.75，HILL 在第 12 轮选择 0.75。种子 29 的 WOW、HILL 都选中第 24 轮，融合权重分别为 0.50 和 0.75。新增分支有 65,633 个参数，每图稠密层乘加为 661,166 MAC，即 1,322,332 FLOPs；此处只计新增稠密层。[首轮验证与测试](test-results.csv) · [首轮验证选型](validation-results.csv) · [第二种子验证](seed29-validation-results.csv) · [第二种子测试](seed29-test-results.csv)

以同轮完整基线为参照，首轮 WOW 同域净多判对 101 张，HILL 同域净多判对 129 张。把同一词表交给独立训练的线性模型并同样允许与基线融合，分层模型在两个同域分别多判对 100、129 张。新增分支单独运行时分别判对 5,064、4,699 张。各结构的逐图比较以原图对为单位重采样 10,000 次。[同输入线性对照](same-input-linear.csv) · [掩蔽路径、纯分支与融合](hierarchy-details.csv) · [同输入、相近参数的深浅 MLP](matched-depth.csv)

## 复现主模型

从仓库根目录执行。先按数据说明自行准备 BOSSbase 清单，并分别拟合 WOW 与 HILL 的 Lite 底座和双视角奖励分类器。下例沿用这些步骤产生的目录名。[数据准备](../../docs/data.md) · [双视角冻结模型及四向评分](../../docs/binocular.md)

```bash
python -m pip install -r requirements.txt
export FEATURES=work/joint-features
export POLICY=work/binocular/policy

for domain in WOW HILL; do
  manifest="work/boss/${domain}_04/manifest.csv"
  python -m cotracenet.feature_depth.cache --root "$FEATURES" --domain "$domain" \
      --manifest "$manifest" --splits train val --workers 4
  python -m cotracenet.feature_depth.hierarchy fit --train "$domain" \
      --root "$FEATURES" --policy "$POLICY" --out work/hier-73 --seed 73
done
```

三路描述子缓存一次后可供不同种子和各统计模型复用。训练和验证入口使用各自源域的数据。每种藏法的训练描述子磁盘大小约 3.08 GiB。训练完成后，对两个目标分别构建测试缓存，归档双视角评分。

```bash
for target in WOW HILL; do
  manifest="work/boss/${target}_04/manifest.csv"
  python -m cotracenet.feature_depth.cache --root "$FEATURES" --domain "$target" \
      --manifest "$manifest" --splits test --workers 4
  for source in WOW HILL; do
    python -m cotracenet.binocular score --domain "$source" --target "$target" \
        --manifest "$manifest" --out work/binocular
  done
done
for source in WOW HILL; do
  python -m cotracenet.feature_depth.hierarchy test --train "$source" \
      --root "$FEATURES" --policy "$POLICY" --out work/hier-73
done
python -m cotracenet.feature_depth.predict --fitted work/hier-73 \
    --attention-root work/binocular/attention --policy-root "$POLICY" predict --source WOW image.pgm
```

单图入口返回基线分数、新分支分数、融合分数和最终类别。可把 `fit` 改用 `--seed 29 --out work/hier-29` 后重跑两个源域。以下回放将每种目标藏法中排序最前的四对测试图逐张从 PGM 读入，再核对实际预测和批量缓存。

```bash
python -m cotracenet.feature_depth.predict --fitted work/hier-73 \
    --attention-root work/binocular/attention --policy-root "$POLICY" parity \
    --wow-manifest work/boss/WOW_04/manifest.csv --hill-manifest work/boss/HILL_04/manifest.csv \
    --output work/hier-73/parity.json
```

## 复现深浅、空间与线性对照

下面提供完整的 Linux / Bash 对照入口。三路 MLP 在每路训练数据内各筛选 512 维，深、浅版使用相同输入。空间卷积分支在整图两个尺度、四个方向的二阶残差上累计局部幅值和邻接共现，保留 4×4 网格。固定空间前端需要支持 C++17 的编译器。基线单图与分层主模型本身只依赖仓库的 Python 环境。

```bash
export GRIDS=work/spatial-grids
export STUDY=work/feature-depth-comparison
c++ -O3 -std=c++17 -shared -fPIC cotracenet/feature_depth/spatial_extract.cpp -o work/spatial_extract.so
for domain in WOW HILL; do
  python -m cotracenet.feature_depth.spatial_cache --manifest "work/boss/${domain}_04/manifest.csv" \
      --domain "$domain" --out "$GRIDS" --library work/spatial_extract.so --splits train val
done
bash experiments/feature_depth/study.sh fit

for domain in WOW HILL; do
  python -m cotracenet.feature_depth.spatial_cache --manifest "work/boss/${domain}_04/manifest.csv" \
      --domain "$domain" --out "$GRIDS" --library work/spatial_extract.so --splits test
done
bash experiments/feature_depth/study.sh test
```

`fit` 在新目录中训练首轮候选、单独的全词表线性模型，以及验证入选多层架构的第二个种子。它在测试前保存权重指纹、源域选型和验证结果。`test` 输出四向准确率、AUC、逐图纠错与引错数、配对区间、全词表线性对照及同输入 MLP 对照。分层模型入选时也导出两个种子的路径掩蔽结果。数据、权重和逐图分数由使用者在本地生成。[实验调度脚本](study.sh) · [模型选择与配对比较](../../cotracenet/feature_depth/compare.py)
