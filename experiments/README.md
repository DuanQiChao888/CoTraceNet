# Seed 73 experiment

The primary endpoint was single-image detection accuracy on HILL at 0.4 bits per pixel (bpp). The two prespecified controls were the earlier fourth-order residual expert C and the earlier validation-selected fixed fusion F. All measurements below use the fourth held-out cohort.

| Detector | WOW 0.1 | WOW 0.4 | HILL 0.1 | HILL 0.4 | J-UNIWARD 0.3 |
|---|---:|---:|---:|---:|---:|
| SRNet | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| ShuffleBANet | 51.17 | 53.13 | 50.59 | 52.15 | 49.22 |
| **CoTraceNet** | **51.56** | **70.12** | **51.17** | **69.73** | **49.80** |

Entries are percentages. The baselines are project reimplementations trained under the small-data protocol. SRNet trained for 12 epochs and its first epoch had the best validation AUC. The measured JPEG condition uses J-UNIWARD at QF85 and 0.3 bits per nonzero AC coefficient (bpnzAC).

## What the ablations establish

On the 512 primary-condition images CoTraceNet correctly classified 185 stego images and 172 covers. The table compares the same inputs at the same frozen decision threshold.

| HILL 0.4 ablation | Correct / 512 | Accuracy | AUC |
|---|---:|---:|---:|
| Earlier texture expert C | 335 | 65.43% | 0.7353 |
| Earlier fixed fusion F | 330 | 64.45% | 0.7244 |
| Full-feature Fisher expert E | 350 | 68.36% | 0.7718 |
| Random-subspace Fisher expert B | 353 | 68.95% | 0.7633 |
| Final experts, fixed fusion Q | 357 | 69.73% | 0.7719 |
| Final gate trained without the explicit reward | 358 | 69.92% | 0.7718 |
| **CoTraceNet** | **357** | **69.73%** | **0.7722** |

F combines the earlier third-order residual expert and C at weights 0.25 and 0.75. Q uses the same final experts as CoTraceNet and combines E and B at equal weights. Against C, CoTraceNet corrected 63 previously incorrect decisions and introduced 41 errors, gaining 4.30 percentage points (paired 95% interval 1.37 to 7.23). Against F the counts were 70 and 43, a gain of 5.27 points (2.34 to 8.40). Against Q it corrected one error and introduced one, for a difference of 0.00 points (−0.59 to +0.59). On these photos the complete detector improved over C and F, while the learned adjustment gave Q no net accuracy gain. At the lower spatial payload and in the JPEG domain, the accuracy of the complete detector was close to chance.

## Data and selection protocol

- Training used 160 cover/stego pairs from BOSSbase 1.01 with HILL at 0.4 bpp. Another 48 source-disjoint pairs were used repeatedly for design and selection. The model and threshold were locked before scoring the final cohort. Training and evaluation ran on NVIDIA RTX 3090 GPUs.
- Each test column has 256 covers and 256 stego images from the same 256 new source photographs. A uniform draw without replacement used NumPy PCG64 seed 20261126 on the numerically sorted source pool after excluding 1,152 previously used source photographs. The training seed was 73. Source images were resized to 256 × 256 grayscale with bicubic interpolation.
- Spatial test pairs used HILL and WOW embedding simulations. JPEG test pairs used conSeal's original J-UNIWARD simulation with real quantized DCT coefficient changes. Both cover and stego JPEGs were decoded with the same codec. The JPEG condition changes the embedding algorithm, coding domain and payload unit together.
- A score greater than zero predicts stego. Each image gets an independent score. Source pairing is used when creating splits, training paired losses and estimating uncertainty. The AUC calculation awards half a win to equal scores.
- For every accuracy contrast we sort the source identifiers lexicographically, keep each cover with its stego counterpart and keep both models' predictions together. We resample 256 source groups with replacement 10,000 times using PCG64 seed 7319. The 2.5th and 97.5th percentiles of the mean paired differences form the interval. These intervals describe test-source sampling variation for one fixed training seed.

The two published CSVs contain only aggregate results. One row in `metrics.csv` gives a detector's confusion counts, accuracy and AUC for one condition. One row in `paired.csv` gives CoTraceNet's corrections, new errors, accuracy difference and interval against a reference. Accuracy is in percent and differences are in percentage points. [Metrics](seed73/metrics.csv) · [Paired differences](seed73/paired.csv)

## Recompute the summaries

Save a score file for each condition with a header, a stable `source` key, `label` (0 for cover or 1 for stego) and one numeric column per detector. The same source keys must identify the same original photos throughout. Each method should supply the signed score used by its frozen decision threshold. For the final project scorer, the default column mapping is built into the script. The ShuffleBANet logit is included automatically when the file contains a `baseline` column. The following command also imports separately scored SRNet predictions.

```sh
python experiments/summarize.py \
  --table 'outputs/srnet/{condition}/predictions.csv' SRNet=score \
  --table 'outputs/cotracenet/{condition}/scores.csv' \
  --out outputs/summary
```

The default condition directories are `WOW_01`, `WOW_04`, `HILL_01`, `HILL_04` and `JPEG_03`. To compare just the main condition with user-selected column names, specify the mapping explicitly.

```sh
python experiments/summarize.py \
  --conditions HILL_04 \
  --table outputs/hill_scores.csv CoTraceNet=cotracenet texture_expert_C=texture previous_fixed_F=fixed_matched same_expert_fixed_Q=strict_fixed \
  --out outputs/hill_summary
```
