# CoTraceNet-Lite on two HILL 0.4 cohorts

Training used seed 73 and 160 paired source photographs. Each source's cover and stego stayed together in four-fold training and evaluation. The 48 validation pairs were used for model selection. The primary endpoint was single-image accuracy on a newly drawn cohort of 256 sources, with the decision threshold fixed at zero. The original five-condition cohort provides a secondary comparison. [Aggregate metrics](metrics.csv) · [Paired differences](paired.csv)

| Detector | Weights and biases | Original HILL 0.4 | Fresh HILL 0.4 | Fresh AUC |
|---|---:|---:|---:|---:|
| CoTraceNet | 730,717 | 357/512 | 346/512 | 0.7555 |
| Same-expert fixed fusion Q | 34,399 | 357/512 | 351/512 | 0.7554 |
| CoTraceNet-Lite | 16,385 | 363/512 | 348/512 | 0.7586 |

On the fresh test, Lite corrected 12 errors by CoTraceNet and introduced 10. The paired accuracy difference was +0.39 percentage points (95% source-bootstrap interval −1.17 to +1.95). Against Q the difference was −0.59 points (−2.15 to +0.98). The two AUCs for Lite were 0.7755 on the original cohort and 0.7586 on the fresh cohort. A single training seed and these intervals leave uncertainty about small gains.

### Three compression approaches

All three families were compared on development data before any fresh candidate test scores were opened. Within the sparse family, 12 recipes were assessed on four source-grouped folds plus validation. Each fold fitted its own teacher, feature ranking and candidate heads. Final nominees were compared by mean out-of-fold and validation accuracy, then mean AUC, then parameter count. The sparse nominee was selected. Only training data were used to fit its final coefficients.

| Family nominee | Parameters | Out-of-fold | Validation | Fresh test |
|---|---:|---:|---:|---:|
| Prune, refit, distill | 16,385 | 205/320 | 70/96 | 348/512 |
| Structured histogram pooling | 24,259 | 210/320 | 66/96 | 339/512 |
| 2D residual neighborhoods combined with Q | 38,359 | 207/320 | 69/96 | 344/512 |

Q is the equal-weight sum of the two final experts, with its weights and bias folded into a single affine readout. The full model also evaluates the CNN and the auxiliary scores supplied to the gate. Counts include dense inference weight and bias slots. Feature extraction is deterministic. Lite stores 16,384 unsigned 16-bit indices in addition to its floating-point weights and bias. The fitted Lite NPZ was 157,129 bytes. On the experiment's RTX 3090, replaying training from all 416 training and validation images produced the identical model SHA-256, `b8019a3272f2a05f7ee94a73fc32d387f3f6a25951a2f1d53f68f136cbc16513`.

### Reproduce the comparison

Run from the repository root. Follow the [data preparation](../../docs/data.md) and [training commands](../../README.md) to create the two test cohorts, the full backbone, the full model and the Lite model. The example uses the same full-model directory as the root README.

```sh
python -m cotracenet score --backbone work/backbone/best.pt --fit work/model --manifest work/fresh/test/HILL_04/manifest.csv --out outputs/fresh_compare/full.csv --device cuda
python -m cotracenet.lite score --fit work/lite --manifest work/fresh/test/HILL_04/manifest.csv --out outputs/fresh_compare/lite.csv
python experiments/summarize.py --conditions HILL_04 --target CoTraceNet-Lite --table outputs/fresh_compare/full.csv CoTraceNet=cotracenet same_expert_fixed_Q=strict_fixed --table outputs/fresh_compare/lite.csv CoTraceNet-Lite=cotracenet_lite --out outputs/fresh_compare/summary
```

For the original cohort, use its own manifest and a separate output directory. The summarizer aligns both detector files by source and class, then resamples complete source pairs 10,000 times with PCG64 seed 7319. Both detectors use the same draws. The original cohort's five test conditions share photographs.

The exploratory families can also be fitted and scored from images.

```sh
python experiments/compact_ablations.py fit --manifest work/data/train/manifest.csv --out work/ablations --device cuda
python experiments/compact_ablations.py score --manifest work/fresh/test/HILL_04/manifest.csv --fit work/ablations --out outputs/fresh/ablations.csv
```

### Single-core inference benchmark

The original timing used one pinned core of an AMD EPYC 7H12. It took the first 12 fresh source pairs (24 images), warmed the file cache, kept weights resident and timed three repeats from image decoding through the final score. Each model received a single unpadded image. Model order alternated across images and repeats. The full detector's median was 313.98 ms (IQR 307.95–317.81), compared with 64.51 ms (63.41–65.23) for Lite, a 4.87× speedup. Timings depend on hardware and system load.

```sh
python experiments/benchmark.py --manifest work/fresh/test/HILL_04/manifest.csv --backbone work/backbone/best.pt --full-fit work/model --lite-fit work/lite --out outputs/latency.json
```
