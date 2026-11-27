# Phase 5 Report — Regression Model Comparison

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase5_regression_models.py` · **Results:** [`results/phase5_regression_results.csv`](../results/phase5_regression_results.csv)

## Goal
Predict **how many minutes a flight will be late** and compare four approaches on the same data:

| Model | How it works | Input format (Phase 4) |
|---|---|---|
| **Baseline** | Predicts the training median (−5 min) for every flight | — |
| **Linear / Ridge Regression** | One weight per column; prediction = weighted sum | One-hot (669 / 671 columns) |
| **RandomForest** | Average of many independent decision trees | One-hot |
| **LightGBM** | Trees trained one after another, each correcting the previous errors (boosting) | Category types (8 / 10 columns) |

- All models are trained on the **full train set** (4,040,848 flights) and evaluated on **validation** (853,977 flights).
- Measured: **validation MAE** (average error in minutes), **training time**, **model file size** (relevant for the website).
- Each model is run **without month** and **with cyclical month** (sin/cos) where time allowed.
- **The test set was not used.**

## Done criterion
> All models and the baseline are in one table with MAE, training time and size; a model is chosen on validation only → **✅ Met**

---

## Results

| Model | Month | Validation MAE | vs. baseline | Training time | Size |
|---|---|---|---|---|---|
| **Baseline (median)** | – | **18.80** | – | – | – |
| Linear Regression | no / cyclical | 20.62 / 20.64 | +1.82 / +1.84 | ≈ 36–64 s | 0.01 MB |
| Ridge Regression (α = 1) | no / cyclical | 20.62 / 20.64 | +1.82 / +1.84 | ≈ 29–43 s | 0.01 MB |
| Ridge + median shift | no | 18.79 | −0.01 | ≈ 35 s | 0.01 MB |
| LightGBM (l2) | no / cyclical | 20.37 / 20.32 | +1.57 / +1.52 | ≈ 12 s | 0.45 MB |
| LightGBM (l1) | no / cyclical | 18.23 / 18.17 | −0.57 / −0.63 | ≈ 19 s | 0.46 MB |
| RandomForest (50 trees) | cyclical | 20.34 | +1.54 | **≈ 1,505 s (25 min)** | **34.5 MB** |
| RandomForest + median shift | cyclical | 18.75 | −0.05 | ≈ 1,505 s | 34.5 MB |

*Training times vary between runs (they depend on the load of the Kaggle machine); only the large differences are meaningful. MAE values were identical across runs.*

### LightGBM (l1) settings (cyclical month)
| Trees | Learning rate | Leaves | Validation MAE | vs. baseline | Time | Size |
|---|---|---|---|---|---|---|
| 100 (default) | 0.1 | 31 | 18.17 | −0.63 | 19 s | 0.46 MB |
| 300 | 0.1 | 31 | 18.06 | −0.74 | 48 s | 1.31 MB |
| **300** | **0.1** | **127** | **17.98** | **−0.82** | **63 s** | **4.88 MB** |
| 600 | 0.05 | 127 | 17.96 | −0.84 | 122 s | 9.71 MB |
| 600 | 0.05 | 255 | 17.94 | −0.86 | 137 s | 18.45 MB |

---

## Findings

### 1. The objective matters more than the model
Linear Regression, Ridge, RandomForest and LightGBM (l2) all minimise **squared error**, which pulls predictions toward the **mean** delay (+4.6 min in train). Their mean predictions are +4.3 to +4.5 min, and all of them are **worse than the baseline** (20.3–20.6 vs 18.80). The baseline predicts the **median** (−5 min), and MAE is smallest at the median (Phase 1: 19.15 vs 21.26 on the full data).

The clearest evidence: **the same LightGBM model with the same data and settings** goes from **20.32 to 18.17** when only the objective changes from `l2` (squared error) to `l1` (absolute error), and its mean prediction moves from +4.5 to −4.7 min.
**Lesson: train the model on the error you are going to measure.**

### 2. A median shift fixes the level, but not more
Adding a constant shift (the median of the training errors: −8.61 min for Ridge, −7.83 min for RandomForest, learned from train only) removes the mean-vs-median gap:
- Ridge: 20.62 → **18.79** (baseline 18.80)
- RandomForest: 20.34 → **18.75**

Both end up **almost equal to the baseline**. A shift moves every prediction by the same amount; LightGBM (l1) instead aims at the median **separately for every kind of flight** while it learns, and is clearly better (18.17).

### 3. Linear models learn almost nothing here
Even after the shift, Ridge only matches the baseline. Delay does not change linearly with the features: an airport's effect depends on the hour and the season. Trees can combine conditions (e.g. *origin = ORD and evening and winter*); a linear model cannot. Ridge's penalty (α = 1) has no visible effect with 4 million flights, so Ridge and Linear Regression give identical results; Ridge is about 1.5–2× faster.

### 4. Cyclical month helps tree models a little
| Model | No month | Cyclical month |
|---|---|---|
| Linear / Ridge | 20.62 | 20.64 |
| LightGBM (l2) | 20.37 | 20.32 |
| LightGBM (l1) | 18.23 | **18.17** |

Small but consistent gains for LightGBM; no gain for linear models. This is possible only because the day-based split (Phase 3) puts all months in train.

### 5. RandomForest is slow and large for the same accuracy
With limits (50 trees, 20% of the rows per tree, ≥ 100 flights per leaf, 30% of the columns per split) RandomForest still needed **≈ 25 minutes** and **34.5 MB**, about 130× slower and 75× larger than LightGBM (l2), for the same MAE. Training it on absolute error (`criterion="absolute_error"`) would take days at this data size, so it could not be tested.

### 6. Tuning LightGBM: diminishing returns
More trees and larger trees improve validation MAE from 18.17 to 17.94, but the last two steps add only 0.02 min each while the model grows from 4.9 to 18.5 MB. No sign of overfitting: validation error keeps decreasing slightly as the model grows.

---

## Chosen model
**LightGBM, objective `l1`, 300 trees, learning rate 0.1, 127 leaves, with cyclical month.**

| | Value |
|---|---|
| Validation MAE | **17.98 min** |
| Improvement over baseline | **0.82 min (4.4%)** |
| Training time | ≈ 1 minute |
| Size | 4.9 MB |

**Why this one:** it is within 0.04 min (2–3 seconds) of the best setting (17.94) but about 4× smaller and 2× faster, which matters for the website. The choice was made on validation only, before looking at the test set, and will not be changed after Phase 7.

**How large is the improvement?** 0.8 minutes looks small, but Phase 1 showed that no allowed feature has a correlation above 0.10 with the delay: most of the delay comes from things that are unknown before the flight (weather, the aircraft's previous flight, operational problems). A clear but modest gain over the baseline is the realistic result for pre-departure information.

### Feature importance (chosen model, gain)
| Feature | Share |
|---|---|
| `ORIGIN_AIRPORT` | 30.9% |
| `DESTINATION_AIRPORT` | 27.0% |
| `AIRLINE` | 9.3% |
| `DEP_MIN` (departure time) | 8.0% |
| `MONTH_SIN` | 6.6% |
| `DAY_OF_WEEK` | 5.8% |
| `SCHEDULED_TIME` | 4.4% |
| `MONTH_COS` | 3.7% |
| `ARR_MIN` (arrival time) | 2.4% |
| `DISTANCE` | 1.9% |

- **Airports** carry most of the information (58% together); **time of day** and **season** about 10% each.
- `DISTANCE` is low because it carries almost the same information as `SCHEDULED_TIME` (correlation ≈ 0.98).
- Gain importance tends to favour columns with many different values (322 airports), so the ranking is more reliable than the exact percentages.

---

## Problems met and how they were solved
| Problem | Cause | Fix |
|---|---|---|
| Train/val sizes of the old month-based split appeared | The Phase 5 notebook was reading an older version of the Phase 3 output | Updated the Phase 3 input to its latest version; checked that the baseline is 18.80 |
| `TypeError: ... not NoneType` when importing LightGBM | `sys.path` contained `None` (the setup cell had once run while the Phase 4 input was missing) | Removed `None`; restarted the kernel; added `assert` checks for both inputs and import LightGBM in the first cell |
| `module 'dask' has no attribute 'utils'` | A library was left half-loaded by the first failed import | Restarting the kernel |
| RandomForest takes ~25 min | 4 M rows × 671 one-hot columns | Limits on trees, rows per tree, leaf size and columns per split; results saved to file right after training |

## Outputs
| File | Location |
|---|---|
| `src/phase5_regression_models.py` | GitHub |
| `results/phase5_regression_results.csv` | GitHub |
| `results/phase5_feature_importance.csv` | GitHub |
| `results/phase5_val_predictions.parquet` | Kaggle output only (large; used for the charts) |
| `results/lgbm_regression.joblib`, `results/categories.joblib` | Kaggle output (inputs for Phases 7–8) |

## Next phase
**Phase 6 — Classification model comparison:** predict whether a flight will be **more than 15 minutes late**, using the same features and split. Baseline accuracy 82.6% (always "not delayed"); metrics: accuracy, precision, recall, F1, ROC-AUC.
