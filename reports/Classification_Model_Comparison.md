# Phase 6 Report — Classification Model Comparison

**Status:** ✅ Done · **Date:** 2026-11-28 · **Code:** `src/phase6_classification_models.py` · **Results:** [`results/phase6_classification_results.csv`](../results/phase6_classification_results.csv)

## Goal
Predict whether a flight will be **more than 15 minutes late** (the official US definition of a delayed flight). The model outputs a **probability** ("34% chance of a delay"), which is turned into a yes/no decision with a **threshold**.

- Target: `1` if `ARRIVAL_DELAY > 15`, otherwise `0`. Delayed share: **18.2%** in train, **17.4%** in validation.
- Same features, split and preprocessing as Phase 5. All models trained on the **full train set** (4,040,848 flights), compared on **validation** (853,977 flights).
- **The test set was not used.**

## Done criterion
> All models and the baseline are compared with the same metrics; a model and a threshold are chosen on validation only → **✅ Met**

---

## Metrics
| | Predicted "not delayed" | Predicted "delayed" |
|---|---|---|
| **Really not delayed** | true negative | false positive (false alarm) |
| **Really delayed** | false negative (missed) | true positive (caught) |

| Metric | Question |
|---|---|
| Accuracy | What share of all predictions is correct? (misleading here, see baseline) |
| Precision | Of the flights we call "delayed", how many really are? |
| Recall | Of the really delayed flights, how many do we catch? |
| F1 | Balance of precision and recall |
| ROC-AUC | How well does the model rank delayed flights above on-time flights? Independent of the threshold. 0.5 = coin toss, 1.0 = perfect. |
| Log loss | Cross entropy: how accurate are the probabilities? Lower is better. |

## Models
| Model | Loss | Input |
|---|---|---|
| Baseline | – (always "not delayed", probability = train delayed share) | – |
| Logistic Regression | cross entropy | one-hot, 671 columns |
| LightGBM | cross entropy | category types, 10 columns |
| LightGBM + class weight | weighted cross entropy (`scale_pos_weight = 4.5`) | same |
| LightGBM + focal loss | focal loss, γ = 2 (custom objective) | same |

All LightGBM versions use the settings chosen in Phase 5 (300 trees, learning rate 0.1, 127 leaves), so **the only difference between them is the loss function**.

### Focal loss
Cross entropy penalises each flight with `−log(p)`, where p is the probability given to the true class. Focal loss multiplies this by `(1 − p)^γ`:
- easy example (p = 0.95): penalty × 0.0025 (400× smaller)
- hard example (p = 0.10): penalty × 0.81 (almost unchanged)

The model therefore focuses on hard examples. With γ = 0 it equals cross entropy.

LightGBM has no built-in focal loss, so it was written as a **custom objective**: LightGBM needs the gradient (first derivative) and the hessian (second derivative) of the loss for every flight. They are computed **numerically** (central differences with step 0.001); the hessian is kept above 10⁻⁶ because LightGBM needs positive values.

**Check:** with γ = 0, the numeric gradient and hessian match the exact cross-entropy formulas (`p − y` and `p(1 − p)`) to 4 decimals:

| Raw score | Label | Numeric gradient | Exact | Numeric hessian | Exact |
|---|---|---|---|---|---|
| −2 | 1 | −0.8808 | −0.8808 | 0.105 | 0.105 |
| 0 | 0 | 0.5 | 0.5 | 0.25 | 0.25 |
| 2 | 1 | −0.1192 | −0.1192 | 0.105 | 0.105 |

The model starts from the delayed share (raw score −1.505 → probability 0.182) via `init_score`; without it, a custom objective would start from 50%.

---

## Results at threshold 0.5
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Training time |
|---|---|---|---|---|---|---|---|
| Baseline | 0.826 | 0.000 | 0.000 | 0.000 | 0.500 | 0.462 | – |
| Logistic Regression | 0.826 | 0.388 | 0.001 | 0.003 | 0.635 | 0.446 | 50 s |
| LightGBM (CE) | 0.825 | 0.469 | 0.042 | 0.077 | **0.672** | **0.436** | 67 s |
| LightGBM (weighted) | 0.644 | 0.264 | 0.585 | 0.364 | 0.670 | 0.624 | 65 s |
| LightGBM (focal) | 0.826 | 0.487 | 0.037 | 0.068 | **0.672** | 0.517 | **243 s** |

- **The baseline reaches 82.6% accuracy without catching a single delay** (recall 0, ROC-AUC 0.5). Accuracy alone is misleading for this data.
- At 0.5, the models without weights almost never say "delayed": with pre-departure information, few flights reach a 50% delay probability.

## Results at the best threshold (highest F1 on validation)
| Model | Threshold | Accuracy | Precision | Recall | **F1** | **ROC-AUC** | **Log loss** |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.17 | 0.561 | 0.231 | 0.659 | 0.343 | 0.635 | 0.446 |
| **LightGBM (CE)** | **0.18** | 0.637 | 0.262 | 0.601 | **0.365** | **0.672** | **0.436** |
| LightGBM (weighted) | 0.50 | 0.644 | 0.264 | 0.585 | 0.364 | 0.670 | 0.624 |
| LightGBM (focal) | 0.36 | 0.630 | 0.260 | 0.613 | 0.365 | 0.672 | 0.517 |

*The threshold was chosen and measured on the same validation data, so these F1 values are slightly optimistic; Phase 7 measures them on the test set.*

---

## Findings

### 1. Trees beat the linear model
LightGBM ranks flights better than Logistic Regression (ROC-AUC 0.672 vs 0.635; F1 0.365 vs 0.343), as in Phase 5: trees can combine conditions (airport × hour × season).

### 2. Class weight and focal loss did not improve the model, they only moved the threshold
- **ROC-AUC is the same** for cross entropy, weighted cross entropy and focal loss (0.670–0.672): the ability to separate delayed from on-time flights did not change.
- **At their best thresholds, all three reach the same F1** (0.364–0.365). The weighted model's best threshold is exactly 0.50; the unweighted model gets the same result at 0.18.
- **Both make the probabilities less accurate:** mean predicted probability 0.445 (weighted) and 0.347 (focal) vs a real delayed share of 0.174; log loss rises from 0.436 to 0.624 and 0.517. The class weight pushes all probabilities up; focal loss pushes them toward 0.5 because it stops "caring" about easy, clearly on-time flights.
- Focal loss was also 3.6× slower (custom objective computed in Python).

**Why focal loss did not help here:** it was designed for problems where the classes are clearly separable but the minority class is rare (e.g. finding small objects in images). In this problem the limit is not the class imbalance but the **information**: pre-departure features only weakly predict delays (Phase 1: no feature correlates above 0.10 with the delay). Focusing on hard examples cannot find information that is not in the data.

### 3. A low threshold is needed
The best thresholds (0.17–0.18 for unweighted models) are close to the delayed share (18%): the model raises a warning when a flight is riskier than an average flight.

---

## Chosen model
**LightGBM, cross entropy (`objective="binary"`), 300 trees, learning rate 0.1, 127 leaves, cyclical month; threshold 0.18.**

| Reason | |
|---|---|
| F1 and ROC-AUC | Equal to the best (0.365 / 0.672) |
| **Log loss** | **Best (0.436): the most accurate probabilities.** The website will show "x% chance of a delay", so the probability itself must be trustworthy. |
| Speed and simplicity | Standard built-in loss; 3.6× faster than focal loss; no custom code needed on the website |

The choice was made on validation only and will not be changed after Phase 7.

### Confusion matrix (validation, threshold 0.18)
| | Predicted "not delayed" | Predicted "delayed" |
|---|---|---|
| **Really not delayed** (705,649) | 454,892 | **250,757** false alarms |
| **Really delayed** (148,328) | **59,193** missed | **89,135** caught |

- **60%** of delayed flights are flagged in advance (recall 0.601).
- **26%** of flagged flights are really delayed (precision 0.262), compared with 17.4% for a random flight: a warning means about **1.5×** the average risk.
- There are about **2.8 false alarms per correct warning**. For the website, the honest way to present the result is a **risk level with its probability** ("above-average delay risk: 34%"), not a "this flight will be late" promise.

---

## Possible improvements (not part of this phase)
The model is limited by the information, not by the algorithm: changing the loss changed nothing, and switching from a linear model to trees helped only once. Larger gains would need **more pre-departure information**, for example airport traffic per hour, the route as a category, the route's historical delay rate, or distance to public holidays. Weather forecasts and the aircraft's earlier flights would probably help most, but they are not in this dataset or cannot be entered by a website user.

## Outputs
| File | Location |
|---|---|
| `src/phase6_classification_models.py` | GitHub |
| `results/phase6_classification_results.csv` | GitHub |
| `results/phase6_val_probabilities.parquet` | Kaggle output (for ROC and confusion-matrix charts) |
| `results/lgbm_classifier.joblib`, `results/categories.joblib` | Kaggle output (inputs for Phases 7–8) |

## Next phase
**Phase 7 — Final test evaluation:** retrain the chosen regression and classification models on train + validation, open the test set once, and report MAE (regression) and accuracy / precision / recall / F1 / ROC-AUC (classification at threshold 0.18), compared with the baselines and with the validation results.
