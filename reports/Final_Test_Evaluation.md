# Phase 7 Report — Final Test Evaluation

**Status:** ✅ Done · **Date:** 2026-11-28 · **Code:** `src/phase7_final_test.py` · **Results:** [`results/phase7_test_regression.csv`](../results/phase7_test_regression.csv), [`results/phase7_test_classification.csv`](../results/phase7_test_classification.csv)

## Goal
Measure the chosen models **once** on data they have never seen. These are the numbers the project reports as its result.

## Decisions fixed before opening the test set
| Decision | Choice |
|---|---|
| Final features | **Version B:** old features + route, airport traffic, distance to holidays (Phase 6b) |
| Regression model | LightGBM, `l1` objective, 300 trees, learning rate 0.1, 127 leaves |
| Classification model | LightGBM, cross entropy, same settings |
| Classification threshold | **0.17** for B, 0.18 for A (both chosen on validation) |
| Training data | **Train + validation** (4,894,825 flights, 313 days) |
| Reference | Version A (old features) measured as well, **only for comparison** |

These decisions are written at the top of the notebook. The test set (819,183 flights, 52 days) was used once, and **no choice was changed after seeing the results**.

**Why train + validation?** Validation was used to choose models, losses, settings, features and thresholds. Once those choices are fixed, its 854,000 flights can be added to training. The test set never took part in any choice or training.

All tables learned from data (airport traffic, category lists, the median of the baseline) were learned from train + validation only; the test set was only transformed.

---

## The test set looks like validation
| | Validation | Test |
|---|---|---|
| Baseline MAE | 18.80 | 18.76 |
| Delayed share (> 15 min) | 17.4% | 17.2% |
| Baseline accuracy | 82.6% | 82.8% |
| Baseline log loss | 0.462 | 0.459 |

The day-based split stratified by month (Phase 3) made test and validation very similar, so results on the two can be compared directly. Only 7 test flights use a route never seen in training, and 12 / 17 have no traffic value (handled as missing).

---

## Regression: how many minutes late?
| Model | Test MAE | Improvement over baseline |
|---|---|---|
| Baseline (median of train + validation, −5 min) | 18.757 | – |
| **LightGBM l1 — B (final)** | **17.725** | **1.032 min (5.5%)** |
| LightGBM l1 — A (reference) | 17.790 | 0.967 min (5.2%) |

| | Validation (trained on train) | Test (trained on train + validation) |
|---|---|---|
| B | 17.877 | **17.725** |
| A | 17.980 | 17.790 |
| B − A | −0.103 | −0.065 |

- **The final model beats the baseline by about one minute on unseen data.**
- The test result is slightly better than on validation, most likely because the model was trained on 0.85 million more flights (and partly because of which days ended up in test).
- **The gain from the new features holds on test** (B better than A), but it is smaller than on validation.

## Classification: more than 15 minutes late?
| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|---|
| Baseline | 0.50 | 0.828 | 0.000 | 0.000 | 0.000 | 0.500 | 0.459 |
| **LightGBM CE — B (final)** | **0.17** | 0.646 | **0.265** | **0.599** | **0.367** | **0.677** | **0.432** |
| LightGBM CE — A (reference) | 0.18 | 0.645 | 0.264 | 0.598 | 0.367 | 0.678 | 0.429 |

| | Validation | Test |
|---|---|---|
| ROC-AUC: A / B | 0.672 / **0.680** | **0.678** / 0.677 |
| F1: A / B | 0.365 / **0.373** | 0.367 / **0.367** |

### Confusion matrix (final model B, test, threshold 0.17)
| | Predicted "not delayed" | Predicted "delayed" |
|---|---|---|
| **Really not delayed** (678,382) | 444,615 | **233,767** false alarms |
| **Really delayed** (140,801) | **56,532** missed | **84,269** caught |

- **60% of delayed flights are flagged before departure** (recall 0.599).
- **26.5% of flagged flights are really delayed** (precision), against 17.2% for a random flight: a warning means about **1.5×** the average risk.
- About **2.8 false alarms per correct warning**, the same as on validation.
- Log loss is better than the baseline (0.432 vs 0.459): the probabilities carry real information, which matters for showing them on the website.
- F1 and ROC-AUC are slightly lower than on validation. This is expected: the threshold was chosen to maximise F1 on validation, so the validation F1 was slightly optimistic.

---

## Did the new features (Phase 6b) help?
| Task | Validation | Test |
|---|---|---|
| Regression | B better (−0.10 min) | **B better (−0.065 min)** ✅ |
| Classification | B better (ROC-AUC +0.008, F1 +0.008) | **No difference** (A ROC-AUC +0.001, log loss −0.002; B F1 +0.0006) ⚠️ |

- **For regression, the improvement holds on test**, a little smaller than on validation.
- **For classification, the improvement seen on validation did not repeat on test.** On test the two versions are practically equal; A is very slightly better on ROC-AUC and log loss.

**Why?** Two likely reasons:
1. **Selection effect:** versions A, B and C were compared on the same validation set and the best one was kept. The winner of such a comparison usually looks a little better than it really is; an independent test set removes that luck. This is exactly why a separate test set is kept.
2. **More data helped version A too:** A's ROC-AUC rose from 0.672 (validation, trained on train) to 0.678 (test, trained on train + validation). With more examples the model may learn route effects from the two airport columns by itself. This explanation is a hypothesis; the experiment cannot prove it.

**The final choice stays B**: it was fixed before the test, changing it now would turn the test set into a selection tool, and B is not worse than A on either task (better for regression, equal for classification). Using the same features for both tasks also keeps the website simpler.

---

## Where does the final model make larger errors? (regression, test)
### By month
| Month | Model MAE | Baseline MAE | Gain |
|---|---|---|---|
| 1 | 16.58 | 17.22 | 0.64 |
| 2 | 17.32 | 17.86 | 0.54 |
| 3 | 17.05 | 18.28 | 1.23 |
| 4 | 17.59 | 18.14 | 0.55 |
| 5 | 15.96 | 16.97 | 1.01 |
| 6 | **22.75** | 24.32 | **1.57** |
| 7 | 19.16 | 20.43 | 1.27 |
| 8 | 22.39 | 23.58 | 1.19 |
| 9 | 14.53 | 15.53 | 1.00 |
| 10 | **12.31** | 13.38 | 1.07 |
| 11 | 15.18 | 16.07 | 0.89 |
| 12 | 22.19 | 23.55 | 1.36 |

### By time of day
| Time of day | Model MAE | Baseline MAE | Gain |
|---|---|---|---|
| Night (0–5) | 12.72 | 14.15 | 1.43 |
| Morning (6–11) | 14.57 | 15.56 | 0.99 |
| Afternoon (12–17) | 19.34 | 20.25 | 0.91 |
| Evening (18–23) | **21.15** | 22.43 | 1.28 |

- **The model beats the baseline in every month and at every time of day.**
- **Errors depend strongly on the season:** from 12.3 min in October to 22.8 min in June. Summer storms and December holidays make delays larger and harder to predict. The largest gains are in the hardest months (June, December).
- **Errors grow steadily during the day** (12.7 min at night → 21.2 min in the evening): delays build up as aircraft carry earlier delays into later flights. The model does not know an aircraft's earlier flights, so evening flights are harder to predict.

---

## Summary
| Task | Final result on test | vs. baseline |
|---|---|---|
| **Regression** (minutes) | **MAE 17.73 min** | 1.03 min better (5.5%) |
| **Classification** (> 15 min late) | **ROC-AUC 0.677, F1 0.367, recall 60%** | Baseline catches no delay at all (82.8% accuracy, recall 0) |

The models give a **clear but modest** improvement with information known before departure. Validation and test agree, so the reported results can be trusted. Most of the delay is still caused by things that are unknown before the flight: weather, delays carried over from the aircraft's earlier flights, and operational problems.

## Outputs
| File | Location |
|---|---|
| `src/phase7_final_test.py` | GitHub |
| `results/phase7_test_regression.csv`, `results/phase7_test_classification.csv` | GitHub |
| `results/phase7_error_by_month.csv`, `results/phase7_error_by_time.csv` | GitHub |
| `results/phase7_test_predictions.parquet` | Kaggle output (for the charts in Phase 9) |

## Next phase
**Phase 8 — Model export for the website:** retrain the final models on all data (train + validation + test), build the route table (typical distance, duration and arrival offset per route), and write one prediction function that takes airline, airports, date and departure time and returns the expected delay and the probability of a delay over 15 minutes.
