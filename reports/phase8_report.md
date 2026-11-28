# Phase 8 Report — Model Export for the Website

**Status:** ✅ Done · **Date:** 2026-11-28 · **Code:** `src/phase8_model_export.py`, `src/predict.py` · **Model files:** `model/`

## Goal
Prepare everything the website needs to predict one flight from five inputs: **airline, origin airport, destination airport, date and departure time.**

No model choice is made in this phase. The models, features and settings are the ones fixed before the test (Phase 7), and **the measured performance is the one reported in Phase 7** (test MAE 17.73 min, ROC-AUC 0.677).

---

## 8.1–8.3 Final models trained on all data
| | Value |
|---|---|
| Training data | **All usable flights: 5,714,008** (train + validation + test, 365 days) |
| Features | Version B, 14 columns (Phase 6b) |
| Routes | 4,657 |
| Delayed share (> 15 min) | 17.9% |
| Missing values | **0** in every column (traffic tables and categories now come from the whole year) |

| Model | Settings | Training time | File size |
|---|---|---|---|
| Regression | LightGBM, `l1`, 300 trees, lr 0.1, 127 leaves | 177.7 s | 12.4 MB |
| Classification | LightGBM, cross entropy, same settings | 120.5 s | 11.7 MB |

**Why train on test as well?** The test set's job ended in Phase 7: the result was measured once and reported. No further choice is made, so the model for real use is trained on every available example. This final model cannot be measured any more; its performance is expected to be close to (or slightly better than) the Phase 7 test result.

---

## 8.4 Route table
The model needs 14 columns, but a website user only knows five things. Distance, scheduled flight time and scheduled arrival time are taken from the **typical (median) value of the route in 2015**.

| Route | Distance | Scheduled time | Arrival − departure (local clocks) | Flights |
|---|---|---|---|---|
| SFO → LAX | 337 | 86 min | 86 min | 14,756 |
| LAX → SFO | 337 | 81 min | 81 min | 14,431 |
| JFK → LAX | 2,475 | 379 min | **199 min** | 12,948 |
| LAX → JFK | 2,475 | 328 min | **508 min** | 12,942 |
| LAS → LAX | 236 | 72 min | 72 min | 10,589 |

`ARR_OFFSET` = scheduled arrival minus scheduled departure, in local clock time (`% 1440` for flights past midnight). It contains both the flight time and the **time zone difference**: JFK → LAX takes 379 minutes but lands only 199 minutes later on the clock (−3 hours), LAX → JFK 328 + 180 = 508. The arrival time therefore cannot be computed from the flight time; the offset per route is stored.

### Route only, or airline + route?
On the same route, airlines can have different schedules, so a table per airline + route was also measured (mean absolute difference between each flight's real value and the table value, all data):

| Table | Scheduled time error | Arrival offset error |
|---|---|---|
| Route only | 4.33 min | 6.65 min |
| Airline + route | 3.96 min | 6.26 min |

Adding the airline reduces the error by only ~0.4 min. Most of the remaining difference comes from schedules changing with the time of day on the same route, which neither table captures. This is very small compared with the model's own error (~17.7 min), so the **simpler route-only table is kept**.

---

## 8.5 Lists for the website
| File | Content |
|---|---|
| `airline_list.csv` | **13 airlines** with names. US Airways (`US`) is removed: it merged with American Airlines and only has flights in the first half of 2015 (Phase 1). Its flights were still used for training. |
| `airport_list.csv` | **322 airports** with name, city and state (all codes have a name thanks to the October fix in Phase 2) |
| `airline_routes.csv` | **8,222 airline–route pairs**: after an airline is chosen, the website only offers routes that the airline really flew, so the model is never asked about a combination it has not seen |

---

## 8.6 The prediction function (`src/predict.py`)
| Function | What it does |
|---|---|
| `load_model_files(folder)` | Loads the two models, the category lists, the traffic tables and the route table **once**, when the website starts |
| `predict_flight(files, airline, origin, destination, date, departure_time)` | Builds the one-row table with the **same functions used in training** (`preprocess.add_features`, `features.add_new_features`, `features.make_table`) and returns the expected delay, the probability of a delay over 15 minutes, a risk level and the estimated arrival time |
| `risk_level(probability)` | low < 0.17 ≤ medium < 0.30 ≤ high |

Checks:
- `date.isoweekday()` (Monday = 1) matches the dataset's `DAY_OF_WEEK` for **100%** of flights.
- A route that does not exist in the data raises an error instead of returning an unreliable prediction.

**The year is not a feature.** The model uses month, day of week and days to the nearest US federal holiday (computed for any year). A prediction for a 2026 date therefore means "how did flights in these conditions behave in 2015?". The website should say this clearly.

---

## 8.7 Checks
### Function vs. model on real rows (5 random flights)
| Flight | Date | Dep. | Real arr. | Function arr. | Delay: real row | Delay: function | Probability: real row | Probability: function |
|---|---|---|---|---|---|---|---|---|
| EV SHV→DFW | 2015-01-16 | 1220 | 1320 | 1325 | −0.8 | −0.7 | 0.225 | 0.214 |
| WN PHL→MCO | 2015-11-11 | 1210 | 1445 | 1445 | −13.2 | −13.2 | 0.071 | 0.071 |
| MQ DFW→GRR | 2015-03-03 | 930 | 1250 | 1300 | 19.8 | 15.3 | 0.523 | 0.525 |
| WN DAL→PHX | 2015-06-11 | 1030 | 1055 | 1105 | −0.8 | −0.6 | 0.120 | 0.132 |
| NK MSY→IAH | 2015-04-24 | 855 | 1005 | 1013 | 1.8 | −1.5 | 0.213 | 0.175 |

When the flight's schedule equals the route's typical schedule, the function gives exactly the model's prediction (WN PHL→MCO). Otherwise the small differences come only from the typical values in the route table (arrival 5–10 minutes off).

### Example questions
| Flight | Date, time | Expected delay | Probability | Risk |
|---|---|---|---|---|
| AA JFK → LAX | 14 Oct, 07:00 | −21.7 min (early) | 4.4% | low |
| AA JFK → LAX | 14 Oct, 19:00 | −20.3 min | 11.4% | low |
| AA JFK → LAX | 26 Jun, 19:00 | −2.0 min | 23.0% | medium |
| AA JFK → LAX | 23 Dec, 19:00 | **+13.4 min** | **50.4%** | **high** |
| DL ATL → LGA | 26 Jun, 19:00 | +4.5 min | 37.7% | high |

- The model follows the patterns found in Phase 7: the risk is lowest on an October morning and highest just before Christmas and on summer evenings.
- A **negative expected delay means an early arrival**; long flights often have extra time in the schedule. The website should show "about 20 min early".
- The expected delay is a **median** prediction, so it stays modest even when the probability is high (50% → +13 min), because delays are very skewed (Phase 1). Showing both numbers together gives the clearest picture.

---

## 8.8 Are the risk levels meaningful?
Checked on the Phase 7 test predictions (model trained on train + validation, flights never seen):

| Risk level | Share of flights | Really delayed (> 15 min) | Median real delay |
|---|---|---|---|
| Low (< 0.17) | 61.2% | **11.3%** | −7 min |
| Medium (0.17–0.30) | 25.1% | **21.6%** | −3 min |
| High (≥ 0.30) | 13.7% | **35.4%** | +5 min |
| All flights | 100% | 17.2% | |

- The real delay rate **rises clearly from level to level**: a "high" flight is delayed **about 3 times as often** as a "low" one (35.4% vs 11.3%), and about 2 times the average.
- The probabilities are also well calibrated in each level (for example, "high" flights have probabilities ≥ 0.30 and 35% of them are really delayed).
- Even in the "high" level, most flights (65%) are not delayed. The website should present the result as a **risk**, not as a yes/no answer.

---

## Outputs
| File | Location |
|---|---|
| `src/phase8_model_export.py`, `src/predict.py` | GitHub |
| `model/regression_model.joblib`, `model/classification_model.joblib`, `model/categories.joblib` | GitHub (`model/`) |
| `model/origin_traffic.csv`, `model/dest_traffic.csv`, `model/route_table.csv` | GitHub (`model/`) |
| `model/airline_list.csv`, `model/airport_list.csv`, `model/airline_routes.csv` | GitHub (`model/`) |
| `results/phase8_risk_levels.csv` | GitHub |

**Using the model in the website** (Python, with `src/` on the path):
```python
import predict
files = predict.load_model_files("model")
predict.predict_flight(files, "AA", "JFK", "LAX", "2026-12-23", 1900)
# {'expected_delay_min': 13.4, 'delay_probability': 0.504, 'risk': 'high', 'arrival_time': 2219}
```
The library versions must match [requirements.txt](../requirements.txt) (especially `lightgbm` and `scikit-learn`), otherwise the saved models may not load.

## Next phase
**Phase 9 — Presentation charts:** the most important figures from the saved results (delay distribution, correlation matrix, delay by month, model comparison, feature importance, confusion matrix, ROC curve).
