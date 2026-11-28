# Phase 6b Report — Feature Engineering

**Status:** ✅ Done · **Date:** 2026-11-28 · **Code:** `src/phase6b_feature_engineering.py` · **Shared module:** `src/features.py`

## Goal
Phases 5 and 6 showed that changing the model or the loss no longer improves the results: the limit is the **information** in the features. This phase adds new **pre-departure** information and checks on validation whether it helps.

**Rules for a new feature:**
1. It must be known **before the flight**.
2. A **website** must be able to compute it from what a user enters (airline, airports, date, time).
3. Anything learned from data is learned **from train only**.
4. The test set is not used.

## Done criterion
> New features compared with the old ones for both tasks on validation; a feature set chosen; the chosen features packaged in a tested module → **✅ Met**

---

## New features
| Feature | Meaning | How it is computed | Uses the delay? |
|---|---|---|---|
| `ROUTE` | Origin–destination pair, e.g. `ATL_ORD` (4,641 routes in train) | Joins the two airport codes | No |
| `ORIGIN_HOUR_FLIGHTS` | Average scheduled departures per day at that airport in that hour | Counted from the train schedule ÷ 258 train days | No |
| `DEST_HOUR_FLIGHTS` | Average scheduled arrivals per day at that airport in that hour | Same | No |
| `DAYS_TO_HOLIDAY` | Days to the nearest US federal holiday (0–49) | `USFederalHolidayCalendar` | No |
| `ROUTE_DELAY_RATE` | Share of train flights on the route that were > 15 min late | Target encoding with smoothing | **Yes** |
| `ROUTE_MEDIAN_DELAY` | Median delay on the route in train | Target encoding with smoothing | **Yes** |

**Airport traffic** — busiest airport-hours in train: all in Atlanta (ATL), up to **88 departures per day** at 19:00 (≈ one every 40 seconds). Only 23 / 27 validation flights fall in an airport-hour with no train flights (left empty; LightGBM handles missing values).

**Route history (target encoding)** — `rate = (late flights on route + 100 × global rate) / (flights on route + 100)`, i.e. 100 imaginary flights at the global rate (18.2%) are added to every route, so routes with few flights are pulled toward the global value. Highest smoothed rates:

| Route | Late rate | Median delay |
|---|---|---|
| LGA → LIT | 42.1% | +12 min |
| OAK → IAH | 38.0% | +8 min |
| ORD → ASE | 37.4% | +5 min |
| LGA → SDF | 36.9% | +7 min |
| DFW → OGG | 35.1% | +7 min |

Only 9 validation flights use a route never seen in train (they get the global values).
*Risk:* every train flight also contributes to its own route's rate (a small in-sample leak). Validation, whose delays are never used, is the real check.

**Holidays (2015):** Jan 1, Jan 19, Feb 16, May 25, **Jul 3** (observed for Jul 4, a Saturday), Sep 7, Oct 12, Nov 11, **Nov 26** (Thanksgiving), **Dec 25**.

---

## Comparison (validation)
Three feature versions, with the **same models and settings** as Phases 5 and 6, so only the features differ:

| Version | Features | Columns |
|---|---|---|
| A | Old features (Phases 5–6) | 10 |
| B | A + route, airport traffic, holiday | 14 |
| C | B + route history (target encoding) | 16 |

| Version | Regression MAE | Classification ROC-AUC | Log loss | Best F1 (threshold) |
|---|---|---|---|---|
| A: old | 17.980 | 0.6721 | 0.4358 | 0.3651 (0.18) |
| **B: + route, traffic, holiday** | **17.877** | **0.6798** | 0.4355 | 0.3730 (0.17) |
| C: + route history | 17.912 | 0.6797 | 0.4349 | 0.3739 (0.18) |

- Version A reproduces Phases 5 and 6 exactly (17.98; 0.6721; 0.3651) — the comparison is set up correctly.
- **B improves both tasks:** MAE −0.10 min (total gain over the 18.80 baseline: 0.92 min), ROC-AUC +0.008, F1 +0.008.
- **C does not improve on B:** worse for regression (17.912 vs 17.877) and only a negligible difference for classification (F1 +0.0009, log loss −0.0006, ROC-AUC identical). Possible reasons: the route effect is already learned from the `ROUTE` category, and the small in-sample leak makes the model trust the encoding more than it deserves.
- Testing B and C separately mattered: adding all features at once would have hidden that the route history does not help.

## Decision: version B
| Reason | |
|---|---|
| Improves both tasks | MAE −0.10, ROC-AUC +0.008, F1 +0.008 |
| No feature uses the delay | No target-encoding leak |
| Simpler for the website | Only two small traffic tables and the holiday calendar |
| C brings nothing reliable | Worse for regression, no measurable gain for classification |

The gain is **small but consistent**. Schedule and calendar information helps a little; most of the delay is caused by things that are not known before the flight (weather, the aircraft's previous flight, operational problems).

---

## `features.py` and the final models
- All new features are computed by one module, `features.py`, which Phases 7–8 import (same idea as `preprocess.py` in Phase 4): `add_hour_and_route`, `learn_traffic_tables` (train only), `days_to_holiday` (works for **any year**, so the website can handle future dates), `add_new_features`, `learn_categories`, `make_table` (14 columns in a fixed order; unseen categories become NaN).
- **Check:** starting again from the raw files and using only the module, the resulting train and validation tables are **identical** to the step-by-step version B tables (`equals()` → `True` / `True`).
- Final models trained with version B reproduce the comparison: **MAE 17.877**, **ROC-AUC 0.6798**.

### Feature importance (gain, %)
| Feature | Regression | Classification |
|---|---|---|
| `ROUTE` | **47.8** | **34.2** |
| `ORIGIN_AIRPORT` | 10.6 | 10.8 |
| `DESTINATION_AIRPORT` | 7.8 | 7.7 |
| `DAYS_TO_HOLIDAY` | 7.2 | 11.6 |
| `MONTH_SIN` | 5.5 | 7.2 |
| `AIRLINE` | 5.4 | 5.6 |
| `DEP_MIN` | 4.7 | 10.3 |
| `DAY_OF_WEEK` | 4.6 | 6.1 |
| `MONTH_COS` | 2.4 | 3.2 |
| `SCHEDULED_TIME` | 2.0 | 0.2 |
| `ARR_MIN` | 1.2 | 2.5 |
| `DISTANCE` | 0.4 | 0.1 |
| `ORIGIN_HOUR_FLIGHTS` | 0.2 | 0.3 |
| `DEST_HOUR_FLIGHTS` | 0.2 | 0.3 |

- **`ROUTE` becomes the main feature** and takes over much of what the two airport columns carried in Phase 5 (origin 30.9% → 10.6%, destination 27.0% → 7.8%). Gain importance favours columns with many values (4,641 routes), so its share is probably somewhat inflated; the ranking is more reliable than the exact percentages.
- **`DAYS_TO_HOLIDAY` matters** (7–12%). It may also act partly as a position in the calendar (e.g. "3 days before Christmas" points to a specific week); the validation gain shows that it carries useful information.
- **Airport traffic is almost unused** (0.2–0.3%): the model already knows the airport and the departure hour, and their combination already says whether that airport is busy at that hour. The gain of version B therefore comes mainly from the **route** and the **holiday** features. Removing the traffic features was not tested; given their tiny share, little change is expected, and they are kept because they cost almost nothing.
- **Departure time matters more for classification** (10.3% vs 4.7%): whether a flight passes the 15-minute mark depends strongly on the time of day, as delays build up during the day.

## Outputs
| File | Location |
|---|---|
| `src/features.py`, `src/phase6b_feature_engineering.py` | GitHub |
| `results/phase6b_regression_features.csv`, `results/phase6b_classification_features.csv`, `results/phase6b_feature_importance.csv` | GitHub |
| `results/origin_traffic.csv`, `results/dest_traffic.csv` | GitHub (small; also needed by the website) |
| `results/lgbm_regression_v2.joblib`, `results/lgbm_classifier_v2.joblib`, `results/categories_v2.joblib`, `features.py` | Kaggle output (inputs for Phases 7–8) |

## Next phase
**Phase 7 — Final test evaluation:** with the chosen settings and version B features, retrain both models on train + validation (traffic tables and categories relearned from train + validation), open the test set once, and report regression MAE and classification metrics (threshold chosen on validation) against the baselines.
