# Phase 1 Report — Data Audit

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase1_data_audit.py`

## Goal
Get to know the data and find its problems before any modeling. No data was changed in this phase; everything was measured on `flights.parquet` from Phase 0. The test months (November–December) were **not** looked at.

---

## 1.1 — Data
`flights.parquet`: **5,819,079 rows × 31 columns**, one row per flight in 2015.

## 1.2 — Which columns can be used?
**Rule:** the model may only use information that is known **before the flight**. A website user asks for a prediction before flying, so anything learned during or after the flight cannot be an input. Using such a column is called **data leakage**: the model looks excellent in training but cannot work in practice.

| Group | Count | Columns |
|---|---|---|
| **Features** (known before the flight) | 9 | `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `MONTH`, `DAY_OF_WEEK`, `SCHEDULED_DEPARTURE`, `SCHEDULED_ARRIVAL`, `SCHEDULED_TIME`, `DISTANCE` |
| **Target** | 1 | `ARRIVAL_DELAY` (minutes; negative = early) |
| **Known only after the flight** (never used as features) | 17 | `DEPARTURE_TIME`, `DEPARTURE_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME`, `CANCELLED`, `CANCELLATION_REASON`, `DIVERTED`, `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` |
| **Other** (not used) | 4 | `YEAR` (always 2015), `DAY` (only to build the date), `FLIGHT_NUMBER`, `TAIL_NUMBER` (a website user would not know the aircraft) |

9 + 1 + 17 + 4 = 31: every column is accounted for. `CANCELLED` and `DIVERTED` are used only to filter rows, not as features.

## 1.3 — October airport codes
Airports are normally written as three-letter IATA codes (`ATL` = Atlanta). In this dataset October uses numeric BTS airport IDs instead (e.g. `14747`).

| Check | Result |
|---|---|
| Months with numeric origin codes | **October only** — 486,165 rows |
| Months with numeric destination codes | **October only** — 486,165 rows |
| All October rows | 486,165 → **every** October flight is affected, in both columns |
| Different numeric codes | **307** |

**Why it matters:** October is half of the validation set. The model is trained on January–August, where airports appear as letter codes. Without a fix, everything learned about `ATL` would be useless for `10397`. **Phase 2 maps the 307 codes to IATA codes.**

## 1.4 — Cancelled and diverted flights
| | Flights |
|---|---|
| Cancelled | 89,884 |
| Diverted | 15,187 |
| **Usable** (neither) | **5,714,008** |
| Missing `ARRIVAL_DELAY` in usable flights | 0 |
| Missing `ARRIVAL_DELAY` in removed flights | 105,071 of 105,071 |

Cancelled flights never arrived and diverted flights landed elsewhere, so they have no arrival delay. Every removed flight has an empty target and every usable flight has one, so removing them (about 1.8% of all flights) is the right choice. **Phase 3 removes them.**

## 1.5 — Missing values in the features
- In the 5,714,008 usable flights, **none of the 9 features has a missing value.**
- `SCHEDULED_TIME` has 6 missing values in the full data, but all of them are cancelled or diverted flights, so they disappear with the filter.

## 1.6 — Distribution of the delay (usable flights)
| Statistic | Minutes |
|---|---|
| Minimum | −87 |
| 5% / 25% | −25 / −13 |
| **Median** | **−5** |
| 75% / 95% / 99% | 8 / 66 / 167 |
| Maximum | 1,971 (≈ 33 hours) |
| **Mean** | **4.4** |
| Standard deviation | 39.3 |

| Share of flights | |
|---|---|
| Early (< 0 min) | 61.3% |
| More than 15 min late (official US "delayed") | **17.9%** |
| More than 3 hours late | 0.83% |

- Most flights arrive **early**; airlines build slack into their schedules.
- The distribution is **right-skewed**: a few very large delays pull the mean about 9 minutes above the median.
- **For classification (Phase 6):** only ~18% of flights are "delayed". A model that always says "not delayed" would already reach ~82% accuracy, so accuracy alone is misleading; recall, precision, F1 and ROC-AUC are needed.

### Baseline
The baseline predicts the same number for every flight: the **training median** (January–August).

| Constant prediction (from train) | Validation MAE (Sep–Oct) |
|---|---|
| **Train median = −4.0 min (baseline)** | **15.54 min** |
| Train mean = +5.88 min | 19.72 min |

Train: 3,836,893 flights · Validation: 945,031 flights.

- **15.54 minutes is the number every regression model has to beat.**
- Predicting the mean is 4.2 minutes worse than predicting the median. MAE is smallest at the median, but Linear Regression and RandomForest aim for the mean by default (they minimise squared error). This is why Phase 5 also trains LightGBM with an absolute-error objective.

## 1.7 — Delay by month (months 1–10)
![Flights more than 15 min late by month](../results/phase1_late_by_month.png)

| Month | Flights | Mean | Median | > 15 min late | Baseline MAE | Split |
|---|---|---|---|---|---|---|
| 1 | 457,013 | 5.8 | −4 | 20.2% | 20.1 | train |
| 2 | 407,663 | 8.3 | −2 | 22.5% | 21.8 | train |
| 3 | 492,138 | 4.9 | −4 | 18.6% | 19.2 | train |
| 4 | 479,251 | 3.2 | −5 | 16.4% | 17.8 | train |
| 5 | 489,641 | 4.5 | −5 | 17.6% | 19.2 | train |
| 6 | 492,847 | 9.6 | −3 | 22.7% | 22.5 | train |
| 7 | 514,384 | 6.4 | −4 | 20.2% | 20.2 | train |
| 8 | 503,956 | 4.6 | −5 | 18.0% | 19.2 | train |
| 9 | 462,153 | −0.8 | −7 | 12.4% | 15.7 | validation |
| 10 | 482,878 | −0.8 | −7 | 11.9% | 15.4 | validation |

- **September and October are the calmest months** (about 12% delayed vs. 16–23% in training months). The hardest months are June (summer storms, peak traffic) and February (winter).
- Consequences:
  1. **Validation results will look somewhat optimistic.** The test months (Nov–Dec: holidays, winter) are likely harder, so test MAE is expected to be higher than validation MAE. This would come from the season, not from an error.
  2. **`MONTH` is a problem as a feature:** training contains only months 1–8, so the model never sees September–December. Phase 5 compares "no month" with a **cyclical month** encoding (sin/cos).
  3. **For classification:** the delayed share is ~20% in training but ~12% in validation, so a classifier may over-predict delays on validation. Phase 6 takes this into account.
- Test months were excluded from this table on purpose: looking at the test target now could influence later decisions.

## 1.8 — Other checks
**Time columns (HHMM format, e.g. 1435 = 14:35):**

| Check | Result |
|---|---|
| `SCHEDULED_DEPARTURE` range | 1 – 2359 |
| `SCHEDULED_ARRIVAL` range | 1 – 2400 |
| Minutes part above 59 | 0 (both columns) ✅ |
| Value `2400` (midnight) | 1 row (arrival) |

HHMM cannot be given to a model as it is: 13:59 → 14:00 is 1 minute, but `1400 − 1359 = 41`. **Phase 4 converts both columns to minutes after midnight** (`2400` → 0).

**Airlines:** 14 airlines in training, 13 in validation. **US Airways (`US`)** appears only in training: it merged with American Airlines in 2015. Its flights add little for predictions after the merger, and the website should not offer it as an option.

---

## Summary: problems found and where they are handled
| Problem | Size | Handled in |
|---|---|---|
| Numeric airport codes in October | 486,165 flights, 307 codes | Phase 2 |
| Cancelled / diverted flights (no target) | 105,071 flights | Phase 3 |
| Columns known only after the flight | 17 columns | Never loaded as features (Phase 3) |
| HHMM time format | 2 columns | Phase 4 |
| Validation period calmer than training | ~12% vs ~20% delayed | Reported; considered in Phases 5–7 |
| `MONTH` values in validation/test never seen in training | months 9–12 | Phase 5 experiment (no month vs cyclical month) |
| Mean ≠ median (right-skewed target) | 4.4 vs −5 min | Phase 5 (absolute-error objective for LightGBM) |
| Class imbalance for "> 15 min late" | ~18% | Phase 6 (metrics beyond accuracy) |

## Outputs
| File | Location |
|---|---|
| `src/phase1_data_audit.py` | GitHub |
| `results/phase1_late_by_month.png` | GitHub (download from the Kaggle notebook output) |

## Next phase
**Phase 2 — October Airport Code Fix:** map the 307 numeric October codes to three-letter IATA codes without losing any row, and check the result.
