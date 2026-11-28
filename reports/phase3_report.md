# Phase 3 Report — Cleaning, Filtering and Split

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase3_clean_split.py` · **Day list:** [`results/day_split.csv`](../results/day_split.csv)

## Goal
Turn the corrected data from Phase 2 into the final modeling dataset: remove flights without a target, keep only the allowed columns, and split the data into train, validation and test in a way that gives **comparable parts** and **no leakage between days**.

## Done criterion
> Every flight belongs to exactly one part, no day is split between parts, the parts have similar delay distributions, and the test set is saved separately (locked) → **✅ Met**

---

## From a month-based split to a day-based split

**First attempt: split by month.** The project started with a month-based split: train January–August, validation September–October, test November–December. Phase 1 showed that this gives very different parts:

| | Month-based split (first attempt) |
|---|---|
| Flights > 15 min late in train (Jan–Aug) | ≈ 20% |
| Flights > 15 min late in validation (Sep–Oct) | ≈ 12% (calmest months of the year) |
| Months seen in training | 1–8 only |
| `MONTH` usable as a feature | No (validation and test months never appear in training) |
| Baseline MAE on validation | 15.54 min |

The low baseline error (15.54) came from the validation period being unusually calm, not from a better prediction. Validation results would therefore have looked **optimistic**, and the test months (November calm, December hard) would have behaved differently again.

**Current approach: split by day, stratified by month.** Inside every month the days are shuffled and divided into train (70%), validation (15%) and test (15%). All flights of one day stay in the same part.

| | Month-based split (first attempt) | **Day-based split (current)** |
|---|---|---|
| Flights > 15 min late: train / validation | ≈ 20% / ≈ 12% | **18.2% / 17.4%** |
| Months in every part | train 8, validation 2, test 2 | **12 in every part** |
| `MONTH` usable as a feature | No | **Yes** |
| Same day in two parts | No | **No** |
| Baseline MAE on validation | 15.54 min (optimistic) | **18.80 min** (realistic) |

**What improved:** the three parts are now comparable, so validation results are a much better estimate of test results, and the model can learn from all seasons. The baseline error is higher, but this number is more honest: it is close to the full-year value of 19.15 min from Phase 1. Model scores in Phases 5–7 will be compared against this baseline.

---

## Steps

### 3.1 — Load only the needed columns
- Input: `flights_fixed.parquet` (Phase 2 output).
- Only 14 of 31 columns were read: the 9 features, the target, `YEAR`/`DAY` (for the date) and `CANCELLED`/`DIVERTED` (for filtering).
- **Leakage protection:** columns known only after the flight (`DEPARTURE_DELAY`, delay causes, actual times, …) are never loaded, so they cannot reach a model by mistake.
- Loaded: 5,819,079 rows × 14 columns.

### 3.2 — Remove cancelled and diverted flights
| | Rows |
|---|---|
| Before | 5,819,079 |
| Removed (cancelled or diverted) | 105,071 |
| **After** | **5,714,008** |
| Missing target after filtering | 0 ✅ |

- `CANCELLED` and `DIVERTED` were dropped after filtering (always 0, and known only after the flight).
- The result is stored in a new variable (`clean`), so the cell can be re-run without errors.

### 3.3 — Date column
- `DATE` built from `YEAR`, `MONTH`, `DAY`: 2015-01-01 → 2015-12-31, **365 days**.
- `YEAR` and `DAY` dropped; `MONTH` kept (needed for the split and as a possible feature).
- Final columns (11): 9 features + `ARRIVAL_DELAY` + `DATE`.

### 3.4 — Day-based split, stratified by month
For each of the 12 months: shuffle the days (`random_state=42`), then give the first 70% to train, the next 15% to validation, the rest to test.

| Days in a month | Train | Validation | Test |
|---|---|---|---|
| 31 | 22 | 5 | 4 |
| 30 | 21 | 4 | 5 |
| 28 (February) | 20 | 4 | 4 |

| Part | Days | Flights | Share |
|---|---|---|---|
| Train | 258 | 4,040,848 | 70.7% |
| Validation | 55 | 853,977 | 14.9% |
| Test | 52 | 819,183 | 14.3% |
| **Total** | **365** | **5,714,008** | |

Flights without a part: 0. The full day list is saved in `results/day_split.csv`.

### 3.5 — Checks
| Check | Result |
|---|---|
| Days that appear in more than one part | **0** ✅ |
| Months in each part | 12 / 12 / 12 ✅ |
| Airlines unseen in train (val / test) | 0 / 0 ✅ |
| Origin airports unseen in train (val / test) | 0 / 0 ✅ |
| Destination airports unseen in train (val / test) | 0 / 0 ✅ |

**The parts look alike:**

| Part | Flights | Mean delay | Median delay | > 15 min late |
|---|---|---|---|---|
| Train | 4,040,848 | 4.58 | −5 | 18.17% |
| Validation | 853,977 | 4.08 | −5 | 17.37% |
| Test | 819,183 | 3.91 | −5 | 17.19% |

The remaining small differences come from which days were drawn; for example, a few stormy days may have landed in train. The test summary was looked at only to confirm that the split is balanced; no model or setting is chosen using test data.

### 3.6 — Baselines (validation)
| Task | Baseline | Validation result |
|---|---|---|
| **Regression** | Predict the training median (−5 min) for every flight | **MAE = 18.80 min** |
| **Classification** ("more than 15 min late?") | Always predict "not delayed" | **Accuracy = 82.6%** (recall = 0) |

These are the numbers the models in Phases 5 and 6 have to beat. The classification baseline shows why accuracy alone is not enough: a model that never detects a delay already reaches 82.6%.

### 3.7 — Saved files and the test lock
| File | Location |
|---|---|
| `train.parquet`, `val.parquet`, `test.parquet` | Kaggle notebook output (not in GitHub) |
| `results/day_split.csv` (365 rows: date, month, part) | GitHub |

**Test lock:** Phases 4–6 only load `train.parquet` and `val.parquet`. `test.parquet` is opened once, in Phase 7.

---

## Decisions
1. Modeling data: 5,714,008 flights, 9 features + target + date.
2. Split by day, stratified by month (70 / 15 / 15, seed 42), replacing the first month-based split.
3. `MONTH` can be used as a feature (all months are in train).
4. Baselines: regression MAE 18.80 min, classification accuracy 82.6%.

## Limitations
- Weather events that last several days can affect neighbouring days that ended up in different parts, so a small amount of information may still be shared between train and test. This effect is much smaller than in a split by individual flights.
- All data is from 2015; the split measures how well the model predicts unseen days of the same year, not of a later year.

## Next phase
**Phase 4 — Preprocessing:** convert HHMM times to minutes, encode the month as a cycle (sin/cos), and prepare the data in two forms: one-hot encoding for Linear/Ridge and RandomForest, and pandas category types for LightGBM.
