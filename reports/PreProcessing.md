# Phase 4 Report — Preprocessing

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase4_preprocessing.py` · **Shared module:** `src/preprocess.py`

## Goal
Models only accept numbers. This phase turns the columns into numeric inputs, in two versions:

| Version | For | Categories become |
|---|---|---|
| **One-hot** | Linear / Ridge, RandomForest | one 0/1 column per category |
| **Category type** | LightGBM | a pandas `category` column (LightGBM handles categories itself) |

**Rule:** everything that is *learned* (category lists, means, standard deviations) is learned **from train only** and then applied unchanged to validation (and later test). The test set was not loaded.

## Done criterion
> Both versions run on the full train and validation data, have the same columns in train and validation, contain no missing values, and an unseen category does not cause an error → **✅ Met**

---

## 4.1 — Data
`train.parquet` (4,040,848 × 11) and `val.parquet` (853,977 × 11) from Phase 3.

## 4.2 — Time: HHMM → minutes after midnight
`minutes = (HHMM // 100) × 60 + HHMM % 100`, then `% 1440`.

| Input | Output | Why |
|---|---|---|
| 1435 | 875 | 14:35 |
| 1359 / 1400 | 839 / 840 | one minute apart is now 1, not 41 |
| 2400 | 0 | midnight (one flight in the data, Phase 1) |

New columns: `DEP_MIN`, `ARR_MIN` (0–1439).

## 4.3 — Month as a cycle
Given as a number, December (12) and January (1) look 11 apart, although they are neighbouring winter months. Placing the months on a circle fixes this:

`MONTH_SIN = sin(2π · month / 12)`, `MONTH_COS = cos(2π · month / 12)`

| Month | sin | cos | |
|---|---|---|---|
| 12 (Dec) | 0.00 | 1.00 | |
| 1 (Jan) | 0.50 | 0.87 | close to December ✅ |
| 6 (Jun) | 0.00 | −1.00 | opposite December (summer vs. winter) |

These two columns are optional; Phase 5 compares models **without month** and **with cyclical month**. Using the month is possible because the day-based split (Phase 3) puts all months in train.

These steps only look at the value in each row, so applying them separately to train and validation cannot leak information.

## 4.4 — One-hot version (Linear / Ridge, RandomForest)
| Column group | Columns | Treatment |
|---|---|---|
| Categorical | `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `DAY_OF_WEEK` | `OneHotEncoder(handle_unknown="ignore")` |
| Numeric | `DEP_MIN`, `ARR_MIN`, `SCHEDULED_TIME`, `DISTANCE` | `StandardScaler` |
| Month (optional) | `MONTH_SIN`, `MONTH_COS` | `StandardScaler` |

- `DAY_OF_WEEK` is treated as a category: 1–7 has no numeric meaning (Sunday is not 7 × Monday).
- `handle_unknown="ignore"`: a category not seen in train becomes all zeros instead of an error (important for the website).
- `StandardScaler`: `(value − train mean) / train std`, so distance (31–4,983) and duration (18–718) get similar ranges; needed for Linear/Ridge.
- Fitted on train with `fit_transform`, applied to validation with `transform`.

| Check | Result |
|---|---|
| `X_train` | 4,040,848 × **671** |
| `X_val` | 853,977 × **671** (same columns) ✅ |
| Missing values | 0 / 0 ✅ |
| Columns without month | **669** (exactly 2 fewer) ✅ |

**Where the 671 columns come from:** 14 airlines + 322 origin airports + 322 destination airports + 7 weekdays + 4 numeric + 2 month = 671.

The result is a **sparse matrix**: each row has only 10 non-zero values out of 671, and only those are stored. Stored as a normal table, 4 million × 671 values would need more than 20 GB.

## 4.5 — Category-type version (LightGBM)
- `learn_categories(train)`: the list of categories in each categorical column, **learned from train only**.
- `make_lgbm_table(...)`: selects the columns and converts the categorical ones with `pd.Categorical(..., categories=train list)`. A value that is not in the train list becomes **NaN**, which LightGBM treats as "unknown" without an error. Using the same list everywhere also keeps the internal category numbers identical in train, validation and test.

| Check | Result |
|---|---|
| `lgbm_train` / `lgbm_val` | 4,040,848 × **10** / 853,977 × **10** ✅ |
| Types | 4 × `category`, 6 × numeric ✅ |
| Unseen categories in validation | 0 in every column ✅ |
| Fake airport `XYZ` | becomes `NaN`, no error ✅ (322 known airports) |

10 columns instead of 671 makes LightGBM much faster, and it can group similar categories (e.g. airports with similar delays) by itself.

## 4.6 — Shared module `preprocess.py`
All functions are written to one file with `%%writefile` (a Jupyter command that writes a cell to a file instead of running it):

| Function | Purpose |
|---|---|
| `hhmm_to_minutes` | HHMM → minutes |
| `add_features` | adds `DEP_MIN`, `ARR_MIN`, `MONTH_SIN`, `MONTH_COS` |
| `make_onehot_preprocessor(use_month)` | one-hot + scaling (Linear/Ridge, RandomForest) |
| `learn_categories(train)` | category lists from train |
| `make_lgbm_table(data, categories, use_month)` | category-type table (LightGBM) |

- `preprocess.py` is saved as an output of this Kaggle notebook. Phases 5–7 attach this notebook as an input and `import preprocess`, so **every model uses exactly the same preprocessing**, and the file in GitHub is the same file that runs on Kaggle.
- Checked with `import preprocess` on 1,000 train rows: LightGBM table 10 columns ✅; one-hot 286 columns (fewer than 671 because 1,000 flights do not contain every airport — expected).

---

## Decisions
1. Features: 4 categorical + 4 numeric + 2 optional month columns.
2. Two input formats: one-hot (671 / 669 columns) and category type (10 / 8 columns).
3. All learned values come from train only.
4. Unseen categories never cause an error (one-hot → all zeros, LightGBM → NaN).
5. No missing-value filling is needed: Phase 1 showed no missing values in the features after filtering.

## Outputs
| File | Location |
|---|---|
| `src/preprocess.py` | GitHub (same file as the Kaggle output) |
| `src/phase4_preprocessing.py` | GitHub |
| `preprocess.py` | Kaggle notebook output; input for Phases 5–7 |

## Next phase
**Phase 5 — Regression model comparison:** median baseline, Linear/Ridge Regression, RandomForest and LightGBM, trained on the full train set and compared on validation (MAE and training time), each without month and with cyclical month. The test set is not used.
