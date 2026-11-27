# Phase 0 Report — Environment & Setup

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase0_environment_setup.py`

## Goal
Make sure the data can be read on Kaggle, record the library versions so the project can be reproduced, and convert the large CSV file into the faster Parquet format.

## Done criterion
> `flights.parquet` exists and has 5,819,079 rows × 31 columns → **✅ Met**

---

## Steps

### 0.1 — Data files
The dataset ([2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays), USDOT) is attached to the Kaggle notebook with **Add Input**.

| File | Size | Content |
|---|---|---|
| `flights.csv` | 592.4 MB | One row per flight |
| `airlines.csv` | < 0.1 MB | Airline codes and names |
| `airports.csv` | < 0.1 MB | Airport codes, names, locations |

- The data folder is found with `os.walk` instead of a hard-coded path. Kaggle mounts the dataset at `/kaggle/input/datasets/usdot/flight-delays`, which differs from older examples (`/kaggle/input/flight-delays`); searching for `flights.csv` makes the code independent of this.
- Raw data is **not** stored in GitHub (`flights.csv` is larger than GitHub's 100 MB file limit). `.gitignore` excludes `*.csv` and `*.parquet`.

### 0.2 — Library versions
| Library | Version | Used for |
|---|---|---|
| Python | 3.13.18 | |
| pandas | 2.3.3 | Tables |
| numpy | 2.1.3 | Numbers |
| scikit-learn | 1.6.1 | Linear/Ridge, RandomForest, preprocessing, MAE |
| lightgbm | 4.6.0 | LightGBM model |
| joblib | 1.6.0 | Saving trained models (for the website) |
| matplotlib | 3.10.0 | Plots |
| pyarrow | 23.0.1 | Reading/writing Parquet |

These versions are pinned in `requirements.txt`. Library updates can change results; pinning them makes the project reproducible.

### 0.3 — Reading the CSV
- Result: **5,819,079 rows × 31 columns**.
- `ORIGIN_AIRPORT` and `DESTINATION_AIRPORT` are read as **text** (`dtype=str`). In October these columns contain numeric airport IDs (e.g. `10397`) instead of letter codes (e.g. `ATL`); without this setting pandas would read the columns with mixed types. Phase 2 fixes the October codes.
- `low_memory=False` lets pandas decide each column's type from the whole file, avoiding a `DtypeWarning`.
- First look at the data:
  - `ARRIVAL_DELAY` is the value to predict (minutes); **negative values mean the flight arrived early**.
  - Delay-cause columns (`WEATHER_DELAY`, `AIRLINE_DELAY`, …) are empty for most flights; they are only filled for flights delayed by 15 minutes or more.
  - Several columns (`DEPARTURE_DELAY`, `ARRIVAL_TIME`, delay causes, …) are only known **after** the flight. They cannot be used by the model, because a website user would not know them before the flight. Phase 1 lists them.

### 0.4 — Parquet
| | CSV | Parquet |
|---|---|---|
| File size | 592.4 MB | **141.5 MB** (about 4× smaller) |
| Keeps column types | No | Yes (airport columns stay text) |
| Can read selected columns only | No | Yes |

### 0.5 — Check
- The Parquet file was read back and compared with the CSV table using `equals()`: **identical** (5,819,079 × 31, `True`). No information was lost.

---

## Decisions
1. Raw data stays on Kaggle; GitHub holds code, reports and small results.
2. Data paths are found with `os.walk`, never typed by hand.
3. All later phases read `flights.parquet` instead of the CSV.

## Outputs
| File | Location |
|---|---|
| `flights.parquet` (141.5 MB) | Kaggle notebook output (input for Phase 1). **Not** in GitHub. |
| `src/phase0_environment_setup.py` | GitHub |
| `requirements.txt` | GitHub |

## Next phase
**Phase 1 — Data Audit:** get to know the data and find its problems: which columns are allowed, the numeric October airport codes, cancelled/diverted flights, missing values and the distribution of the delay.
