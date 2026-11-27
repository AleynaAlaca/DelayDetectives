# Phase 2 Report — October Airport Code Fix

**Status:** ✅ Done · **Date:** 2026-11-27 · **Code:** `src/phase2_airport_fix.py` · **Mapping:** [`results/airport_id_map.csv`](../results/airport_id_map.csv)

## Goal
In October 2015 all 486,165 flights store the origin and destination airports as numeric BTS airport IDs (e.g. `14747`) instead of three-letter IATA codes (e.g. `SEA`). Phase 1 showed this affects half of the validation set. This phase replaces the 307 numeric codes with IATA codes **without losing any row**.

## The difficulty
`airports.csv` contains IATA codes but **no** numeric IDs, so the answer is not in the dataset. It was found in two independent ways:

| | Method A — from the data | Method B — BTS tables |
|---|---|---|
| Source | 2015 flight schedules in September and November | Official BTS code tables `L_AIRPORT_ID` and `L_AIRPORT` (downloaded) |
| Idea | Airlines fly the same weekly schedule. An October flight with the same schedule as a September/November flight flies the same route, so its numeric codes must be that route's IATA codes. | `L_AIRPORT_ID` gives *numeric code → airport name*, `L_AIRPORT` gives *IATA code → airport name*. Joining on the name gives *numeric code → IATA code*. |

If two independent sources give the **same answer**, the mapping can be trusted; and each method can fill the other's gaps.

---

## Method A — matching schedules (2.2–2.3)
- A schedule is identified by 6 columns (the **key**): `AIRLINE, FLIGHT_NUMBER, DAY_OF_WEEK, SCHEDULED_DEPARTURE, SCHEDULED_TIME, DISTANCE`.
- Example: AA flight 1230, Thursday, 00:05, 237 min, 1,660 mi flies `SEA → DFW` in September and `14747 → 11298` in October → `14747 = SEA`, `11298 = DFW`.

| Step | Result |
|---|---|
| Distinct schedules in September + November | 377,733 |
| Distinct schedules in October | 157,540 |
| Matched schedules (inner merge on the key) | 102,956 |
| Different (numeric code, IATA) pairs | 286 |
| Codes matched to more than one IATA code | **0** |
| IATA codes matched to more than one numeric code | **0** |
| **Codes found** | **286 / 307** |
| Codes not found | 21, used by 3,043 October flights (0.63%) |

Across 102,956 matched schedules, no code was ever linked to two different airports. The 21 missing codes belong to small airports whose October schedules do not repeat exactly in September or November.

## Method B — BTS tables (2.4)
- Downloaded with `pd.read_csv(url)` (Kaggle **Internet On**).
- `encoding="latin-1"`: the files use an older text encoding; without it, accented letters in some Puerto Rico airport names cause a `UnicodeDecodeError`.
- `dtype=str`: codes are read as text, like the codes in our data, so the join works.

| Step | Result |
|---|---|
| `L_AIRPORT_ID` rows | 6,918 |
| `L_AIRPORT` rows | 6,942 |
| Our 307 codes found after joining on the airport name | 307, but in **311** rows |

Four codes had two candidates with the same airport name:

| Code | Candidates | Kept |
|---|---|---|
| 10170 (Kodiak, AK) | ADQ, T44 | ADQ |
| 10423 (Austin, TX) | AUS, BSM | AUS |
| 12173 (Honolulu, HI) | HNL, HIK | HNL |
| 16218 (Yuma, AZ) | YUM, NYL | YUM |

Keeping only codes that exist in `airports.csv` (the 322 codes used in our data) resolved all four, but **one code was lost: 307 → 306.**

**The lost code: `14027` (West Palm Beach, FL).** Today's BTS table lists this airport under a new name with the code `DJT`, which does not appear in the 2015 data. The 2015 data uses `PBI`.
**Lesson:** a reference table downloaded today describes the present, not 2015. Always check which period an external table belongs to.

## Comparison (2.5)
| | Codes |
|---|---|
| Found by both methods | 285 |
| — **same answer** | **285** |
| — **different answer** | **0** |
| Found only by A | 1 (`14027 → PBI`) |
| Found only by B | 21 (small airports) |
| **Total** | **307 / 307** ✅ |

Two completely independent sources agree on all 285 shared codes, and each one covers the other's gaps.

## Final mapping and fix (2.6)
- `IATA = answer of method A, otherwise answer of method B`; a `source` column records where each answer came from (`both` 285 · `B only` 21 · `A only` 1). Saved as `airport_id_map.csv`.
- All 307 codes exist in `airports.csv`; no IATA code is used for two numeric codes.
- Applied to October only, with a dictionary (`{"14747": "SEA", ...}`) and `.map()`. Example: the first October rows changed from `14747 → 11298` to `SEA → DFW`.

## Checks (2.7)
| Check | Result |
|---|---|
| Rows before / after | 5,819,079 / 5,819,079 ✅ |
| Missing airport codes created | 0 ✅ |
| Numeric codes left (whole year) | 0 ✅ |
| Codes not in `airports.csv` | 0 ✅ |

**Distance check.** A route has the same distance every month. A wrong code (e.g. another city's airport instead of `ORD`) would change the October distance by tens or hundreds of miles.

| | Routes |
|---|---|
| Routes flown both in October and in other months | 3,913 |
| Identical distance | 3,877 (99.1%) |
| Different distance | 36 |
| Largest difference | **2 miles** |

Differences of 1–2 miles are consistent with rounding or a small BTS distance update; the exact cause is unknown, but they cannot come from a wrong airport. The effect on the `DISTANCE` feature (~0.2%) is negligible.

**Performance note:** the distance check first keeps only the 4 needed columns and removes duplicates (5.8 M rows → a few tens of thousands), so it runs in seconds. The outputs are saved before this check, so a stuck kernel could not lose the corrected data.

---

## Decisions
1. Final mapping = method A, completed with method B; both agree on every shared code.
2. `14027 → PBI` (from the 2015 data), not `DJT` (today's BTS table).
3. The 1–2 mile distance differences are left as they are.
4. From now on, phases read `flights_fixed.parquet`.

## Outputs
| File | Location |
|---|---|
| `src/phase2_airport_fix.py` | GitHub |
| `results/airport_id_map.csv` (307 rows: ID, IATA, source) | GitHub (download from the Kaggle notebook output) |
| `flights_fixed.parquet` (5,819,079 × 31) | Kaggle notebook output; **not** in GitHub; input for Phase 3 |

## Next phase
**Phase 3 — Cleaning, Filtering and Split:** remove cancelled and diverted flights, keep only the allowed columns, split into train (Jan–Aug), validation (Sep–Oct) and test (Nov–Dec), and lock the test set.
