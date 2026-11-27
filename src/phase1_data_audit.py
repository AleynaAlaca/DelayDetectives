"""
Phase 1 — Data Audit

Goal: get to know the data and find its problems before modeling.
No data is changed in this phase; we only look and measure.

Runs on Kaggle with two inputs attached (Add Input):
    - Phase 0 notebook output   -> flights.parquet
    - usdot/flight-delays       -> airlines.csv
Output (in /kaggle/working/results):
    phase1_late_by_month.png
"""

# %% 1.1 — Load the data
import os
import pandas as pd
import numpy as np

parquet_path = None
raw_dir = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "flights.parquet" in files:
        parquet_path = folder + "/flights.parquet"
    if "airlines.csv" in files:
        raw_dir = folder

print("Parquet file:", parquet_path)
print("Raw data folder:", raw_dir)

df = pd.read_parquet(parquet_path)
print("Rows:", df.shape[0], "| Columns:", df.shape[1])

# %% 1.2 — Which columns can be used? (only information known before the flight)
features = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH", "DAY_OF_WEEK",
            "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "SCHEDULED_TIME", "DISTANCE"]
target = "ARRIVAL_DELAY"

after_flight = ["DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
                "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME",
                "CANCELLED", "CANCELLATION_REASON", "DIVERTED",
                "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
                "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]

other = []
for col in df.columns:
    if col not in features and col not in after_flight and col != target:
        other.append(col)

print("Features (known before the flight):", len(features))
print("Target:", target)
print("Known only after the flight:", len(after_flight))
print("Other columns:", other)

# %% 1.3 — Numeric airport codes: which months?
numeric_origin = df["ORIGIN_AIRPORT"].str.isdigit()
numeric_dest = df["DESTINATION_AIRPORT"].str.isdigit()

print("Rows with a numeric origin code, by month:")
print(df[numeric_origin]["MONTH"].value_counts())

print("\nRows with a numeric destination code, by month:")
print(df[numeric_dest]["MONTH"].value_counts())

print("\nAll October rows:", (df["MONTH"] == 10).sum())

# %% 1.3 — How many different codes in October?
october = df[df["MONTH"] == 10]

october_codes = set(october["ORIGIN_AIRPORT"]) | set(october["DESTINATION_AIRPORT"])
print("Different codes used in October:", len(october_codes))
print("Examples:", list(october_codes)[:5])

print(october[["MONTH", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]].head())

# %% 1.4 — Cancelled and diverted flights
n_cancelled = (df["CANCELLED"] == 1).sum()
n_diverted = (df["DIVERTED"] == 1).sum()

usable = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
removed = df[(df["CANCELLED"] == 1) | (df["DIVERTED"] == 1)]

print("Cancelled flights:", n_cancelled)
print("Diverted flights :", n_diverted)
print("Usable flights   :", len(usable))
print("\nMissing ARRIVAL_DELAY in usable flights :", usable["ARRIVAL_DELAY"].isna().sum())
print("Missing ARRIVAL_DELAY in removed flights:", removed["ARRIVAL_DELAY"].isna().sum(), "of", len(removed))

# %% 1.5 — Missing values in the features
print("Missing values in the 9 features (usable flights):")
print(usable[features].isna().sum())

print("\nMissing SCHEDULED_TIME in ALL flights:", df["SCHEDULED_TIME"].isna().sum())

# %% 1.6 — Distribution of the delay
y = usable["ARRIVAL_DELAY"]

print(y.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1))
print("\nEarly (< 0 min)      :", round((y < 0).mean() * 100, 1), "%")
print("Late more than 15 min:", round((y > 15).mean() * 100, 1), "%")
print("Late more than 3 hours:", round((y > 180).mean() * 100, 2), "%")

# %% 1.6 — One constant for every flight: median or mean?
median = y.median()
mean = y.mean()

print("Median:", median, "| Mean:", round(mean, 2))
print("MAE if we always predict the median:", round((y - median).abs().mean(), 2))
print("MAE if we always predict the mean  :", round((y - mean).abs().mean(), 2))

# %% 1.7 — Delay by month
monthly = usable.groupby("MONTH")["ARRIVAL_DELAY"].agg(["count", "mean", "median"])
monthly["late_15_percent"] = usable.groupby("MONTH")["ARRIVAL_DELAY"].apply(lambda s: (s > 15).mean() * 100)

print(monthly.round(1))

# %% 1.7 — Plot: share of flights more than 15 minutes late
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.bar(monthly.index, monthly["late_15_percent"], color="tab:blue")
plt.title("Flights more than 15 min late, by month")
plt.xlabel("Month")
plt.ylabel("% of flights")
plt.xticks(range(1, 13))
plt.tight_layout()

os.makedirs("/kaggle/working/results", exist_ok=True)
plt.savefig("/kaggle/working/results/phase1_late_by_month.png", dpi=120)
plt.show()

# %% 1.8 — Time columns (HHMM format)
dep = usable["SCHEDULED_DEPARTURE"]
arr = usable["SCHEDULED_ARRIVAL"]

print("SCHEDULED_DEPARTURE: min", dep.min(), "| max", dep.max())
print("SCHEDULED_ARRIVAL  : min", arr.min(), "| max", arr.max())
print("Minutes part above 59 (departure):", (dep % 100 > 59).sum())
print("Minutes part above 59 (arrival)  :", (arr % 100 > 59).sum())
print("Value 2400 in arrival:", (arr == 2400).sum())

# %% 1.8 — In how many months does each airline fly?
months_per_airline = usable.groupby("AIRLINE")["MONTH"].nunique()
print("Number of months each airline flies:")
print(months_per_airline.sort_values())

airlines = pd.read_csv(raw_dir + "/airlines.csv")
print(airlines[airlines["IATA_CODE"].isin(months_per_airline[months_per_airline < 12].index)])
