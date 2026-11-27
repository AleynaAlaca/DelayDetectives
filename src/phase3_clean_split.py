"""
Phase 3 — Cleaning, Filtering and Split

Goal: remove flights without a target, keep only the allowed columns,
and split the days of every month into train (70%), validation (15%) and test (15%).
All flights of one day stay in the same part. The test set is locked until the final phase.

Runs on Kaggle with one input attached (Add Input):
    - Phase 2 notebook output   -> flights_fixed.parquet
Outputs (in /kaggle/working):
    train.parquet, val.parquet, test.parquet   (test is not opened again until Phase 7)
    results/day_split.csv                      (which part each of the 365 days belongs to)
"""

# %% 3.1 — Load only the needed columns (columns known after the flight are never loaded)
import os
import pandas as pd
import numpy as np

fixed_path = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "flights_fixed.parquet" in files:
        fixed_path = folder + "/flights_fixed.parquet"

features = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH", "DAY_OF_WEEK",
            "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "SCHEDULED_TIME", "DISTANCE"]
target = "ARRIVAL_DELAY"
helper_cols = ["YEAR", "DAY", "CANCELLED", "DIVERTED"]

df = pd.read_parquet(fixed_path, columns=features + [target] + helper_cols)

print("File:", fixed_path)
print("Rows:", df.shape[0], "| Columns:", df.shape[1])

# %% 3.2 — Remove cancelled and diverted flights (result in a new variable)
clean = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
clean = clean.drop(columns=["CANCELLED", "DIVERTED"])

print("Rows before:", len(df))
print("Rows after :", len(clean))
print("Removed    :", len(df) - len(clean))
print("Missing target:", clean[target].isna().sum())

# %% 3.3 — Date column
clean["DATE"] = pd.to_datetime(clean[["YEAR", "MONTH", "DAY"]])
clean = clean.drop(columns=["YEAR", "DAY"])

print("First date:", clean["DATE"].min().date())
print("Last date :", clean["DATE"].max().date())
print("Number of days:", clean["DATE"].nunique())
print("Columns:", clean.columns.tolist())

# %% 3.4 — Split the days of every month: 70% train, 15% validation, 15% test
days = pd.DataFrame({"DATE": clean["DATE"].unique()})
days["MONTH"] = days["DATE"].dt.month

month_parts = []
for m in range(1, 13):
    month_days = days[days["MONTH"] == m].sample(frac=1, random_state=42)
    n = len(month_days)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    month_days["SPLIT"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    month_parts.append(month_days)

days = pd.concat(month_parts)

print(days.groupby(["MONTH", "SPLIT"]).size().unstack())
print("\nDays per part:")
print(days["SPLIT"].value_counts())

# %% 3.4 — Give every flight the part of its day
day_to_split = dict(zip(days["DATE"], days["SPLIT"]))
clean["SPLIT"] = clean["DATE"].map(day_to_split)

print(clean["SPLIT"].value_counts())
print("Flights without a split:", clean["SPLIT"].isna().sum())

# %% 3.5 — Check: every day is in only one part
parts_per_day = clean.groupby("DATE")["SPLIT"].nunique()
print("Days that appear in more than one part:", (parts_per_day > 1).sum())

# %% 3.5 — Check: the parts look alike
summary = clean.groupby("SPLIT")["ARRIVAL_DELAY"].agg(["count", "mean", "median"])
summary["late_15_percent"] = clean.groupby("SPLIT")["ARRIVAL_DELAY"].apply(lambda s: (s > 15).mean() * 100)
summary["months"] = clean.groupby("SPLIT")["MONTH"].nunique()

print(summary.round(2))

# %% 3.5 — Check: airlines/airports in validation and test that never appear in train
train = clean[clean["SPLIT"] == "train"]

for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
    train_values = set(train[col])
    unseen_val = (~clean[clean["SPLIT"] == "val"][col].isin(train_values)).sum()
    unseen_test = (~clean[clean["SPLIT"] == "test"][col].isin(train_values)).sum()
    print(col, "| unseen in val:", unseen_val, "| unseen in test:", unseen_test)

# %% 3.6 — Baselines (validation only; test stays locked)
val = clean[clean["SPLIT"] == "val"]
test = clean[clean["SPLIT"] == "test"]

train_median = train["ARRIVAL_DELAY"].median()
baseline_mae = (val["ARRIVAL_DELAY"] - train_median).abs().mean()

print("Train median:", train_median)
print("Baseline MAE on validation:", round(baseline_mae, 2))

always_on_time = (val["ARRIVAL_DELAY"] <= 15).mean() * 100
print("Accuracy on validation if we always say 'not delayed':", round(always_on_time, 1), "%")

# %% 3.7 — Save the three parts and the day list
os.makedirs("/kaggle/working/results", exist_ok=True)

train.drop(columns=["SPLIT"]).to_parquet("/kaggle/working/train.parquet", index=False)
val.drop(columns=["SPLIT"]).to_parquet("/kaggle/working/val.parquet", index=False)
test.drop(columns=["SPLIT"]).to_parquet("/kaggle/working/test.parquet", index=False)

days.sort_values("DATE").to_csv("/kaggle/working/results/day_split.csv", index=False)

print("Saved:", sorted(os.listdir("/kaggle/working")))
