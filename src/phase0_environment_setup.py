"""
Phase 0 — Environment & Setup

Goal: check that the data is available, record library versions,
and convert flights.csv to Parquet for fast reading in later phases.

Runs on Kaggle with the usdot/flight-delays dataset attached (Add Input).
Output (in /kaggle/working):
    flights.parquet   -> input for Phase 1 (not pushed to GitHub)
"""

# %% 0.1 — Find the data files
import os

raw_dir = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "flights.csv" in files:
        raw_dir = folder

print("Data folder:", raw_dir)

for name in sorted(os.listdir(raw_dir)):
    size_mb = os.path.getsize(raw_dir + "/" + name) / 1e6
    print(name, round(size_mb, 1), "MB")

# %% 0.2 — Library versions (copied into requirements.txt)
import platform
import pandas as pd
import numpy as np
import sklearn
import lightgbm
import joblib
import matplotlib
import pyarrow

print("python:", platform.python_version())
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("lightgbm:", lightgbm.__version__)
print("joblib:", joblib.__version__)
print("matplotlib:", matplotlib.__version__)
print("pyarrow:", pyarrow.__version__)

# %% 0.3 — Read the CSV and check its size
# Airport columns are read as text: in October they contain numbers instead of letters
df = pd.read_csv(raw_dir + "/flights.csv",
                 dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str},
                 low_memory=False)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())

# %% 0.4 — Save as Parquet
df.to_parquet("/kaggle/working/flights.parquet", index=False)

size_mb = os.path.getsize("/kaggle/working/flights.parquet") / 1e6
print("Parquet size:", round(size_mb, 1), "MB")

# %% 0.5 — Check that the Parquet file is identical to the CSV
check = pd.read_parquet("/kaggle/working/flights.parquet")

print("Rows:", check.shape[0], "| Columns:", check.shape[1])
print("Same as CSV:", check.equals(df))
