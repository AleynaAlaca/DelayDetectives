"""
Phase 4 — Preprocessing

Goal: turn the columns into numbers a model can use.
Two versions are prepared: one-hot encoding (Linear/Ridge, RandomForest)
and pandas category types (LightGBM). Everything is learned from train only.
The test set is not loaded.

Runs on Kaggle with one input attached (Add Input):
    - Phase 3 notebook output   -> train.parquet, val.parquet
Output (in /kaggle/working):
    preprocess.py   -> imported by Phases 5-7 (same file as src/preprocess.py)
"""

# %% 4.1 — Load train and validation (not test)
import os
import numpy as np
import pandas as pd

data_dir = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "train.parquet" in files:
        data_dir = folder

train = pd.read_parquet(data_dir + "/train.parquet")
val = pd.read_parquet(data_dir + "/val.parquet")

print("Data folder:", data_dir)
print("train:", train.shape)
print("val  :", val.shape)

# %% 4.2 — HHMM -> minutes after midnight
def hhmm_to_minutes(hhmm):
    hours = hhmm // 100
    minutes = hhmm % 100
    return (hours * 60 + minutes) % 1440

print("1435 ->", hhmm_to_minutes(1435))
print("1359 ->", hhmm_to_minutes(1359), "| 1400 ->", hhmm_to_minutes(1400))
print("2400 ->", hhmm_to_minutes(2400))

# %% 4.3 — Add time in minutes and the month as a cycle
def add_features(data):
    data = data.copy()
    data["DEP_MIN"] = hhmm_to_minutes(data["SCHEDULED_DEPARTURE"])
    data["ARR_MIN"] = hhmm_to_minutes(data["SCHEDULED_ARRIVAL"])
    data["MONTH_SIN"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_COS"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    return data

train = add_features(train)
val = add_features(val)

months = pd.DataFrame({"MONTH": range(1, 13)})
months["MONTH_SIN"] = np.sin(2 * np.pi * months["MONTH"] / 12).round(2)
months["MONTH_COS"] = np.cos(2 * np.pi * months["MONTH"] / 12).round(2)
print(months)

print(train[["SCHEDULED_DEPARTURE", "DEP_MIN", "SCHEDULED_ARRIVAL", "ARR_MIN"]].head())

# %% 4.4 — One-hot preparation (Linear/Ridge, RandomForest)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK"]
numeric_cols = ["DEP_MIN", "ARR_MIN", "SCHEDULED_TIME", "DISTANCE"]
month_cols = ["MONTH_SIN", "MONTH_COS"]

def make_onehot_preprocessor(use_month):
    if use_month:
        number_cols = numeric_cols + month_cols
    else:
        number_cols = numeric_cols
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numbers", StandardScaler(), number_cols),
    ])

# %% 4.4 — Learn from train, apply to validation
prep = make_onehot_preprocessor(use_month=True)

X_train = prep.fit_transform(train)
X_val = prep.transform(val)

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("NaN in X_train:", np.isnan(X_train.data).sum())
print("NaN in X_val  :", np.isnan(X_val.data).sum())

prep_no_month = make_onehot_preprocessor(use_month=False)
X_train_no_month = prep_no_month.fit_transform(train)
print("Columns without month:", X_train_no_month.shape[1])

# %% 4.5 — Category types (LightGBM)
def learn_categories(train_data):
    categories = {}
    for col in categorical_cols:
        categories[col] = sorted(train_data[col].unique())
    return categories

def make_lgbm_table(data, categories, use_month):
    if use_month:
        cols = categorical_cols + numeric_cols + month_cols
    else:
        cols = categorical_cols + numeric_cols
    table = data[cols].copy()
    for col in categorical_cols:
        table[col] = pd.Categorical(table[col], categories=categories[col])
    return table

# %% 4.5 — Apply and test (unseen categories become NaN)
categories = learn_categories(train)

lgbm_train = make_lgbm_table(train, categories, use_month=True)
lgbm_val = make_lgbm_table(val, categories, use_month=True)

print("lgbm_train:", lgbm_train.shape)
print("lgbm_val  :", lgbm_val.shape)
print(lgbm_train.dtypes)
print("\nUnseen categories in val (became NaN):")
print(lgbm_val[categorical_cols].isna().sum())

test_row = val.head(1).copy()
test_row["ORIGIN_AIRPORT"] = "XYZ"
print("\nFake airport 'XYZ' becomes:", make_lgbm_table(test_row, categories, use_month=True)["ORIGIN_AIRPORT"].values)

# %% 4.6 — Write preprocess.py (in the notebook this cell starts with: %%writefile /kaggle/working/preprocess.py)
# The content of that cell is identical to src/preprocess.py.

# %% 4.6 — Check that preprocess.py works
import preprocess

check_train = preprocess.add_features(train.head(1000))
check_prep = preprocess.make_onehot_preprocessor(use_month=True)
print("One-hot columns:", check_prep.fit_transform(check_train).shape[1])

check_categories = preprocess.learn_categories(check_train)
print("LightGBM columns:", preprocess.make_lgbm_table(check_train, check_categories, use_month=True).shape[1])
