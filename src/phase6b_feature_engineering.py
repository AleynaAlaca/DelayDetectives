"""
Phase 6b — Feature Engineering

Goal: add pre-departure features that a website can also compute (route, airport traffic,
route history, distance to holidays) and check on validation whether they improve LightGBM.
Everything learned from data comes from train only. The test set is not used.

Runs on Kaggle with two inputs attached (Add Input):
    - Phase 3 notebook output   -> train.parquet, val.parquet
    - Phase 4 notebook output   -> preprocess.py
Outputs (in /kaggle/working):
    features.py                                         (same file as src/features.py)
    results/phase6b_regression_features.csv             versions A / B / C, regression
    results/phase6b_classification_features.csv         versions A / B / C, classification
    results/phase6b_feature_importance.csv              chosen version B, both models
    results/lgbm_regression_v2.joblib, lgbm_classifier_v2.joblib, categories_v2.joblib
    results/origin_traffic.csv, dest_traffic.csv        traffic tables learned from train
"""

# %% 6b.1 — Load, hour and route columns
import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

data_dir = None
code_dir = None
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "train.parquet" in files:
        data_dir = folder
    if "preprocess.py" in files:
        code_dir = folder

assert data_dir is not None, "train.parquet not found -> add Phase 3 notebook as Input"
assert code_dir is not None, "preprocess.py not found -> add Phase 4 notebook as Input"

sys.path.append(code_dir)
import preprocess

train = pd.read_parquet(data_dir + "/train.parquet")
val = pd.read_parquet(data_dir + "/val.parquet")

train = preprocess.add_features(train)
val = preprocess.add_features(val)

for data in [train, val]:
    data["DEP_HOUR"] = data["DEP_MIN"] // 60
    data["ARR_HOUR"] = data["ARR_MIN"] // 60
    data["ROUTE"] = data["ORIGIN_AIRPORT"] + "_" + data["DESTINATION_AIRPORT"]

os.makedirs("/kaggle/working/results", exist_ok=True)
print("train:", train.shape, "| val:", val.shape)
print("Different routes in train:", train["ROUTE"].nunique())
print(train[["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE", "DEP_MIN", "DEP_HOUR"]].head(3))

# %% 6b.2 — Airport traffic: average scheduled flights per day, per airport and hour (train only)
train_days = train["DATE"].nunique()

origin_traffic = train.groupby(["ORIGIN_AIRPORT", "DEP_HOUR"]).size() / train_days
origin_traffic = origin_traffic.rename("ORIGIN_HOUR_FLIGHTS").reset_index()

dest_traffic = train.groupby(["DESTINATION_AIRPORT", "ARR_HOUR"]).size() / train_days
dest_traffic = dest_traffic.rename("DEST_HOUR_FLIGHTS").reset_index()

train = train.merge(origin_traffic, on=["ORIGIN_AIRPORT", "DEP_HOUR"], how="left")
train = train.merge(dest_traffic, on=["DESTINATION_AIRPORT", "ARR_HOUR"], how="left")
val = val.merge(origin_traffic, on=["ORIGIN_AIRPORT", "DEP_HOUR"], how="left")
val = val.merge(dest_traffic, on=["DESTINATION_AIRPORT", "ARR_HOUR"], how="left")

print("Train days:", train_days)
print(origin_traffic.sort_values("ORIGIN_HOUR_FLIGHTS", ascending=False).head(5))
print("\nMissing in val:", val["ORIGIN_HOUR_FLIGHTS"].isna().sum(), val["DEST_HOUR_FLIGHTS"].isna().sum())

# %% 6b.3 — Route history (target encoding with smoothing, train only)
global_late_rate = (train["ARRIVAL_DELAY"] > 15).mean()
global_median = train["ARRIVAL_DELAY"].median()
m = 100

route_stats = train.groupby("ROUTE")["ARRIVAL_DELAY"].agg(["count", "median"])
route_stats["late_rate"] = train.groupby("ROUTE")["ARRIVAL_DELAY"].apply(lambda s: (s > 15).mean())

route_stats["ROUTE_DELAY_RATE"] = (route_stats["late_rate"] * route_stats["count"] + global_late_rate * m) / (route_stats["count"] + m)
route_stats["ROUTE_MEDIAN_DELAY"] = (route_stats["median"] * route_stats["count"] + global_median * m) / (route_stats["count"] + m)
route_stats = route_stats[["ROUTE_DELAY_RATE", "ROUTE_MEDIAN_DELAY"]].reset_index()

train = train.merge(route_stats, on="ROUTE", how="left")
val = val.merge(route_stats, on="ROUTE", how="left")

val["ROUTE_DELAY_RATE"] = val["ROUTE_DELAY_RATE"].fillna(global_late_rate)
val["ROUTE_MEDIAN_DELAY"] = val["ROUTE_MEDIAN_DELAY"].fillna(global_median)

print("Global late rate:", round(global_late_rate, 3), "| global median:", global_median)
print(route_stats.sort_values("ROUTE_DELAY_RATE", ascending=False).head(5))
print("\nValidation flights on routes not seen in train:", (~val["ROUTE"].isin(route_stats["ROUTE"])).sum())

# %% 6b.4 — Days to the nearest US federal holiday
from pandas.tseries.holiday import USFederalHolidayCalendar

holidays = USFederalHolidayCalendar().holidays(start="2014-12-01", end="2016-01-31")
print("Holidays in 2015:")
print([str(h.date()) for h in holidays if h.year == 2015])

days_to_holiday = {}
for day in pd.date_range("2015-01-01", "2015-12-31"):
    days_to_holiday[day] = int(np.abs((holidays - day).days).min())

for data in [train, val]:
    data["DAYS_TO_HOLIDAY"] = data["DATE"].map(days_to_holiday)

print("\nDAYS_TO_HOLIDAY range:", train["DAYS_TO_HOLIDAY"].min(), "-", train["DAYS_TO_HOLIDAY"].max())
print("Missing:", train["DAYS_TO_HOLIDAY"].isna().sum(), val["DAYS_TO_HOLIDAY"].isna().sum())

# %% 6b.5 — Three feature versions
old_cols = preprocess.categorical_cols + preprocess.numeric_cols + preprocess.month_cols
schedule_cols = ["ROUTE", "ORIGIN_HOUR_FLIGHTS", "DEST_HOUR_FLIGHTS", "DAYS_TO_HOLIDAY"]
history_cols = ["ROUTE_DELAY_RATE", "ROUTE_MEDIAN_DELAY"]

versions = {
    "A: old features": old_cols,
    "B: + route, traffic, holiday": old_cols + schedule_cols,
    "C: + route history": old_cols + schedule_cols + history_cols,
}

cat_cols = preprocess.categorical_cols + ["ROUTE"]
categories = {}
for col in cat_cols:
    categories[col] = sorted(train[col].unique())

def make_table(data, cols):
    table = data[cols].copy()
    for col in cat_cols:
        if col in cols:
            table[col] = pd.Categorical(table[col], categories=categories[col])
    return table

for name, cols in versions.items():
    print(name, "->", len(cols), "columns")

# %% 6b.5 — Regression (chosen model from Phase 5)
from sklearn.metrics import mean_absolute_error

y_train_reg = train["ARRIVAL_DELAY"]
y_val_reg = val["ARRIVAL_DELAY"]

reg_results = []
for name, cols in versions.items():
    model = LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.1,
                          num_leaves=127, random_state=42, verbose=-1)
    start = time.time()
    model.fit(make_table(train, cols), y_train_reg)
    seconds = time.time() - start
    pred = model.predict(make_table(val, cols))
    mae = mean_absolute_error(y_val_reg, pred)
    reg_results.append({"version": name, "val_mae": round(mae, 3), "seconds": round(seconds, 1)})
    print(name, "| MAE:", round(mae, 3), "| time:", round(seconds, 1), "s")

# %% 6b.5 — Classification (chosen model from Phase 6)
from sklearn.metrics import roc_auc_score, log_loss, f1_score

y_train_clf = (train["ARRIVAL_DELAY"] > 15).astype(int)
y_val_clf = (val["ARRIVAL_DELAY"] > 15).astype(int)

clf_results = []
for name, cols in versions.items():
    model = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                           num_leaves=127, random_state=42, verbose=-1)
    start = time.time()
    model.fit(make_table(train, cols), y_train_clf)
    seconds = time.time() - start
    proba = model.predict_proba(make_table(val, cols))[:, 1]

    best_f1 = 0
    best_t = 0
    for t in np.arange(0.05, 0.6, 0.01):
        f1 = f1_score(y_val_clf, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    clf_results.append({"version": name, "roc_auc": round(roc_auc_score(y_val_clf, proba), 4),
                        "log_loss": round(log_loss(y_val_clf, proba), 4),
                        "best_f1": round(best_f1, 4), "threshold": round(best_t, 2),
                        "seconds": round(seconds, 1)})
    print(clf_results[-1])

# %% 6b.6 — Save the comparison tables
pd.DataFrame(reg_results).to_csv("/kaggle/working/results/phase6b_regression_features.csv", index=False)
pd.DataFrame(clf_results).to_csv("/kaggle/working/results/phase6b_classification_features.csv", index=False)
print(pd.DataFrame(reg_results))
print(pd.DataFrame(clf_results))

# %% 6b.6 — Write features.py (in the notebook this cell starts with: %%writefile /kaggle/working/features.py)
# The content of that cell is identical to src/features.py.

# %% 6b.6 — Check that features.py gives exactly the same table as the step-by-step version B
import features

train_b = preprocess.add_features(pd.read_parquet(data_dir + "/train.parquet"))
val_b = preprocess.add_features(pd.read_parquet(data_dir + "/val.parquet"))

train_b = features.add_hour_and_route(train_b)
origin_traffic, dest_traffic = features.learn_traffic_tables(train_b)

train_b = features.add_new_features(train_b, origin_traffic, dest_traffic)
val_b = features.add_new_features(val_b, origin_traffic, dest_traffic)

categories_b = features.learn_categories(train_b)
X_train_b = features.make_table(train_b, categories_b)
X_val_b = features.make_table(val_b, categories_b)

print("X_train_b:", X_train_b.shape, "| X_val_b:", X_val_b.shape)
print("Same as the step-by-step version (train):", X_train_b.equals(make_table(train, versions["B: + route, traffic, holiday"])))
print("Same as the step-by-step version (val)  :", X_val_b.equals(make_table(val, versions["B: + route, traffic, holiday"])))

# %% 6b.6 — Final models with version B, saved for Phases 7-8
import joblib
from sklearn.metrics import mean_absolute_error, roc_auc_score

reg_model = LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.1,
                          num_leaves=127, random_state=42, verbose=-1)
reg_model.fit(X_train_b, train_b["ARRIVAL_DELAY"])
print("Regression MAE:", round(mean_absolute_error(val_b["ARRIVAL_DELAY"], reg_model.predict(X_val_b)), 3))

clf_model = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                           num_leaves=127, random_state=42, verbose=-1)
clf_model.fit(X_train_b, (train_b["ARRIVAL_DELAY"] > 15).astype(int))
proba_b = clf_model.predict_proba(X_val_b)[:, 1]
print("Classification ROC-AUC:", round(roc_auc_score((val_b["ARRIVAL_DELAY"] > 15).astype(int), proba_b), 4))

joblib.dump(reg_model, "/kaggle/working/results/lgbm_regression_v2.joblib")
joblib.dump(clf_model, "/kaggle/working/results/lgbm_classifier_v2.joblib")
joblib.dump(categories_b, "/kaggle/working/results/categories_v2.joblib")
origin_traffic.to_csv("/kaggle/working/results/origin_traffic.csv", index=False)
dest_traffic.to_csv("/kaggle/working/results/dest_traffic.csv", index=False)
print("Saved:", sorted(os.listdir("/kaggle/working/results")))

# %% 6b.6 — Feature importance (gain, %) of both final models
importance = pd.DataFrame({
    "feature": X_train_b.columns,
    "regression": reg_model.booster_.feature_importance(importance_type="gain"),
    "classification": clf_model.booster_.feature_importance(importance_type="gain"),
})
importance["regression"] = (importance["regression"] / importance["regression"].sum() * 100).round(1)
importance["classification"] = (importance["classification"] / importance["classification"].sum() * 100).round(1)
importance = importance.sort_values("regression", ascending=False)

importance.to_csv("/kaggle/working/results/phase6b_feature_importance.csv", index=False)
print(importance.to_string(index=False))
