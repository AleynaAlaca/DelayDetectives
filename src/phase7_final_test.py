"""
Phase 7 — Final Test Evaluation

Decisions fixed before opening the test set:
- final features: version B (route, airport traffic, distance to holidays)
- regression: LightGBM l1, 300 trees, lr 0.1, 127 leaves
- classification: LightGBM cross entropy, same settings, threshold 0.17 (version A: 0.18)
- models retrained on train + validation; version A is measured only as a reference
The test set is used once. Results are reported as they are, and no choice is changed afterwards.

Runs on Kaggle with three inputs attached (Add Input):
    - Phase 3 notebook output    -> train.parquet, val.parquet, test.parquet
    - Phase 4 notebook output    -> preprocess.py
    - Phase 6b notebook output   -> features.py
Outputs (in /kaggle/working/results):
    phase7_test_regression.csv, phase7_test_classification.csv
    phase7_error_by_month.csv, phase7_error_by_time.csv
    phase7_test_predictions.parquet (for the charts)
"""

# %% 7.1 — Load: train + validation for training, test for evaluation
import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier

data_dir = None
code_dirs = []
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "test.parquet" in files:
        data_dir = folder
    if "preprocess.py" in files or "features.py" in files:
        code_dirs.append(folder)

assert data_dir is not None, "test.parquet not found -> add Phase 3 notebook as Input"
assert len(code_dirs) == 2, "add Phase 4 (preprocess.py) and Phase 6b (features.py) as Input"

for folder in code_dirs:
    sys.path.append(folder)
import preprocess
import features

train = pd.read_parquet(data_dir + "/train.parquet")
val = pd.read_parquet(data_dir + "/val.parquet")
test = pd.read_parquet(data_dir + "/test.parquet")

full_train = pd.concat([train, val], ignore_index=True)
del train, val

full_train = preprocess.add_features(full_train)
test = preprocess.add_features(test)

os.makedirs("/kaggle/working/results", exist_ok=True)
print("train + validation:", full_train.shape, "| days:", full_train["DATE"].nunique())
print("test              :", test.shape, "| days:", test["DATE"].nunique())

# %% 7.2 — Baselines on test
from sklearn.metrics import (mean_absolute_error, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, log_loss)

y_full_reg = full_train["ARRIVAL_DELAY"]
y_test_reg = test["ARRIVAL_DELAY"]
y_full_clf = (y_full_reg > 15).astype(int)
y_test_clf = (y_test_reg > 15).astype(int)

reg_results = []
clf_results = []

full_median = y_full_reg.median()
baseline_mae = mean_absolute_error(y_test_reg, np.full(len(y_test_reg), full_median))
reg_results.append({"model": "Baseline (median)", "test_mae": baseline_mae})

baseline_proba = np.full(len(y_test_clf), y_full_clf.mean())
clf_results.append({"model": "Baseline (always 'not delayed')", "threshold": 0.5,
                    "accuracy": accuracy_score(y_test_clf, baseline_proba >= 0.5),
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "roc_auc": roc_auc_score(y_test_clf, baseline_proba),
                    "log_loss": log_loss(y_test_clf, baseline_proba)})

print("Median of train + validation:", full_median)
print("Baseline MAE on test:", round(baseline_mae, 2))
print("Delayed share in test:", round(y_test_clf.mean() * 100, 1), "%")
print(clf_results[0])

# %% 7.3 — Version B features (tables and categories learned from train + validation only)
full_b = features.add_hour_and_route(full_train)
origin_traffic, dest_traffic = features.learn_traffic_tables(full_b)

full_b = features.add_new_features(full_b, origin_traffic, dest_traffic)
test_b = features.add_new_features(test, origin_traffic, dest_traffic)

categories_b = features.learn_categories(full_b)
X_full_b = features.make_table(full_b, categories_b)
X_test_b = features.make_table(test_b, categories_b)

print("X_full_b:", X_full_b.shape, "| X_test_b:", X_test_b.shape)
print("Test flights with missing traffic:", X_test_b["ORIGIN_HOUR_FLIGHTS"].isna().sum(), X_test_b["DEST_HOUR_FLIGHTS"].isna().sum())
print("Test flights on routes never seen in training:", X_test_b["ROUTE"].isna().sum())

# %% 7.3 — Version B regression (final model)
reg_b = LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.1,
                      num_leaves=127, random_state=42, verbose=-1)

start = time.time()
reg_b.fit(X_full_b, full_b["ARRIVAL_DELAY"])
seconds = time.time() - start

pred_reg_b = reg_b.predict(X_test_b)
mae_b = mean_absolute_error(test_b["ARRIVAL_DELAY"], pred_reg_b)
reg_results.append({"model": "LightGBM l1 - B (final)", "test_mae": mae_b})

print("Test MAE (B):", round(mae_b, 3), "| baseline:", round(baseline_mae, 3), "| training time:", round(seconds, 1), "s")

# %% 7.3 — Version B classification (final model, threshold 0.17 from validation)
from sklearn.metrics import confusion_matrix

def classification_row(name, proba, threshold):
    pred = (proba >= threshold).astype(int)
    row = {"model": name, "threshold": threshold,
           "accuracy": accuracy_score(y_test_clf, pred),
           "precision": precision_score(y_test_clf, pred),
           "recall": recall_score(y_test_clf, pred),
           "f1": f1_score(y_test_clf, pred),
           "roc_auc": roc_auc_score(y_test_clf, proba),
           "log_loss": log_loss(y_test_clf, proba)}
    clf_results.append(row)
    for key, value in row.items():
        print("  ", key, round(value, 4) if isinstance(value, float) else value)
    return pred

clf_b = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                       num_leaves=127, random_state=42, verbose=-1)
clf_b.fit(X_full_b, (full_b["ARRIVAL_DELAY"] > 15).astype(int))

proba_b = clf_b.predict_proba(X_test_b)[:, 1]
pred_b = classification_row("LightGBM CE - B (final)", proba_b, threshold=0.17)

print("\nConfusion matrix (rows = real, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test_clf, pred_b),
                   index=["real: not delayed", "real: delayed"],
                   columns=["pred: not delayed", "pred: delayed"]))

# %% 7.4 — Version A (reference only; the final choice stays B)
categories_a = preprocess.learn_categories(full_train)
X_full_a = preprocess.make_lgbm_table(full_train, categories_a, use_month=True)
X_test_a = preprocess.make_lgbm_table(test, categories_a, use_month=True)

reg_a = LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.1,
                      num_leaves=127, random_state=42, verbose=-1)
reg_a.fit(X_full_a, y_full_reg)
mae_a = mean_absolute_error(y_test_reg, reg_a.predict(X_test_a))
reg_results.append({"model": "LightGBM l1 - A (reference)", "test_mae": mae_a})
print("Test MAE (A):", round(mae_a, 3), "| B:", round(mae_b, 3), "| baseline:", round(baseline_mae, 3))

clf_a = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                       num_leaves=127, random_state=42, verbose=-1)
clf_a.fit(X_full_a, y_full_clf)
proba_a = clf_a.predict_proba(X_test_a)[:, 1]
print("\nClassification (A):")
pred_a = classification_row("LightGBM CE - A (reference)", proba_a, threshold=0.18)

# %% 7.5 — Summary tables
reg_table = pd.DataFrame(reg_results)
reg_table["improvement_vs_baseline"] = baseline_mae - reg_table["test_mae"]
clf_table = pd.DataFrame(clf_results)

reg_table.round(3).to_csv("/kaggle/working/results/phase7_test_regression.csv", index=False)
clf_table.round(4).to_csv("/kaggle/working/results/phase7_test_classification.csv", index=False)

print(reg_table.round(3).to_string(index=False))
print()
print(clf_table.round(4).to_string(index=False))

# %% 7.5 — Where does the final model make larger errors? (by month and time of day)
errors = pd.DataFrame({
    "MONTH": test_b["MONTH"].values,
    "DEP_HOUR": test_b["DEP_HOUR"].values,
    "AIRLINE": test_b["AIRLINE"].values,
    "model_error": np.abs(test_b["ARRIVAL_DELAY"].values - pred_reg_b),
    "baseline_error": np.abs(test_b["ARRIVAL_DELAY"].values - full_median),
})

by_month = errors.groupby("MONTH")[["model_error", "baseline_error"]].mean().round(2)
by_month["gain"] = by_month["baseline_error"] - by_month["model_error"]
print("Mean absolute error by month:")
print(by_month)

errors["time_of_day"] = pd.cut(errors["DEP_HOUR"], bins=[-1, 5, 11, 17, 23],
                               labels=["night (0-5)", "morning (6-11)", "afternoon (12-17)", "evening (18-23)"])
by_time = errors.groupby("time_of_day", observed=True)[["model_error", "baseline_error"]].mean().round(2)
by_time["gain"] = by_time["baseline_error"] - by_time["model_error"]
print("\nMean absolute error by time of day:")
print(by_time)

by_month.to_csv("/kaggle/working/results/phase7_error_by_month.csv")
by_time.to_csv("/kaggle/working/results/phase7_error_by_time.csv")

# %% 7.6 — Save test predictions for the charts (Phase 9)
test_predictions = pd.DataFrame({
    "actual_delay": test_b["ARRIVAL_DELAY"].values,
    "predicted_delay_b": pred_reg_b,
    "actual_late": y_test_clf.values,
    "probability_b": proba_b,
    "probability_a": proba_a,
    "MONTH": test_b["MONTH"].values,
    "DEP_HOUR": test_b["DEP_HOUR"].values,
})
test_predictions.to_parquet("/kaggle/working/results/phase7_test_predictions.parquet", index=False)

print("Saved:", sorted(os.listdir("/kaggle/working/results")))
