"""
Phase 5 — Regression Model Comparison

Goal: compare a median baseline, Linear/Ridge Regression, RandomForest and LightGBM
for predicting the arrival delay in minutes. All models are trained on the full train set
and evaluated on validation (MAE, training time, model size). The test set is not used.

Runs on Kaggle with two inputs attached (Add Input):
    - Phase 3 notebook output   -> train.parquet, val.parquet
    - Phase 4 notebook output   -> preprocess.py
Outputs (in /kaggle/working/results):
    phase5_regression_results.csv     all models: MAE, time, size
    phase5_feature_importance.csv     chosen LightGBM model
    phase5_val_predictions.parquet    chosen model, validation predictions (for charts)
    lgbm_regression.joblib            chosen model (for Phase 7 and the website)
    categories.joblib                 category lists learned from train
"""

# %% 5.1 — Load (LightGBM is imported first so a loading problem shows up immediately)
import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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

y_train = train["ARRIVAL_DELAY"]
y_val = val["ARRIVAL_DELAY"]

os.makedirs("/kaggle/working/results", exist_ok=True)
print("train:", train.shape, "| val:", val.shape)

# %% 5.2 — Baseline: always predict the training median
from sklearn.metrics import mean_absolute_error

results = []

train_median = y_train.median()
baseline_pred = np.full(len(y_val), train_median)
baseline_mae = mean_absolute_error(y_val, baseline_pred)

results.append({"model": "Baseline (median)", "month": "-", "val_mae": baseline_mae,
                "train_seconds": 0.0, "size_mb": 0.0})

print("Train median:", train_median)
print("Baseline validation MAE:", round(baseline_mae, 2))

# %% 5.3 — One-hot data, without month and with cyclical month
prep_no_month = preprocess.make_onehot_preprocessor(use_month=False)
X_train_no_month = prep_no_month.fit_transform(train)
X_val_no_month = prep_no_month.transform(val)

prep_month = preprocess.make_onehot_preprocessor(use_month=True)
X_train_month = prep_month.fit_transform(train)
X_val_month = prep_month.transform(val)

print("Without month:", X_train_no_month.shape, X_val_no_month.shape)
print("With month   :", X_train_month.shape, X_val_month.shape)

# %% 5.4 — Helper: train, time, predict, measure MAE and model size, store the result
import joblib

def run_model(name, month, model, X_tr, X_v):
    start = time.time()
    model.fit(X_tr, y_train)
    seconds = time.time() - start

    pred = model.predict(X_v)
    mae = mean_absolute_error(y_val, pred)

    joblib.dump(model, "/kaggle/working/temp_model.joblib")
    size_mb = os.path.getsize("/kaggle/working/temp_model.joblib") / 1e6

    results.append({"model": name, "month": month, "val_mae": mae,
                    "train_seconds": seconds, "size_mb": size_mb})

    print(name, "| month:", month)
    print("  validation MAE :", round(mae, 2), "(baseline", round(baseline_mae, 2), ")")
    print("  training time  :", round(seconds, 1), "s")
    print("  model size     :", round(size_mb, 2), "MB")
    print("  mean prediction:", round(pred.mean(), 2))
    return pred

# %% 5.5 — Linear Regression
from sklearn.linear_model import LinearRegression

pred_lr_no_month = run_model("Linear Regression", "no", LinearRegression(), X_train_no_month, X_val_no_month)
pred_lr_month = run_model("Linear Regression", "cyclical", LinearRegression(), X_train_month, X_val_month)

# %% 5.5 — Ridge Regression
from sklearn.linear_model import Ridge

pred_ridge_no_month = run_model("Ridge Regression", "no", Ridge(alpha=1.0), X_train_no_month, X_val_no_month)
pred_ridge_month = run_model("Ridge Regression", "cyclical", Ridge(alpha=1.0), X_train_month, X_val_month)

# %% 5.6 — Ridge + median shift (shift learned from train only)
ridge = Ridge(alpha=1.0)

start = time.time()
ridge.fit(X_train_no_month, y_train)
seconds = time.time() - start

train_errors = y_train - ridge.predict(X_train_no_month)
shift = np.median(train_errors)

pred_ridge_shifted = ridge.predict(X_val_no_month) + shift
mae = mean_absolute_error(y_val, pred_ridge_shifted)

results.append({"model": "Ridge + median shift", "month": "no", "val_mae": mae,
                "train_seconds": seconds, "size_mb": 0.01})

print("Shift learned from train:", round(shift, 2), "min")
print("Validation MAE:", round(mae, 2), "(baseline", round(baseline_mae, 2), ")")
print("Mean prediction:", round(pred_ridge_shifted.mean(), 2))

# %% 5.7 — LightGBM tables (category types, categories learned from train)
categories = preprocess.learn_categories(train)

lgbm_train_no_month = preprocess.make_lgbm_table(train, categories, use_month=False)
lgbm_val_no_month = preprocess.make_lgbm_table(val, categories, use_month=False)

lgbm_train_month = preprocess.make_lgbm_table(train, categories, use_month=True)
lgbm_val_month = preprocess.make_lgbm_table(val, categories, use_month=True)

print("Without month:", lgbm_train_no_month.shape, "| With month:", lgbm_train_month.shape)

# %% 5.7 — LightGBM with l2 objective (squared error)
pred_lgbm_l2_no_month = run_model("LightGBM (l2)", "no",
    LGBMRegressor(objective="l2", random_state=42, verbose=-1),
    lgbm_train_no_month, lgbm_val_no_month)

pred_lgbm_l2_month = run_model("LightGBM (l2)", "cyclical",
    LGBMRegressor(objective="l2", random_state=42, verbose=-1),
    lgbm_train_month, lgbm_val_month)

# %% 5.7 — LightGBM with l1 objective (absolute error = what MAE measures)
pred_lgbm_l1_no_month = run_model("LightGBM (l1)", "no",
    LGBMRegressor(objective="l1", random_state=42, verbose=-1),
    lgbm_train_no_month, lgbm_val_no_month)

pred_lgbm_l1_month = run_model("LightGBM (l1)", "cyclical",
    LGBMRegressor(objective="l1", random_state=42, verbose=-1),
    lgbm_train_month, lgbm_val_month)

# %% 5.8 — RandomForest (limited so that it finishes in reasonable time; about 25 minutes on Kaggle CPU)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=50,
    max_samples=0.2,
    min_samples_leaf=100,
    max_features=0.3,
    n_jobs=-1,
    random_state=42,
)

pred_rf_month = run_model("RandomForest", "cyclical", rf, X_train_month, X_val_month)

# %% 5.9 — RandomForest + median shift (shift from 500,000 random train rows)
rf_seconds = results[-1]["train_seconds"]
rf_size = results[-1]["size_mb"]

rng = np.random.default_rng(42)
sample_rows = rng.choice(X_train_month.shape[0], size=500_000, replace=False)

train_errors = y_train.values[sample_rows] - rf.predict(X_train_month[sample_rows])
rf_shift = np.median(train_errors)

pred_rf_shifted = pred_rf_month + rf_shift
mae = mean_absolute_error(y_val, pred_rf_shifted)

results.append({"model": "RandomForest + median shift", "month": "cyclical", "val_mae": mae,
                "train_seconds": rf_seconds, "size_mb": rf_size})

print("Shift learned from train:", round(rf_shift, 2), "min")
print("Validation MAE:", round(mae, 2), "(baseline", round(baseline_mae, 2), ")")
print("Mean prediction:", round(pred_rf_shifted.mean(), 2))

# %% 5.10 — Save the results so far
results_table = pd.DataFrame(results)
results_table["beats_baseline"] = results_table["val_mae"] < baseline_mae
results_table = results_table.round(2)

results_table.to_csv("/kaggle/working/results/phase5_regression_results.csv", index=False)
print(results_table.to_string(index=False))

# %% 5.11 — LightGBM (l1) settings, chosen on validation
settings = [
    {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 127},
    {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 127},
    {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 255},
]

for s in settings:
    name = "LightGBM (l1) " + str(s["n_estimators"]) + " trees, lr " + str(s["learning_rate"]) + ", " + str(s["num_leaves"]) + " leaves"
    model = LGBMRegressor(objective="l1", random_state=42, verbose=-1,
                          n_estimators=s["n_estimators"],
                          learning_rate=s["learning_rate"],
                          num_leaves=s["num_leaves"])
    run_model(name, "cyclical", model, lgbm_train_month, lgbm_val_month)

# %% 5.12 — Chosen model: LightGBM (l1), 300 trees, lr 0.1, 127 leaves, cyclical month
final_model = LGBMRegressor(objective="l1", random_state=42, verbose=-1,
                            n_estimators=300, learning_rate=0.1, num_leaves=127)
final_model.fit(lgbm_train_month, y_train)

final_pred = final_model.predict(lgbm_val_month)
print("Validation MAE:", round(mean_absolute_error(y_val, final_pred), 2))

joblib.dump(final_model, "/kaggle/working/results/lgbm_regression.joblib")
joblib.dump(categories, "/kaggle/working/results/categories.joblib")
print("Model and category lists saved")

# %% 5.12 — Feature importance, validation predictions, final results table
importance = pd.DataFrame({
    "feature": lgbm_train_month.columns,
    "importance": final_model.booster_.feature_importance(importance_type="gain"),
})
importance["percent"] = (importance["importance"] / importance["importance"].sum() * 100).round(1)
importance = importance.sort_values("percent", ascending=False)
importance.to_csv("/kaggle/working/results/phase5_feature_importance.csv", index=False)
print(importance[["feature", "percent"]].to_string(index=False))

predictions = pd.DataFrame({"actual": y_val.values, "predicted": final_pred,
                            "MONTH": val["MONTH"].values, "DEP_MIN": val["DEP_MIN"].values})
predictions.to_parquet("/kaggle/working/results/phase5_val_predictions.parquet", index=False)

results_table = pd.DataFrame(results)
results_table["beats_baseline"] = results_table["val_mae"] < baseline_mae
results_table.round(2).to_csv("/kaggle/working/results/phase5_regression_results.csv", index=False)
print("\nSaved:", sorted(os.listdir("/kaggle/working/results")))
