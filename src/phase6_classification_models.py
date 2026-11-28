"""
Phase 6 — Classification Model Comparison

Goal: predict whether a flight will be more than 15 minutes late.
Models: baseline, Logistic Regression, LightGBM with cross entropy, class weights and focal loss.
All models are trained on the full train set and compared on validation. The test set is not used.

Runs on Kaggle with two inputs attached (Add Input):
    - Phase 3 notebook output   -> train.parquet, val.parquet
    - Phase 4 notebook output   -> preprocess.py
Outputs (in /kaggle/working/results):
    phase6_classification_results.csv    all models, threshold 0.5 and best threshold
    phase6_val_probabilities.parquet     validation probabilities of all models (for charts)
    lgbm_classifier.joblib               chosen model (for Phase 7 and the website)
    categories.joblib                    category lists learned from train
"""

# %% 6.1 — Load and build the target (1 = more than 15 minutes late)
import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

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

y_train = (train["ARRIVAL_DELAY"] > 15).astype(int)
y_val = (val["ARRIVAL_DELAY"] > 15).astype(int)

os.makedirs("/kaggle/working/results", exist_ok=True)
print("train:", train.shape, "| val:", val.shape)
print("Delayed share in train:", round(y_train.mean() * 100, 1), "%")
print("Delayed share in val  :", round(y_val.mean() * 100, 1), "%")

# %% 6.2 — Evaluation function and baseline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

results = []

def evaluate(name, proba, seconds, size_mb, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    row = {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred),
        "f1": f1_score(y_val, pred),
        "roc_auc": roc_auc_score(y_val, proba),
        "log_loss": log_loss(y_val, proba),
        "train_seconds": seconds,
        "size_mb": size_mb,
    }
    results.append(row)
    print(name, "| threshold", threshold)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]:
        print("  ", key.ljust(9), round(row[key], 3))

baseline_proba = np.full(len(y_val), y_train.mean())
evaluate("Baseline (always 'not delayed')", baseline_proba, 0.0, 0.0)

# %% 6.3 — One-hot data (with cyclical month)
prep = preprocess.make_onehot_preprocessor(use_month=True)
X_train_oh = prep.fit_transform(train)
X_val_oh = prep.transform(val)

print("One-hot:", X_train_oh.shape, X_val_oh.shape)

# %% 6.3 — Logistic Regression (cross entropy)
from sklearn.linear_model import LogisticRegression
import joblib

def model_size(model):
    joblib.dump(model, "/kaggle/working/temp_model.joblib")
    return os.path.getsize("/kaggle/working/temp_model.joblib") / 1e6

logreg = LogisticRegression(max_iter=300)

start = time.time()
logreg.fit(X_train_oh, y_train)
seconds = time.time() - start

proba_logreg = logreg.predict_proba(X_val_oh)[:, 1]
evaluate("Logistic Regression", proba_logreg, seconds, model_size(logreg))
print("  training time:", round(seconds, 1), "s")

# %% 6.4 — LightGBM tables (category types, categories learned from train)
categories = preprocess.learn_categories(train)
lgbm_train = preprocess.make_lgbm_table(train, categories, use_month=True)
lgbm_val = preprocess.make_lgbm_table(val, categories, use_month=True)

print("LightGBM tables:", lgbm_train.shape, lgbm_val.shape)

# %% 6.4 — LightGBM with cross entropy (same settings as the chosen regression model)
lgbm_ce = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                         num_leaves=127, random_state=42, verbose=-1)

start = time.time()
lgbm_ce.fit(lgbm_train, y_train)
seconds = time.time() - start

proba_lgbm_ce = lgbm_ce.predict_proba(lgbm_val)[:, 1]
evaluate("LightGBM (cross entropy)", proba_lgbm_ce, seconds, model_size(lgbm_ce))
print("  training time:", round(seconds, 1), "s")

# %% 6.5 — LightGBM with class weights (weighted cross entropy)
weight = (y_train == 0).sum() / (y_train == 1).sum()
print("Weight for delayed flights:", round(weight, 2))

lgbm_weighted = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                               num_leaves=127, scale_pos_weight=weight,
                               random_state=42, verbose=-1)

start = time.time()
lgbm_weighted.fit(lgbm_train, y_train)
seconds = time.time() - start

proba_lgbm_weighted = lgbm_weighted.predict_proba(lgbm_val)[:, 1]
evaluate("LightGBM (weighted cross entropy)", proba_lgbm_weighted, seconds, model_size(lgbm_weighted))
print("  training time:", round(seconds, 1), "s")
print("  mean predicted probability:", round(proba_lgbm_weighted.mean(), 3), "| real delayed share:", round(y_val.mean(), 3))

# %% 6.6 — Focal loss: -(1 - p)^gamma * log(p), derivatives computed numerically
GAMMA = 2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def focal_loss(raw, y):
    p = np.clip(sigmoid(raw), 1e-7, 1 - 1e-7)
    p_true = np.where(y == 1, p, 1 - p)
    return -((1 - p_true) ** GAMMA) * np.log(p_true)

def focal_objective(y, raw):
    step = 1e-3
    loss_mid = focal_loss(raw, y)
    loss_up = focal_loss(raw + step, y)
    loss_down = focal_loss(raw - step, y)
    gradient = (loss_up - loss_down) / (2 * step)
    hessian = (loss_up - 2 * loss_mid + loss_down) / step ** 2
    hessian = np.maximum(hessian, 1e-6)
    return gradient, hessian

# %% 6.6 — Check: with gamma = 0 focal loss must equal cross entropy
raw_test = np.array([-2.0, 0.0, 2.0])
y_test = np.array([1, 0, 1])

GAMMA = 0
gradient, hessian = focal_objective(y_test, raw_test)
print("gamma = 0 (should equal cross entropy)")
print("  numeric gradient :", gradient.round(4))
print("  exact CE gradient:", (sigmoid(raw_test) - y_test).round(4))
print("  numeric hessian  :", hessian.round(4))
print("  exact CE hessian :", (sigmoid(raw_test) * (1 - sigmoid(raw_test))).round(4))

GAMMA = 2

# %% 6.6 — LightGBM with focal loss (start from the delayed share, not from 50%)
base_score = np.log(y_train.mean() / (1 - y_train.mean()))
print("Starting raw score:", round(base_score, 3), "-> probability", round(sigmoid(base_score), 3))

lgbm_focal = LGBMClassifier(objective=focal_objective, n_estimators=300, learning_rate=0.1,
                            num_leaves=127, random_state=42, verbose=-1)

start = time.time()
lgbm_focal.fit(lgbm_train, y_train, init_score=np.full(len(y_train), base_score))
seconds = time.time() - start

raw_val = lgbm_focal.predict(lgbm_val, raw_score=True) + base_score
proba_lgbm_focal = sigmoid(raw_val)

evaluate("LightGBM (focal loss, gamma 2)", proba_lgbm_focal, seconds, model_size(lgbm_focal))
print("  training time:", round(seconds, 1), "s")
print("  mean predicted probability:", round(proba_lgbm_focal.mean(), 3), "| real delayed share:", round(y_val.mean(), 3))

# %% 6.7 — Best threshold (highest F1 on validation) for each model
def best_threshold(proba):
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (proba >= t).astype(int))
        if f1 > best_f1:
            best_t = t
            best_f1 = f1
    return round(best_t, 2)

probabilities = {
    "Logistic Regression": proba_logreg,
    "LightGBM (cross entropy)": proba_lgbm_ce,
    "LightGBM (weighted cross entropy)": proba_lgbm_weighted,
    "LightGBM (focal loss, gamma 2)": proba_lgbm_focal,
}

seconds_and_size = {}
for row in results:
    seconds_and_size[row["model"]] = (row["train_seconds"], row["size_mb"])

for name, proba in probabilities.items():
    t = best_threshold(proba)
    seconds, size = seconds_and_size[name]
    evaluate(name, proba, seconds, size, threshold=t)

# %% 6.8 — Results table
results_table = pd.DataFrame(results).round(3)
results_table.to_csv("/kaggle/working/results/phase6_classification_results.csv", index=False)
print(results_table[["model", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]].to_string(index=False))

# %% 6.8 — Chosen model: LightGBM (cross entropy), threshold 0.18
from sklearn.metrics import confusion_matrix

chosen_threshold = 0.18
chosen_pred = (proba_lgbm_ce >= chosen_threshold).astype(int)

matrix = confusion_matrix(y_val, chosen_pred)
print("Confusion matrix (rows = real, columns = predicted):")
print(pd.DataFrame(matrix, index=["real: not delayed", "real: delayed"],
                   columns=["pred: not delayed", "pred: delayed"]))

joblib.dump(lgbm_ce, "/kaggle/working/results/lgbm_classifier.joblib")
joblib.dump(categories, "/kaggle/working/results/categories.joblib")

val_probabilities = pd.DataFrame({"actual": y_val.values,
                                  "logistic_regression": proba_logreg,
                                  "lgbm_cross_entropy": proba_lgbm_ce,
                                  "lgbm_weighted": proba_lgbm_weighted,
                                  "lgbm_focal": proba_lgbm_focal})
val_probabilities.to_parquet("/kaggle/working/results/phase6_val_probabilities.parquet", index=False)

print("\nSaved:", sorted(os.listdir("/kaggle/working/results")))
