"""
Phase 8 — Model Export for the Website

The final models are retrained on all data (train + validation + test) with the settings fixed in Phase 7.
No model choice is made here; the measured performance is the one reported in Phase 7.

Runs on Kaggle with these inputs attached (Add Input):
    - Phase 3 notebook output    -> train.parquet, val.parquet, test.parquet
    - Phase 4 notebook output    -> preprocess.py
    - Phase 6b notebook output   -> features.py
    - Phase 7 notebook output    -> phase7_test_predictions.parquet (risk level check)
    - usdot/flight-delays        -> airlines.csv, airports.csv (names for the website)
Outputs (in /kaggle/working/model): models, tables, lists, predict.py (= src/predict.py)
"""

# %% 8.1 — Load all data
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

all_data = pd.concat([train, val, test], ignore_index=True)
del train, val, test

all_data = preprocess.add_features(all_data)

os.makedirs("/kaggle/working/model", exist_ok=True)
print("all data:", all_data.shape, "| days:", all_data["DATE"].nunique())

# %% 8.2 — Version B features (learned from all data)
all_b = features.add_hour_and_route(all_data)
origin_traffic, dest_traffic = features.learn_traffic_tables(all_b)
all_b = features.add_new_features(all_b, origin_traffic, dest_traffic)

categories = features.learn_categories(all_b)
X_all = features.make_table(all_b, categories)

y_reg = all_b["ARRIVAL_DELAY"]
y_clf = (y_reg > 15).astype(int)

print("X_all:", X_all.shape)
print("routes:", all_b["ROUTE"].nunique())
print("delayed share:", round(y_clf.mean() * 100, 1), "%")
print("missing values per column:")
print(X_all.isna().sum())

# %% 8.3 — Train the final models and save them
import joblib

reg_final = LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.1,
                          num_leaves=127, random_state=42, verbose=-1)
start = time.time()
reg_final.fit(X_all, y_reg)
print("regression trained in", round(time.time() - start, 1), "s")

clf_final = LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1,
                           num_leaves=127, random_state=42, verbose=-1)
start = time.time()
clf_final.fit(X_all, y_clf)
print("classification trained in", round(time.time() - start, 1), "s")

joblib.dump(reg_final, "/kaggle/working/model/regression_model.joblib")
joblib.dump(clf_final, "/kaggle/working/model/classification_model.joblib")
joblib.dump(categories, "/kaggle/working/model/categories.joblib")
origin_traffic.to_csv("/kaggle/working/model/origin_traffic.csv", index=False)
dest_traffic.to_csv("/kaggle/working/model/dest_traffic.csv", index=False)

for name in sorted(os.listdir("/kaggle/working/model")):
    size = os.path.getsize("/kaggle/working/model/" + name) / 1024 / 1024
    print(name, round(size, 1), "MB")

# %% 8.4 — Route table: typical distance, scheduled time and arrival offset per route
all_b["ARR_OFFSET"] = (all_b["ARR_MIN"] - all_b["DEP_MIN"]) % 1440

route_table = all_b.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]).agg(
    DISTANCE=("DISTANCE", "median"),
    SCHEDULED_TIME=("SCHEDULED_TIME", "median"),
    ARR_OFFSET=("ARR_OFFSET", "median"),
    FLIGHTS=("DISTANCE", "size"),
).reset_index()

route_table["ARR_OFFSET"] = route_table["ARR_OFFSET"].round().astype(int)

print("routes:", route_table.shape)
print(route_table.sort_values("FLIGHTS", ascending=False).head(5).to_string(index=False))

# %% 8.5 — Lists for the website (US Airways removed: it only flew until mid-2015)
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "airlines.csv" in files:
        raw_dir = folder
airlines_info = pd.read_csv(raw_dir + "/airlines.csv")
airports_info = pd.read_csv(raw_dir + "/airports.csv")

active = all_b[all_b["AIRLINE"] != "US"]
airline_routes = active[["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]].drop_duplicates()

airline_list = airlines_info[airlines_info["IATA_CODE"].isin(airline_routes["AIRLINE"])]

used_airports = set(airline_routes["ORIGIN_AIRPORT"]) | set(airline_routes["DESTINATION_AIRPORT"])
airport_list = airports_info[airports_info["IATA_CODE"].isin(used_airports)]
airport_list = airport_list[["IATA_CODE", "AIRPORT", "CITY", "STATE"]]

route_table.to_csv("/kaggle/working/model/route_table.csv", index=False)
airline_routes.to_csv("/kaggle/working/model/airline_routes.csv", index=False)
airline_list.to_csv("/kaggle/working/model/airline_list.csv", index=False)
airport_list.to_csv("/kaggle/working/model/airport_list.csv", index=False)

print("airlines:", len(airline_list), "| airports used:", len(used_airports), "| with names:", len(airport_list))
print("airline-route pairs:", len(airline_routes))
print(airline_list.to_string(index=False))

# %% 8.6 — predict.py
# In the notebook this cell starts with:  %%writefile /kaggle/working/model/predict.py
# Its content is the file src/predict.py in this repository.

# %% 8.6 — Copy preprocess.py and features.py next to predict.py, check DAY_OF_WEEK
import shutil

for folder in code_dirs:
    for name in ["preprocess.py", "features.py"]:
        if os.path.exists(folder + "/" + name):
            shutil.copy(folder + "/" + name, "/kaggle/working/model/" + name)

check = (all_b["DATE"].dt.dayofweek + 1 == all_b["DAY_OF_WEEK"]).mean()
print("DAY_OF_WEEK matches Monday = 1:", round(check * 100, 2), "%")
print(sorted(os.listdir("/kaggle/working/model")))

# %% 8.7 — Check 1: the function against the model on real rows
sys.path.append("/kaggle/working/model")
import predict

files = predict.load_model_files("/kaggle/working/model")

sample = all_b.sample(5, random_state=1)
direct_delay = reg_final.predict(X_all.loc[sample.index])
direct_proba = clf_final.predict_proba(X_all.loc[sample.index])[:, 1]

rows = []
for i in range(len(sample)):
    flight = sample.iloc[i]
    result = predict.predict_flight(files, flight["AIRLINE"], flight["ORIGIN_AIRPORT"],
                                    flight["DESTINATION_AIRPORT"], str(flight["DATE"].date()),
                                    int(flight["SCHEDULED_DEPARTURE"]))
    rows.append({"flight": flight["AIRLINE"] + " " + flight["ROUTE"],
                 "date": flight["DATE"].date(),
                 "dep": int(flight["SCHEDULED_DEPARTURE"]),
                 "real_arr": int(flight["SCHEDULED_ARRIVAL"]),
                 "function_arr": result["arrival_time"],
                 "direct_delay": round(direct_delay[i], 1),
                 "function_delay": result["expected_delay_min"],
                 "direct_proba": round(direct_proba[i], 3),
                 "function_proba": result["delay_probability"]})

print(pd.DataFrame(rows).to_string(index=False))

# %% 8.7 — Check 2: example questions a user could ask
examples = [
    ("AA", "JFK", "LAX", "2026-10-14", 700),
    ("AA", "JFK", "LAX", "2026-10-14", 1900),
    ("AA", "JFK", "LAX", "2026-06-26", 1900),
    ("AA", "JFK", "LAX", "2026-12-23", 1900),
    ("DL", "ATL", "LGA", "2026-06-26", 1900),
]

for airline, origin, destination, date, dep in examples:
    result = predict.predict_flight(files, airline, origin, destination, date, dep)
    print(airline, origin, "->", destination, date, dep, "|", result)

# %% 8.7b — Route only or airline + route? How close is each table to the real values?
route_keys = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
airline_keys = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

for name, keys in [("route only", route_keys), ("airline + route", airline_keys)]:
    typical = all_b.groupby(keys)[["SCHEDULED_TIME", "ARR_OFFSET"]].transform("median")
    time_error = (all_b["SCHEDULED_TIME"] - typical["SCHEDULED_TIME"]).abs().mean()
    offset_error = (all_b["ARR_OFFSET"] - typical["ARR_OFFSET"]).abs().mean()
    print(name, "| scheduled time error:", round(time_error, 2), "min | arrival offset error:", round(offset_error, 2), "min")

# %% 8.8 — Risk levels, checked on the Phase 7 test predictions (model never saw these flights)
for folder, subfolders, files_in in os.walk("/kaggle/input"):
    if "phase7_test_predictions.parquet" in files_in:
        test_predictions = pd.read_parquet(folder + "/phase7_test_predictions.parquet")

test_predictions["risk"] = pd.cut(test_predictions["probability_b"], bins=[0, 0.17, 0.30, 1],
                                  labels=["low", "medium", "high"], right=False)

risk_table = test_predictions.groupby("risk", observed=True).agg(
    flights=("actual_late", "size"),
    really_delayed=("actual_late", "mean"),
    median_delay=("actual_delay", "median"),
)
risk_table["share_of_flights"] = risk_table["flights"] / len(test_predictions)
print(risk_table.round(3))
print("All test flights delayed:", round(test_predictions["actual_late"].mean(), 3))

# %% 8.8 — Save the risk table
os.makedirs("/kaggle/working/results", exist_ok=True)
risk_table.round(3).to_csv("/kaggle/working/results/phase8_risk_levels.csv")
print(sorted(os.listdir("/kaggle/working/results")))
