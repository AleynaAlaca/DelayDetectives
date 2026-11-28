"""Prediction for one flight (used by the website). Written in Phase 8.

Needs preprocess.py and features.py in the same folder, and the model files from Phase 8:
regression_model.joblib, classification_model.joblib, categories.joblib,
origin_traffic.csv, dest_traffic.csv, route_table.csv
"""
import joblib
import pandas as pd
import preprocess
import features


def load_model_files(folder):
    """Load the models and tables once, when the website starts."""
    code_types = {"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str}
    files = {}
    files["regression"] = joblib.load(folder + "/regression_model.joblib")
    files["classification"] = joblib.load(folder + "/classification_model.joblib")
    files["categories"] = joblib.load(folder + "/categories.joblib")
    files["origin_traffic"] = pd.read_csv(folder + "/origin_traffic.csv", dtype=code_types)
    files["dest_traffic"] = pd.read_csv(folder + "/dest_traffic.csv", dtype=code_types)
    files["routes"] = pd.read_csv(folder + "/route_table.csv", dtype=code_types)
    return files


def risk_level(probability):
    """0.17 is the classification threshold chosen in Phase 6b."""
    if probability < 0.17:
        return "low"
    elif probability < 0.30:
        return "medium"
    else:
        return "high"


def predict_flight(files, airline, origin, destination, date, departure_time):
    """departure_time as HHMM, for example 1430. date as 'YYYY-MM-DD'."""
    routes = files["routes"]
    route = routes[(routes["ORIGIN_AIRPORT"] == origin) & (routes["DESTINATION_AIRPORT"] == destination)]
    if len(route) == 0:
        raise ValueError("No flights on this route in the data: " + origin + " -> " + destination)
    route = route.iloc[0]

    date = pd.Timestamp(date)
    dep_min = preprocess.hhmm_to_minutes(departure_time)
    arr_min = (dep_min + int(route["ARR_OFFSET"])) % 1440
    arrival_time = (arr_min // 60) * 100 + arr_min % 60

    flight = pd.DataFrame({
        "MONTH": [date.month],
        "DAY_OF_WEEK": [date.isoweekday()],
        "AIRLINE": [airline],
        "ORIGIN_AIRPORT": [origin],
        "DESTINATION_AIRPORT": [destination],
        "SCHEDULED_DEPARTURE": [departure_time],
        "SCHEDULED_ARRIVAL": [arrival_time],
        "SCHEDULED_TIME": [route["SCHEDULED_TIME"]],
        "DISTANCE": [route["DISTANCE"]],
        "DATE": [date],
    })
    flight = preprocess.add_features(flight)
    flight = features.add_new_features(flight, files["origin_traffic"], files["dest_traffic"])
    X = features.make_table(flight, files["categories"])

    delay = float(files["regression"].predict(X)[0])
    probability = float(files["classification"].predict_proba(X)[0, 1])
    return {
        "expected_delay_min": round(delay, 1),
        "delay_probability": round(probability, 3),
        "risk": risk_level(probability),
        "arrival_time": int(arrival_time),
    }
