"""Shared preprocessing for all models (Phase 4). Learn from train only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK"]
numeric_cols = ["DEP_MIN", "ARR_MIN", "SCHEDULED_TIME", "DISTANCE"]
month_cols = ["MONTH_SIN", "MONTH_COS"]


def hhmm_to_minutes(hhmm):
    """1435 -> 875 minutes after midnight. 2400 -> 0."""
    hours = hhmm // 100
    minutes = hhmm % 100
    return (hours * 60 + minutes) % 1440


def add_features(data):
    """Add time in minutes and the month as a cycle (sin, cos)."""
    data = data.copy()
    data["DEP_MIN"] = hhmm_to_minutes(data["SCHEDULED_DEPARTURE"])
    data["ARR_MIN"] = hhmm_to_minutes(data["SCHEDULED_ARRIVAL"])
    data["MONTH_SIN"] = np.sin(2 * np.pi * data["MONTH"] / 12)
    data["MONTH_COS"] = np.cos(2 * np.pi * data["MONTH"] / 12)
    return data


def make_onehot_preprocessor(use_month):
    """One-hot for categories, scaling for numbers (Linear/Ridge, RandomForest)."""
    if use_month:
        number_cols = numeric_cols + month_cols
    else:
        number_cols = numeric_cols
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numbers", StandardScaler(), number_cols),
    ])


def learn_categories(train_data):
    """Category lists learned from train only (for LightGBM)."""
    categories = {}
    for col in categorical_cols:
        categories[col] = sorted(train_data[col].unique())
    return categories


def make_lgbm_table(data, categories, use_month):
    """Columns for LightGBM. Categories not seen in train become NaN."""
    if use_month:
        cols = categorical_cols + numeric_cols + month_cols
    else:
        cols = categorical_cols + numeric_cols
    table = data[cols].copy()
    for col in categorical_cols:
        table[col] = pd.Categorical(table[col], categories=categories[col])
    return table
