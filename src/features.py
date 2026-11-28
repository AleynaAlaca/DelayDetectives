"""New pre-departure features (Phase 6b). Traffic tables are learned from train only."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK", "ROUTE"]
model_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DAY_OF_WEEK",
              "DEP_MIN", "ARR_MIN", "SCHEDULED_TIME", "DISTANCE", "MONTH_SIN", "MONTH_COS",
              "ROUTE", "ORIGIN_HOUR_FLIGHTS", "DEST_HOUR_FLIGHTS", "DAYS_TO_HOLIDAY"]


def add_hour_and_route(data):
    """Needs DEP_MIN and ARR_MIN from preprocess.add_features."""
    data = data.copy()
    data["DEP_HOUR"] = data["DEP_MIN"] // 60
    data["ARR_HOUR"] = data["ARR_MIN"] // 60
    data["ROUTE"] = data["ORIGIN_AIRPORT"] + "_" + data["DESTINATION_AIRPORT"]
    return data


def learn_traffic_tables(train):
    """Average number of scheduled flights per day, per airport and hour (train only)."""
    days = train["DATE"].nunique()
    origin = train.groupby(["ORIGIN_AIRPORT", "DEP_HOUR"]).size() / days
    origin = origin.rename("ORIGIN_HOUR_FLIGHTS").reset_index()
    dest = train.groupby(["DESTINATION_AIRPORT", "ARR_HOUR"]).size() / days
    dest = dest.rename("DEST_HOUR_FLIGHTS").reset_index()
    return origin, dest


def days_to_holiday(dates):
    """Days to the nearest US federal holiday, for any year."""
    start = str(dates.min().year - 1) + "-12-01"
    end = str(dates.max().year + 1) + "-01-31"
    holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
    distance = {}
    for day in dates.unique():
        distance[day] = int(np.abs((holidays - day).days).min())
    return dates.map(distance)


def add_new_features(data, origin_traffic, dest_traffic):
    """Route, airport traffic and distance to holidays."""
    data = add_hour_and_route(data)
    data = data.merge(origin_traffic, on=["ORIGIN_AIRPORT", "DEP_HOUR"], how="left")
    data = data.merge(dest_traffic, on=["DESTINATION_AIRPORT", "ARR_HOUR"], how="left")
    data["DAYS_TO_HOLIDAY"] = days_to_holiday(data["DATE"])
    return data


def learn_categories(train):
    categories = {}
    for col in categorical_cols:
        categories[col] = sorted(train[col].unique())
    return categories


def make_table(data, categories):
    """Model input. Categories not seen in train become NaN."""
    table = data[model_cols].copy()
    for col in categorical_cols:
        table[col] = pd.Categorical(table[col], categories=categories[col])
    return table
