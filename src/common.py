"""Settings and helpers shared by the training, monitoring and retraining scripts."""
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Your local MLflow server, unless MLFLOW_TRACKING_URI says otherwise
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "taxi-trips"
MODEL_NAME = "taxi-model"

FEATURES = ["PULocationID", "DOLocationID", "trip_distance", "hour", "weekday"]
TARGET = "duration"
MAX_MAE = 4.1  # raise the alarm above this error, about 20% worse than at launch
MAX_TRAIN_ROWS = 100_000  # keeps training fast on a laptop

DATA_DIR = Path("data")
ALL_MONTHS = pd.period_range("2019-07", "2021-03", freq="M").astype(str).tolist()


def connect():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.MlflowClient()


def load_month(month):
    """Read one month of trips and remove rows that make no sense."""
    trips = pd.read_parquet(DATA_DIR / f"green_tripdata_{month}.parquet")
    trips["pickup"] = trips["lpep_pickup_datetime"]
    trips[TARGET] = (trips["lpep_dropoff_datetime"] - trips["pickup"]).dt.total_seconds() / 60
    trips["hour"] = trips["pickup"].dt.hour
    trips["weekday"] = trips["pickup"].dt.dayofweek

    keep = (
        (trips["pickup"].dt.strftime("%Y-%m") == month)
        & trips[TARGET].between(1, 60)
        & trips["trip_distance"].between(0.1, 50)
    )
    return trips.loc[keep, ["pickup", *FEATURES, TARGET]].dropna()


def load_months(months):
    return pd.concat([load_month(month) for month in months], ignore_index=True)


def train_model(trips, max_depth):
    if len(trips) > MAX_TRAIN_ROWS:
        trips = trips.sample(MAX_TRAIN_ROWS, random_state=42)
    model = RandomForestRegressor(
        n_estimators=30, max_depth=max_depth, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    return model.fit(trips[FEATURES], trips[TARGET])


def error_of(model, trips):
    """Average error in minutes (MAE). Lower is better."""
    return mean_absolute_error(trips[TARGET], model.predict(trips[FEATURES]))


def register(model, trips):
    """Save the model in the active run and add it to the registry as a new version."""
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=trips[FEATURES].head(3),
        registered_model_name=MODEL_NAME,
        # We built this model ourselves, so its tree objects are safe
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    return info.registered_model_version
