"""Tests for the API. They use a small test model, so they need no MLflow server and no data."""
import importlib
import os
import sys

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from common import FEATURES, train_model

GOOD_TRIP = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 2.5, "hour": 8, "weekday": 1}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny model on made-up trips and save it the way export_model.py does
    folder = tmp_path_factory.mktemp("api")
    rng = np.random.default_rng(0)
    trips = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, 266, 500),
            "DOLocationID": rng.integers(1, 266, 500),
            "trip_distance": rng.uniform(0.5, 10, 500),
            "hour": rng.integers(0, 24, 500),
            "weekday": rng.integers(0, 7, 500),
        }
    )
    trips["duration"] = trips["trip_distance"] * 4 + 3
    model = train_model(trips, max_depth=3)
    mlflow.sklearn.save_model(model, folder / "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    (folder / "model" / "version.txt").write_text("test")

    # serve.py loads model/ from the current folder when it is imported
    old_folder = os.getcwd()
    os.chdir(folder)
    try:
        sys.modules.pop("serve", None)  # import it fresh, with our test model
        serve = importlib.import_module("serve")
    finally:
        os.chdir(old_folder)
    return TestClient(serve.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_predict_returns_minutes(client):
    response = client.post("/predict", json=GOOD_TRIP)
    assert response.status_code == 200
    assert response.json()["minutes"] > 0
    assert response.json()["model_version"] == "test"


def test_api_fields_match_model_features(client):
    # If someone renames a field, the model would get the wrong columns
    import serve

    assert list(serve.Trip.model_fields) == FEATURES


def test_rejects_unknown_zone(client):
    response = client.post("/predict", json={**GOOD_TRIP, "PULocationID": 999})
    assert response.status_code == 422


def test_rejects_missing_field(client):
    trip = {key: value for key, value in GOOD_TRIP.items() if key != "trip_distance"}
    assert client.post("/predict", json=trip).status_code == 422


def test_rejects_wrong_hour(client):
    assert client.post("/predict", json={**GOOD_TRIP, "hour": 24}).status_code == 422
