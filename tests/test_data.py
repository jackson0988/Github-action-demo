"""Tests for load_month: bad trips must be removed and good trips kept."""
import pandas as pd
import pytest

import common


def make_trip(pickup, minutes, miles=2.0):
    start = pd.Timestamp(pickup)
    return {
        "lpep_pickup_datetime": start,
        "lpep_dropoff_datetime": start + pd.Timedelta(minutes=minutes),
        "PULocationID": 74,
        "DOLocationID": 75,
        "trip_distance": miles,
    }


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Write trips to a small test file, then load them with load_month."""
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)

    def _load(trips, month="2020-01"):
        pd.DataFrame(trips).to_parquet(tmp_path / f"green_tripdata_{month}.parquet")
        return common.load_month(month)

    return _load


def test_keeps_a_normal_trip(load):
    trips = load([make_trip("2020-01-15 08:30", minutes=12)])
    assert len(trips) == 1


def test_calculates_duration_hour_and_weekday(load):
    trip = load([make_trip("2020-01-15 08:30", minutes=12)]).iloc[0]  # a Wednesday
    assert trip["duration"] == 12
    assert trip["hour"] == 8
    assert trip["weekday"] == 2  # Monday is 0


def test_removes_a_trip_under_one_minute(load):
    assert len(load([make_trip("2020-01-15 08:30", minutes=0)])) == 0


def test_removes_a_trip_over_an_hour(load):
    assert len(load([make_trip("2020-01-15 08:30", minutes=90)])) == 0


def test_removes_a_trip_with_no_distance(load):
    assert len(load([make_trip("2020-01-15 08:30", minutes=12, miles=0)])) == 0


def test_removes_a_trip_from_another_month(load):
    assert len(load([make_trip("2019-12-31 23:50", minutes=12)])) == 0


def test_returns_the_columns_the_model_needs(load):
    trips = load([make_trip("2020-01-15 08:30", minutes=12)])
    for column in common.FEATURES + [common.TARGET]:
        assert column in trips.columns
