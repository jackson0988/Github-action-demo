"""Predict one trip with the model in model/.

Usage: python src/predict.py --pickup 74 --dropoff 75 --distance 2.5 --hour 8 --weekday 1
On GitHub, the result is also written to the run's summary page.
"""
import argparse
import os

import pandas as pd

from serve import Trip, model, version  # same model, checks and fields as the API

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

parser = argparse.ArgumentParser()
parser.add_argument("--pickup", type=int, required=True, help="pickup zone, 1 to 265")
parser.add_argument("--dropoff", type=int, required=True, help="drop-off zone, 1 to 265")
parser.add_argument("--distance", type=float, required=True, help="miles")
parser.add_argument("--hour", type=int, required=True, help="0 to 23")
parser.add_argument("--weekday", type=int, required=True, help="0 is Monday, 6 is Sunday")
args = parser.parse_args()

# Trip(...) rejects bad values, exactly like the API does
trip = Trip(
    PULocationID=args.pickup,
    DOLocationID=args.dropoff,
    trip_distance=args.distance,
    hour=args.hour,
    weekday=args.weekday,
)
minutes = model.predict(pd.DataFrame([trip.model_dump()]))[0]

result = f"""### Predicted trip time: {minutes:.1f} minutes

| Input | Value |
|---|---|
| Pickup zone | {trip.PULocationID} |
| Drop-off zone | {trip.DOLocationID} |
| Distance | {trip.trip_distance} miles |
| Time | {DAYS[trip.weekday]}, {trip.hour}:00 |
| Model version | {version} |
"""
print(result)

summary_file = os.environ.get("GITHUB_STEP_SUMMARY")  # set by GitHub Actions
if summary_file:
    with open(summary_file, "a") as summary:
        summary.write(result)
