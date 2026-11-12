"""Batch inference: predict every trip in a CSV file at once.

Usage: python src/batch_predict.py samples/trips.csv
The file needs the 5 model columns. If it also has actual_minutes, we show how far off each prediction was.
Writes predictions.csv. On GitHub, the table also goes to the run's summary page.
"""
import os
import sys

import pandas as pd
from pydantic import ValidationError

from common import FEATURES
from serve import Trip, model, version  # same model, checks and fields as the API

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

file = sys.argv[1] if len(sys.argv) > 1 else "samples/trips.csv"
trips = pd.read_csv(file)

missing = [column for column in FEATURES if column not in trips.columns]
if missing:
    sys.exit(f"{file} is missing these columns: {missing}")

# Check every row with the same rules as the API, and stop at the first bad one
for row_number, row in enumerate(trips[FEATURES].to_dict("records"), start=2):  # row 1 is the header
    try:
        Trip(**row)
    except ValidationError as error:
        sys.exit(f"Row {row_number} of {file} is not valid:\n{error}")

# One call predicts every trip
trips["predicted_minutes"] = model.predict(trips[FEATURES]).round(1)
has_actual = "actual_minutes" in trips.columns
if has_actual:
    trips["error_minutes"] = (trips["predicted_minutes"] - trips["actual_minutes"]).abs().round(1)

trips.to_csv("predictions.csv", index=False)

# Build a table that is easy to read
lines = [
    f"### Batch predictions: {len(trips)} trips from `{file}`",
    "",
    f"Model version: `{version}`",
    "",
    "| Pickup | Drop-off | Miles | When | Predicted | Actual | Off by |" if has_actual
    else "| Pickup | Drop-off | Miles | When | Predicted |",
    "|---|---|---|---|---|---|---|" if has_actual else "|---|---|---|---|---|",
]
for trip in trips.itertuples():
    row = (
        f"| {trip.PULocationID} | {trip.DOLocationID} | {trip.trip_distance} "
        f"| {DAYS[trip.weekday]} {trip.hour}:00 | {trip.predicted_minutes} min"
    )
    if has_actual:
        row += f" | {trip.actual_minutes} min | {trip.error_minutes} min"
    lines.append(row + " |")
if has_actual:
    lines += ["", f"**Average error: {trips['error_minutes'].mean():.2f} minutes**"]
result = "\n".join(lines) + "\n"

print(result)
print("Saved all predictions to predictions.csv")

summary_file = os.environ.get("GITHUB_STEP_SUMMARY")  # set by GitHub Actions
if summary_file:
    with open(summary_file, "a") as summary:
        summary.write(result)
