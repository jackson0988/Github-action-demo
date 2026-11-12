"""Train a new model and promote it only if it beats the champion.

Usage: python src/retrain.py 2020-04
Try:   python src/retrain.py 2020-04 --window 1
       python src/retrain.py 2020-02 --max-depth 2
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd

from common import MODEL_NAME, connect, error_of, load_months, register, train_model

parser = argparse.ArgumentParser()
parser.add_argument("month", help="the newest month of data, for example 2020-04")
parser.add_argument("--window", type=int, default=3, help="how many recent months to train on")
parser.add_argument("--max-depth", type=int, help="default: the same setting as the champion")
args = parser.parse_args()

client = connect()
current = client.get_model_version_by_alias(MODEL_NAME, "champion")
champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
max_depth = args.max_depth or int(mlflow.get_run(current.run_id).data.params["max_depth"])

newest = pd.Period(args.month, freq="M")
train_months = [str(newest - i) for i in range(args.window - 1, -1, -1)]
trips = load_months(train_months)

# Hold back the final week. Both models are judged on trips the new one never saw
cutoff = trips["pickup"].max() - pd.Timedelta(days=7)
train_trips = trips[trips["pickup"] < cutoff]
test_trips = trips[trips["pickup"] >= cutoff]

with mlflow.start_run(run_name=f"retrain-{args.month}"):
    candidate = train_model(train_trips, max_depth)
    champion_error = error_of(champion, test_trips)
    candidate_error = error_of(candidate, test_trips)

    mlflow.log_params({"max_depth": max_depth, "train_months": ",".join(train_months)})
    mlflow.log_metrics({"champion_error": champion_error, "candidate_error": candidate_error})

    print(f"Training on {train_months[0]} to {train_months[-1]}, max_depth {max_depth}")
    print(f"Final week: champion error {champion_error:.2f}, candidate error {candidate_error:.2f}")

    if candidate_error < champion_error:
        version = register(candidate, train_trips)
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        print(f"PROMOTED: version {version} is the new champion")
    else:
        print(f"REJECTED: version {current.version} stays the champion")
