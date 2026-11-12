"""Compare settings on validation data, then launch the best model as the champion."""
import mlflow

from common import MODEL_NAME, connect, error_of, load_months, register, train_model

MAX_DEPTHS = [5, 10, 15]
TRAIN_MONTHS = ["2019-07", "2019-08", "2019-09", "2019-10"]
VALIDATION_MONTHS = ["2019-11", "2019-12"]
LAUNCH_MONTHS = ["2019-10", "2019-11", "2019-12"]  # the newest data available at launch

client = connect()
train_trips = load_months(TRAIN_MONTHS)
validation_trips = load_months(VALIDATION_MONTHS)

with mlflow.start_run(run_name="train"):
    # Choose settings on data the model did not train on. 2020 stays unseen
    results = {}
    for max_depth in MAX_DEPTHS:
        with mlflow.start_run(run_name=f"max_depth={max_depth}", nested=True):
            model = train_model(train_trips, max_depth)
            results[max_depth] = error_of(model, validation_trips)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("validation_mae", results[max_depth])
        print(f"max_depth {max_depth:>2}  validation error {results[max_depth]:.2f} minutes")

    best_depth = min(results, key=results.get)

    # Train the winner again on the newest months. This is the model we launch
    with mlflow.start_run(run_name="launch", nested=True):
        launch_trips = load_months(LAUNCH_MONTHS)
        model = train_model(launch_trips, best_depth)
        mlflow.log_params({"max_depth": best_depth, "train_months": ",".join(LAUNCH_MONTHS)})
        version = register(model, launch_trips)

# 'launch' never moves, so we can always compare against the original model
client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.set_registered_model_alias(MODEL_NAME, "launch", version)
print(f"Launched version {version} with max_depth {best_depth}")
