"""Check the champion on new months. Usage: python src/monitor.py 2020-01 2020-02"""
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from common import ALL_MONTHS, FEATURES, MAX_MAE, MODEL_NAME, TARGET, connect, error_of, load_month, load_months

SAMPLE_ROWS = 20_000  # enough for a drift check, and much faster than every trip
COLUMNS = DataDefinition(
    numerical_columns=["trip_distance", "hour", "weekday"],
    categorical_columns=["PULocationID", "DOLocationID"],
)
REPORTS_DIR = Path("reports")


def sample(trips):
    rows = trips[FEATURES].sample(min(SAMPLE_ROWS, len(trips)), random_state=42)
    return Dataset.from_pandas(rows, data_definition=COLUMNS)


def check_drift(reference, trips, month):
    """Save an Evidently report and return how many input columns changed."""
    result = Report([DataDriftPreset()]).run(reference_data=sample(reference), current_data=sample(trips))
    REPORTS_DIR.mkdir(exist_ok=True)
    report_file = REPORTS_DIR / f"drift_{month}.html"
    result.save_html(str(report_file))
    mlflow.log_artifact(str(report_file), artifact_path="drift_reports")
    return int(result.dict()["metrics"][0]["value"]["count"])


months = sys.argv[1:]
client = connect()

champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
launch = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@launch")

# Compare new trips with the data the champion learned from
version = client.get_model_version_by_alias(MODEL_NAME, "champion")
train_months = mlflow.get_run(version.run_id).data.params["train_months"].split(",")
reference = load_months(train_months)

print(f"Champion: version {version.version}, trained on {train_months[0]} to {train_months[-1]}")
print("month    trips    error  launch-error  predicted/actual  drifted-columns")

with mlflow.start_run(run_name=f"monitor-v{version.version}"):
    mlflow.log_param("champion_version", version.version)

    for month in months:
        trips = load_month(month)
        error = error_of(champion, trips)
        launch_error = error_of(launch, trips)
        predicted_vs_actual = champion.predict(trips[FEATURES]).mean() / trips[TARGET].mean()
        drifted_columns = check_drift(reference, trips, month)

        mlflow.log_metrics(
            {
                "trips": len(trips),
                "error": error,
                "launch_error": launch_error,
                "predicted_vs_actual": predicted_vs_actual,
                "drifted_columns": drifted_columns,
            },
            step=ALL_MONTHS.index(month),
        )

        # Drift means "look into this". A high error means "retrain"
        status = "ok"
        if drifted_columns >= 3:
            status = "drift warning"
        if error > MAX_MAE:
            status = "ALARM: retrain"
        print(
            f"{month}  {len(trips):>7}  {error:>5.2f}  {launch_error:>12.2f}"
            f"  {predicted_vs_actual:>16.2f}  {drifted_columns:>15}  {status}"
        )
