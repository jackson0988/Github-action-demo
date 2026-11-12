"""Download the champion to the model/ folder, ready to go into the Docker image."""
import shutil
from pathlib import Path

import mlflow

from common import MODEL_NAME, TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

# Remove the old export first, so no old model files end up in the new image
shutil.rmtree("model", ignore_errors=True)

version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@champion", dst_path="model")
Path("model/version.txt").write_text(version)

print(f"Exported version {version} to model/")
