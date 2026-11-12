"""Model quality check. Trains with our current code and fails if the model is not good enough.

CI runs this on every push. It needs October and November 2019:
    python src/download_data.py 2019-10 2019-11
    python src/check_model.py
"""
import os
import shutil
import sys
from pathlib import Path

import mlflow.sklearn

from common import error_of, load_month, train_model

MAX_DEPTH = 15  # the setting train.py chose in Class 3
MAX_ERROR = 3.6  # minutes. Our model scores about 3.3, so this leaves some room

train_trips = load_month("2019-10")
test_trips = load_month("2019-11")

model = train_model(train_trips, MAX_DEPTH)
error = error_of(model, test_trips)
print(f"Trained on 2019-10, tested on 2019-11: error {error:.2f} minutes (limit {MAX_ERROR})")

if error > MAX_ERROR:
    print("FAILED: the model is worse than the limit")
    sys.exit(1)  # a non-zero exit code makes the CI job fail

# Save the model that passed, so CI can put exactly this model in the Docker image
shutil.rmtree("model", ignore_errors=True)
mlflow.sklearn.save_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
version = "ci-" + os.environ.get("GITHUB_SHA", "local")[:7]
Path("model/version.txt").write_text(version)
print(f"PASSED: saved the model to model/ as version {version}")
