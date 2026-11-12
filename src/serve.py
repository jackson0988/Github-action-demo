from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the exported model once, when the API starts
model = mlflow.sklearn.load_model("model")
version = Path("model/version.txt").read_text()

app = FastAPI(title="Taxi trip duration API")


# The API rejects a request if any value is missing or out of range.
# The fields are in the same order as the model's features
class Trip(BaseModel):
    PULocationID: int = Field(ge=1, le=265, description="Pickup zone")
    DOLocationID: int = Field(ge=1, le=265, description="Drop-off zone")
    trip_distance: float = Field(ge=0.1, le=50, description="Miles")
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6, description="0 is Monday")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": version}


@app.post("/predict")
def predict(trip: Trip):
    minutes = model.predict(pd.DataFrame([trip.model_dump()]))[0]
    return {"minutes": round(float(minutes), 1), "model_version": version}
