"""FastAPI service that serves the NumPy spiral classifier."""
import math
from typing import Annotated

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from api.model import SpiralModel

MAX_POINTS = 1000

model = SpiralModel()

app = FastAPI(
    title="Spiral Classifier API",
    description=(
        "Serves a neural network built from scratch in NumPy "
        "(no PyTorch/TensorFlow), trained on a 3-class spiral dataset."
    ),
    version="1.0.0",
)

Point = Annotated[list[float], Field(min_length=2, max_length=2, examples=[[0.3, -0.5]])]


class PredictRequest(BaseModel):
    points: list[Point] = Field(min_length=1, max_length=MAX_POINTS)

    @field_validator("points")
    @classmethod
    def finite_values(cls, points):
        if not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("coordinates must be finite numbers")
        return points


class Prediction(BaseModel):
    point: list[float]
    predicted_class: int
    probabilities: list[float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@app.get("/")
def root():
    return {
        "service": app.title,
        "docs": "/docs",
        "endpoints": ["GET /health", "GET /model-info", "POST /predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return model.metrics


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    probabilities = model.predict_proba(request.points)
    classes = np.argmax(probabilities, axis=1)
    return {
        "predictions": [
            {
                "point": point,
                "predicted_class": int(cls),
                "probabilities": [round(float(p), 6) for p in probs],
            }
            for point, cls, probs in zip(request.points, classes, probabilities)
        ]
    }
