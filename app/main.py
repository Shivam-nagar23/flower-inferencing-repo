import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "iris_finetuned_model.pkl")

model = None

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: dict[str, float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Iris Inference Service",
    description="ML inference API for fine-tuned Iris classification model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    input_data = np.array(
        [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    )

    prediction_index = int(model.predict(input_data)[0])
    prediction = IRIS_CLASSES[prediction_index]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        probabilities = {IRIS_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return PredictionResponse(
        prediction=prediction,
        prediction_index=prediction_index,
        probabilities=probabilities,
    )
