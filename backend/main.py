"""FastAPI endpoints for the crop-yield prediction models."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.predictions import predict_all_models


logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BACKEND_DIR / "results" / "model_comparison.json"

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if origin.strip()
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    fertilizer: Annotated[float, Field(description="Fertilizer input")]
    temp: Annotated[float, Field(description="Temperature input")]
    N: Annotated[float, Field(description="Nitrogen input")]
    P: Annotated[float, Field(description="Phosphorus input")]
    K: Annotated[float, Field(description="Potassium input")]


class PredictionResponse(BaseModel):
    input: dict[str, float]
    predictions: dict[str, float]


app = FastAPI(
    title="Crop Yield Prediction API",
    description="Classical and quantum-integrated crop-yield model predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def get_models() -> dict[str, list[dict[str, str]]]:
    return {
        "models": [
            {
                "id": "linear_regression",
                "name": "Linear Regression",
                "type": "classical",
            },
            {
                "id": "linear_regression_quantum",
                "name": "Linear Regression + Quantum",
                "type": "quantum_integrated",
            },
            {
                "id": "random_forest",
                "name": "Random Forest",
                "type": "classical",
            },
            {
                "id": "random_forest_quantum",
                "name": "Random Forest + Quantum",
                "type": "quantum_integrated",
            },
        ]
    }


@app.get("/api/results")
def get_results() -> dict[str, object]:
    try:
        with RESULTS_PATH.open(encoding="utf-8") as results_file:
            results = json.load(results_file)
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("Could not load verified model comparison results.")
        raise HTTPException(
            status_code=500,
            detail="Model comparison results are currently unavailable.",
        ) from error

    if not isinstance(results, dict):
        logger.error("Model comparison results must contain a JSON object.")
        raise HTTPException(
            status_code=500,
            detail="Model comparison results are invalid.",
        )
    return results


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction_input = {
        "Fertilizer": request.fertilizer,
        "temp": request.temp,
        "N": request.N,
        "P": request.P,
        "K": request.K,
    }
    try:
        predictions = predict_all_models(prediction_input)
    except Exception as error:
        logger.exception("Model prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs for details.",
        ) from error

    return PredictionResponse(
        input={
            "fertilizer": request.fertilizer,
            "temp": request.temp,
            "N": request.N,
            "P": request.P,
            "K": request.K,
        },
        predictions=predictions,
    )
