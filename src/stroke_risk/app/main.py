import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import gradio as gr
from fastapi import FastAPI, HTTPException, Request

from stroke_risk.app.schemas import Patient
from stroke_risk.serving.inference import load_model, predict
from stroke_risk.app.gradio import build_demo

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion model into app state once at startup, if one exists."""
    try:
        app.state.model = load_model()
    except Exception:
        logger.exception(
            "No champion model available at startup. "
            "/predict will 503 until one is trained and promoted."
        )
        app.state.model = None
    yield


app = FastAPI(
    title="Stroke Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def get_health_status() -> dict[str, str]:
    """Report service liveness for health checks."""
    return {"status": "ok"}


@app.post("/predict")
def predict_stroke(patient: Patient, request: Request) -> dict:
    """Score a patient using the model loaded at startup."""
    if request.app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Train and promote a model first.",
        )
    result = predict(patient.model_dump(), request.app.state.model)
    return result


gr.mount_gradio_app(app, build_demo(app), path="/gradio")
