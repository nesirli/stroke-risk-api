from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from stroke_risk.app.schemas import Patient
from stroke_risk.serving.inference import load_model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title='Stroke Prediction API',
    version='0.1.0',
    lifespan=lifespan,
)

@app.get('/health')
def get_health_status():
    return {'status': 'ok'}

@app.post('/predict')
def predict_stroke(patient: Patient, request: Request):
    result = predict(patient.model_dump(), request.app.state.model)
    return result