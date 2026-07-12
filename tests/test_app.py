import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stroke_risk.app import main as app_main

VALID_PATIENT = {
    "id": 1,
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "gender": "male",
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "formerly_smoked",
}


class StubModel:
    def predict(self, X: pd.DataFrame):
        return [1]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_main, "load_model", lambda: StubModel())
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_for_valid_patient(client):
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    assert response.json() == {"patient_id": 1, "prediction": "High Risk of Stroke"}


def test_predict_rejects_invalid_patient(client):
    response = client.post("/predict", json={**VALID_PATIENT, "gender": "invalid"})

    assert response.status_code == 422
