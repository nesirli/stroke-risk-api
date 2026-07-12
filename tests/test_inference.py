import pandas as pd

from stroke_risk.serving.inference import predict

PATIENT = {
    "id": 7,
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


class RecordingModel:
    def __init__(self, prediction):
        self._prediction = prediction
        self.received: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame):
        self.received = X
        return [self._prediction]


def test_predict_high_risk():
    model = RecordingModel(prediction=1)

    result = predict(dict(PATIENT), model)

    assert result == {"patient_id": 7, "prediction": "High Risk of Stroke"}


def test_predict_low_risk():
    model = RecordingModel(prediction=0)

    result = predict(dict(PATIENT), model)

    assert result == {"patient_id": 7, "prediction": "Low Risk of Stroke"}


def test_predict_strips_id_before_scoring():
    model = RecordingModel(prediction=0)

    predict(dict(PATIENT), model)

    assert "id" not in model.received.columns
    assert model.received.shape[0] == 1
