import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from stroke_risk.models.model import build_log_reg_pipeline


def _toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "age": [25, 40, 60, 70, 30, 50, 65, 45],
            "hypertension": [0, 0, 1, 1, 0, 1, 1, 0],
            "heart_disease": [0, 0, 0, 1, 0, 0, 1, 0],
            "avg_glucose_level": [90, 110, 130, 200, 95, 150, 180, 100],
            "bmi": [22, 25, 28, 31, 23, 27, 30, 24],
            "gender": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "ever_married": ["no", "yes", "yes", "yes", "no", "yes", "yes", "no"],
            "work_type": ["private"] * 8,
            "residence_type": ["urban", "rural"] * 4,
            "smoking_status": ["never_smoked"] * 8,
        }
    )
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 0])
    return X, y


def test_build_log_reg_pipeline_structure():
    pipeline = build_log_reg_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]
    assert pipeline.named_steps["model"].C == 1.0
    assert pipeline.named_steps["model"].penalty == "l2"


def test_build_log_reg_pipeline_applies_custom_hyperparams():
    pipeline = build_log_reg_pipeline(C=0.5, penalty="l1")

    assert pipeline.named_steps["model"].C == 0.5
    assert pipeline.named_steps["model"].penalty == "l1"


def test_pipeline_fits_and_predicts_on_toy_data():
    X, y = _toy_dataset()
    pipeline = build_log_reg_pipeline()

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
