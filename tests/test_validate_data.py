import pandas as pd
import pytest

from stroke_risk.utils.validate_data import validate_stroke_data


def _valid_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({
        "gender": ["male", "female"],
        "age": [45.0, 60.0],
        "hypertension": [0, 1],
        "heart_disease": [0, 0],
        "ever_married": ["yes", "no"],
        "work_type": ["private", "self_employed"],
        "residence_type": ["urban", "rural"],
        "avg_glucose_level": [100.0, 150.0],
        "bmi": [25.0, 28.0],
        "smoking_status": ["never_smoked", "smokes"],
    })
    y = pd.Series([0, 1], name="stroke")
    return X, y


def test_valid_data_passes_validation():
    X, y = _valid_data()

    validate_stroke_data(X, y)


def test_out_of_range_age_raises():
    X, y = _valid_data()
    X.loc[0, "age"] = 200.0

    with pytest.raises(ValueError):
        validate_stroke_data(X, y)


def test_unknown_categorical_value_raises():
    X, y = _valid_data()
    X.loc[0, "gender"] = "nonbinary"

    with pytest.raises(ValueError):
        validate_stroke_data(X, y)


def test_missing_required_column_raises():
    X, y = _valid_data()
    X = X.drop(columns=["bmi"])

    with pytest.raises(ValueError):
        validate_stroke_data(X, y)
