import numpy as np
import pandas as pd

from stroke_risk.features.build_features import CAT_COLS, NUM_COLS, build_preprocessor


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [45.0, 60.0, None],
        "hypertension": [0, 1, 0],
        "heart_disease": [0, 0, 1],
        "avg_glucose_level": [100.0, 150.0, 120.0],
        "bmi": [25.0, None, 30.0],
        "gender": ["male", "female", "male"],
        "ever_married": ["yes", "no", "yes"],
        "work_type": ["private", "self_employed", "private"],
        "residence_type": ["urban", "rural", "urban"],
        "smoking_status": ["never_smoked", "smokes", "unknown"],
    })


def test_build_preprocessor_uses_expected_columns():
    assert NUM_COLS == ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    assert CAT_COLS == ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]


def test_build_preprocessor_transforms_all_rows_without_nans():
    X = _sample_frame()
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert dense.shape[0] == len(X)
    assert not np.isnan(dense).any()


def test_build_preprocessor_handles_unseen_category_at_transform_time():
    X = _sample_frame()
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen = X.copy()
    unseen.loc[0, "gender"] = "nonbinary"

    transformed = preprocessor.transform(unseen)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert dense.shape[0] == len(unseen)
