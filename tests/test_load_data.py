from stroke_risk.ingest.load_data import load_stroke_data
import pandas as pd


def test_load_stroke_data():
    X, y = load_stroke_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 5110
    assert X.shape[1] == 10
    assert len(y) == 5110
