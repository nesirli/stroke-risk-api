import pandas as pd
from pathlib import Path
from stroke_risk_api.config import settings

def load_stroke_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load stroke.csv data and split X/y"""
    path = Path(settings.data_dir) / "raw" / "stroke.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found."
        )

    df = pd.read_csv(path)
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    return X, y