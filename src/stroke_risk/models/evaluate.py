import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from stroke_risk.ingest.load_data import load_stroke_data
from stroke_risk.config import settings

def evaluate_model(X: pd.DataFrame, y: pd.Series) -> None:
    """Evaluate the most recent MLflow run's model on a held-out test split."""
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["start_time DESC"],
        max_results=1
    )
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")


if __name__ == '__main__':
    X, y = load_stroke_data()
    evaluate_model(X, y)