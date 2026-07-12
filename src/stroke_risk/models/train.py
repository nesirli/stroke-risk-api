import json
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from stroke_risk.models.model import build_log_reg_pipeline
from stroke_risk.ingest.load_data import load_stroke_data
from stroke_risk.config import settings

trusted_types = ["numpy.dtype"]


def train_model(X: pd.DataFrame, y: pd.Series) -> None:
    """Train a logistic regression pipeline on the best known params and log it to MLflow."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    best_params = json.loads(settings.best_params_path.read_text())
    pipeline = build_log_reg_pipeline(**best_params)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        experiment_id = client.create_experiment(
            settings.mlflow_experiment,
            artifact_location=str(settings.mlflow_artifact_location),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params(best_params)
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(
            pipeline, name="model", skops_trusted_types=trusted_types
        )

        print(f"Model trained. AUC score: {auc:.4f}")


if __name__ == "__main__":
    X, y = load_stroke_data()
    train_model(X, y)
