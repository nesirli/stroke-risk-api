from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stroke_risk.models import promote as promote_module


def test_promote_best_model_registers_top_run(monkeypatch):
    runs = pd.DataFrame({"run_id": ["run-1"], "metrics.auc": [0.9]})
    monkeypatch.setattr(promote_module.mlflow, "set_tracking_uri", MagicMock())
    search_runs_mock = MagicMock(return_value=runs)
    monkeypatch.setattr(promote_module.mlflow, "search_runs", search_runs_mock)

    model_version = SimpleNamespace(version="3")
    register_model_mock = MagicMock(return_value=model_version)
    monkeypatch.setattr(promote_module.mlflow, "register_model", register_model_mock)

    client_mock = MagicMock()
    monkeypatch.setattr(promote_module.mlflow, "MlflowClient", MagicMock(return_value=client_mock))

    promote_module.promote_best_model()

    search_runs_mock.assert_called_once_with(
        experiment_names=[promote_module.settings.mlflow_experiment],
        order_by=["metrics.auc DESC"],
        max_results=1,
    )
    register_model_mock.assert_called_once_with(
        model_uri="runs:/run-1/model",
        name=promote_module.settings.mlflow_model_name,
    )
    client_mock.set_registered_model_alias.assert_called_once_with(
        name=promote_module.settings.mlflow_model_name,
        alias=promote_module.settings.mlflow_model_alias,
        version="3",
    )


def test_promote_best_model_raises_when_no_runs(monkeypatch):
    monkeypatch.setattr(promote_module.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(promote_module.mlflow, "search_runs", MagicMock(return_value=pd.DataFrame()))

    with pytest.raises(ValueError):
        promote_module.promote_best_model()
