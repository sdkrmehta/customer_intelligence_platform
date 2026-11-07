"""MLflow experiment tracking and model registry integration.

Logs hyperparameters, cross-validation metrics, artifacts (plots, model binaries),
and enables cross-experiment auditing and leaderboard comparisons.
Uses SQLite backend (sqlite:///mlflow.db) for reliable database storage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

# Allow file store as fallback if needed
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def get_tracking_uri() -> str:
    """Return local SQLite database URI for MLflow tracking."""
    db_path = Path(__file__).resolve().parent.parent.parent / "mlflow.db"
    # Forward slashes for sqlite URI compatibility on Windows
    return f"sqlite:///{db_path.as_posix()}"


def get_tracking_dir() -> Path:
    """Return local directory for MLflow tracking data."""
    tracking_dir = Path(__file__).resolve().parent.parent.parent / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir


def log_training_run(
    artifact: dict[str, Any],
    experiment_name: str = "retail-purchase-prediction",
    run_name: str | None = None,
) -> str | None:
    """Log an end-to-end model training run to MLflow.

    Parameters
    ----------
    artifact : dict
        Model artifact dictionary containing 'training_summary', 'model_name',
        'horizon_days', 'quality', and optional 'shap'/'clv_summary'.
    experiment_name : str
        MLflow experiment name.
    run_name : str, optional
        Custom run name; defaults to '{model_name}_h{horizon}d'.

    Returns
    -------
    str or None
        MLflow run_id if successful, None if logging failed.
    """
    try:
        import mlflow

        tracking_uri = get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        summary = artifact.get("training_summary", {})
        model_name = artifact.get("model_name", "model")
        horizon = artifact.get("horizon_days", 365)

        if not run_name:
            run_name = f"{model_name}_h{horizon}d"

        with mlflow.start_run(run_name=run_name) as run:
            # 1. Log Parameters
            mlflow.log_params({
                "model_key": model_name,
                "model_type": summary.get("model_type", model_name),
                "horizon_days": horizon,
                "input_rows": artifact.get("quality", {}).get("input_rows", 0),
                "valid_transactions": artifact.get("quality", {}).get("valid_rows", 0),
                "unique_customers": artifact.get("quality", {}).get("unique_customers", 0),
            })

            # 2. Log Holdout Metrics
            metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
            for m in metric_keys:
                val = summary.get(m)
                if val is not None and isinstance(val, (int, float)):
                    mlflow.log_metric(m, float(val))

            # 3. Log Cross-Validation Metrics if present
            if "cv" in summary:
                cv = summary["cv"]
                for m, val in cv.get("mean", {}).items():
                    if val is not None:
                        mlflow.log_metric(f"cv_mean_{m}", float(val))
                for m, val in cv.get("std", {}).items():
                    if val is not None:
                        mlflow.log_metric(f"cv_std_{m}", float(val))

            # 4. Log CLV Regressor Metrics if present
            if artifact.get("clv_summary"):
                clv = artifact["clv_summary"]
                mlflow.log_metric("clv_mae", float(clv["mae"]))
                mlflow.log_metric("clv_rmse", float(clv["rmse"]))
                mlflow.log_metric("clv_r2", float(clv["r2_score"]))

            # 5. Log SHAP Plots as Artifacts
            if artifact.get("shap"):
                shap_info = artifact["shap"]
                bar_plot = shap_info.get("bar_plot_path")
                if bar_plot and Path(bar_plot).exists():
                    mlflow.log_artifact(bar_plot, artifact_path="explainability")
                summary_plot = shap_info.get("summary_plot_path")
                if summary_plot and Path(summary_plot).exists():
                    mlflow.log_artifact(summary_plot, artifact_path="explainability")

            # 6. Tags
            mlflow.set_tags({
                "pipeline_version": artifact.get("version", "2.0"),
                "trained_at": artifact.get("trained_at", ""),
            })

            return run.info.run_id

    except Exception as e:
        print(f"[MLflow] Warning: Logging run to MLflow was skipped: {e}")
        return None


def get_experiment_runs(experiment_name: str = "retail-purchase-prediction") -> pd.DataFrame:
    """Retrieve historical logged runs from local MLflow for leaderboard analysis."""
    try:
        import mlflow

        tracking_uri = get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        return runs
    except Exception:
        return pd.DataFrame()
