"""Tests for MLflow experiment tracking and run logging."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.tracking import get_experiment_runs, get_tracking_dir, log_training_run


def test_get_tracking_dir():
    """Verify MLflow local tracking directory exists and is created."""
    t_dir = get_tracking_dir()
    assert t_dir.exists()
    assert t_dir.name == "mlruns"


def test_log_training_run_mock():
    """Verify log_training_run logs parameters and metrics without exceptions."""
    mock_artifact = {
        "version": "2.0",
        "trained_at": "2026-11-05 20:00 UTC",
        "horizon_days": 90,
        "model_name": "logistic_regression",
        "quality": {
            "input_rows": 1000,
            "valid_rows": 1000,
            "unique_customers": 500,
        },
        "training_summary": {
            "model_type": "Logistic Regression",
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
            "roc_auc": 0.88,
        },
        "clv_summary": {
            "model_type": "CLV Regressor (Ridge)",
            "mae": 12.5,
            "rmse": 18.2,
            "r2_score": 0.45,
        },
    }

    run_id = log_training_run(
        mock_artifact,
        experiment_name="test-pytest-experiments",
        run_name="pytest_unit_test_run",
    )

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 5

    # Verify run can be retrieved
    runs_df = get_experiment_runs("test-pytest-experiments")
    assert isinstance(runs_df, pd.DataFrame)
    assert not runs_df.empty
    assert "metrics.accuracy" in runs_df.columns
    assert "metrics.f1_score" in runs_df.columns
