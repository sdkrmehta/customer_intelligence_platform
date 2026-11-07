"""Tests for model factory, training pipelines, evaluation metrics, and inference."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_classification_metrics
from src.models.model_factory import build_model_pipeline, get_display_name
from src.models.purchase_predictor import (
    ConstantProbabilityModel,
    customer_feature_row,
    predict_customer_purchase,
    train_and_save,
    train_purchase_model,
)
from src.utils.config import AVAILABLE_MODELS, FEATURE_COLUMNS


def test_model_factory_all_models():
    """Verify all configured models instantiate valid sklearn Pipelines."""
    for model_name in AVAILABLE_MODELS:
        pipeline = build_model_pipeline(model_name)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict_proba")
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps
        assert get_display_name(model_name) != ""


def test_model_factory_invalid_name():
    """Verify requesting an unsupported model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model_pipeline("non_existent_neural_net")


def test_compute_classification_metrics():
    """Verify classification metric calculations and bounds."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_constant_probability_fallback():
    """Verify fallback model behavior when data has insufficient class diversity."""
    fallback = ConstantProbabilityModel(0.75)
    dummy_X = pd.DataFrame({"feat": [1, 2, 3]})
    probs = fallback.predict_proba(dummy_X)
    assert probs.shape == (3, 2)
    assert np.allclose(probs[:, 1], 0.75)
    assert np.allclose(probs[:, 0], 0.25)


def test_customer_feature_row(sample_transactions: pd.DataFrame):
    """Verify feature row generation extracts the latest transaction features."""
    row_df, history = customer_feature_row(sample_transactions, "Alice")
    assert len(row_df) == 1
    assert set(FEATURE_COLUMNS).issubset(set(row_df.columns))
    assert len(history) == 4


def test_customer_feature_row_missing(sample_transactions: pd.DataFrame):
    """Verify searching for an unknown customer raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        customer_feature_row(sample_transactions, "NonExistentUser")


def test_end_to_end_train_and_predict(tmp_path: Path, sample_csv_text: str):
    """Verify training, saving, and customer prediction with SHAP enabled."""
    csv_file = tmp_path / "test_transactions.csv"
    csv_file.write_text(sample_csv_text, encoding="utf-8")
    model_output = tmp_path / "model.joblib"

    artifact = train_and_save(
        source=csv_file,
        artifact_path=model_output,
        horizon_days=90,
        model_name="logistic_regression",
        cv_folds=0,
        compute_shap=True,
    )

    assert model_output.exists()
    assert "model" in artifact
    assert "quality" in artifact
    assert "training_summary" in artifact

    # Test prediction
    result = predict_customer_purchase(artifact, "Alice")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["band"] in ("Low", "Medium", "High")
    assert result["transaction_count"] == 4
    assert result["total_spend"] > 0
