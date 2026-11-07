"""Tests for Customer Lifetime Value (CLV) regression and future spend target."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import build_training_frame, compute_future_spend
from src.models.clv_predictor import (
    build_clv_pipeline,
    predict_customer_clv,
    train_clv_model,
)
from src.models.purchase_predictor import customer_feature_row


def test_compute_future_spend(sample_transactions: pd.DataFrame):
    """Verify forward spend calculation sums future purchases within horizon."""
    spends = compute_future_spend(sample_transactions, horizon_days=90)

    assert len(spends) == len(sample_transactions)
    assert (spends >= 0.0).all()
    # At least some repeat customers should have positive forward spend
    assert (spends > 0.0).any()


def test_build_clv_pipeline():
    """Verify CLV regression pipeline construction."""
    pipeline = build_clv_pipeline("ridge")
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps


def test_train_and_predict_clv(sample_transactions: pd.DataFrame):
    """Verify CLV model training, evaluation metrics, and customer spend inference."""
    training_df = build_training_frame(sample_transactions, horizon_days=90)
    assert "Future_Spend_In_Horizon" in training_df.columns

    pipeline, summary = train_clv_model(training_df, model_name="ridge")

    assert "mae" in summary
    assert "rmse" in summary
    assert "r2_score" in summary
    assert summary["mae"] >= 0.0
    assert summary["rmse"] >= 0.0

    # Predict for customer
    feature_row, _ = customer_feature_row(sample_transactions, "Alice")
    predicted_spend = predict_customer_clv(pipeline, feature_row)

    assert isinstance(predicted_spend, float)
    assert predicted_spend >= 0.0  # Spend cannot be negative
