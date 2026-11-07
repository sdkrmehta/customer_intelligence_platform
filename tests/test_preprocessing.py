"""Tests for feature engineering, label creation, and data transformations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DataQualityError
from src.data.preprocessing import (
    add_history_features,
    build_training_frame,
    dashboard_data,
    temporal_train_test_split,
)


def test_add_history_features(sample_transactions: pd.DataFrame):
    """Verify history-aware feature calculations are mathematically correct."""
    df_feat = add_history_features(sample_transactions)

    # Check that required history columns are created
    expected_cols = [
        "Transactions_So_Far",
        "Average_Spend_So_Far",
        "Average_Items_So_Far",
        "Discount_Rate_So_Far",
        "Days_Since_Previous_Purchase",
        "Unique_Products_So_Far",
    ]
    for col in expected_cols:
        assert col in df_feat.columns

    # Alice has 4 transactions in sample data (sorted by timestamp)
    alice_rows = df_feat[df_feat["Customer_Name"] == "Alice"].reset_index(drop=True)
    assert len(alice_rows) == 4
    assert list(alice_rows["Transactions_So_Far"]) == [1, 2, 3, 4]
    # First purchase has no previous purchase
    assert pd.isna(alice_rows["Days_Since_Previous_Purchase"].iloc[0])
    # Second purchase was 14 days later (01-01 to 15-01)
    assert alice_rows["Days_Since_Previous_Purchase"].iloc[1] == pytest.approx(14.0625, rel=1e-2)
    # Average spend expands cumulatively: [20.0, (20+35)/2, (20+35+18)/3, ...]
    assert alice_rows["Average_Spend_So_Far"].iloc[0] == 20.0
    assert alice_rows["Average_Spend_So_Far"].iloc[1] == 27.5


def test_build_training_frame(sample_transactions: pd.DataFrame):
    """Verify horizon labelling creates binary target and filters unobservable tail."""
    # Use 90 days horizon
    training = build_training_frame(sample_transactions, horizon_days=90)
    assert "Will_Purchase_In_Horizon" in training.columns
    # Binary classes only
    assert set(training["Will_Purchase_In_Horizon"].unique()).issubset({0, 1})
    # Target should not be entirely empty
    assert len(training) > 0


def test_build_training_frame_invalid_horizon(sample_transactions: pd.DataFrame):
    """Verify horizon under 30 days is rejected with ValueError."""
    with pytest.raises(ValueError, match="at least 30 days"):
        build_training_frame(sample_transactions, horizon_days=10)


def test_temporal_train_test_split(sample_transactions: pd.DataFrame):
    """Verify temporal train/test split partitions chronological sequences."""
    training = build_training_frame(sample_transactions, horizon_days=90)
    train_df, test_df = temporal_train_test_split(training, test_ratio=0.3)

    assert len(train_df) > 0
    assert len(test_df) > 0
    # Every timestamp in train must be strictly earlier than or equal to the earliest in test
    assert train_df["Timestamp"].max() <= test_df["Timestamp"].min()


def test_dashboard_data(sample_transactions: pd.DataFrame):
    """Verify dashboard aggregations produce monthly, city, and product tables."""
    data = dashboard_data(sample_transactions)
    assert "monthly" in data
    assert "city" in data
    assert "products" in data

    assert "Revenue" in data["monthly"].columns
    assert "Revenue" in data["city"].columns
    assert "Times purchased" in data["products"].columns
    assert len(data["products"]) > 0
