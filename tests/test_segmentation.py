"""Tests for RFM customer segmentation and K-Means clustering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.segmentation import compute_rfm_features, perform_customer_segmentation


def test_compute_rfm_features(sample_transactions: pd.DataFrame):
    """Verify RFM calculation calculates correct Recency, Frequency, and Monetary."""
    rfm = compute_rfm_features(sample_transactions)

    assert isinstance(rfm, pd.DataFrame)
    assert len(rfm) == sample_transactions["Customer_Name"].nunique()

    expected_cols = ["Customer_Name", "Recency", "Frequency", "Monetary", "Avg_Order_Value"]
    for col in expected_cols:
        assert col in rfm.columns

    # Alice has 4 transactions, total spend 20 + 35 + 18 + 25 = 98.0
    alice = rfm[rfm["Customer_Name"] == "Alice"].iloc[0]
    assert alice["Frequency"] == 4
    assert alice["Monetary"] == pytest.approx(98.0)
    assert alice["Avg_Order_Value"] == pytest.approx(24.5)


def test_perform_customer_segmentation(sample_transactions: pd.DataFrame):
    """Verify K-Means clustering partitions customers and assigns personas."""
    result = perform_customer_segmentation(sample_transactions, n_clusters=3, random_state=42)

    assert "rfm_df" in result
    assert "segment_summary" in result
    assert "cluster_labels" in result
    assert "silhouette_score" in result

    rfm_df = result["rfm_df"]
    assert "Cluster" in rfm_df.columns
    assert "Segment" in rfm_df.columns
    assert "PCA1" in rfm_df.columns
    assert "PCA2" in rfm_df.columns

    # Check segment summary
    summary = result["segment_summary"]
    assert "Segment" in summary.columns
    assert "Customer_Count" in summary.columns
    assert summary["Customer_Count"].sum() == len(rfm_df)
