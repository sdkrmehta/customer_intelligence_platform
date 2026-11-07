"""Tests for the hybrid product recommendation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.recommender import recommend_products


def test_recommend_products_structure(sample_transactions: pd.DataFrame):
    """Verify recommendation output schema, non-empty results, and score ordering."""
    recs = recommend_products(sample_transactions, "Alice", top_n=5)

    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 5
    assert len(recs) > 0

    expected_cols = ["Recommended product", "Recommendation score", "Why it is recommended"]
    for col in expected_cols:
        assert col in recs.columns

    # Verify recommendations are sorted in descending order of score
    scores = recs["Recommendation score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommend_products_respects_top_n(sample_transactions: pd.DataFrame):
    """Verify top_n parameter constrains the number of returned recommendations."""
    recs_3 = recommend_products(sample_transactions, "Alice", top_n=3)
    assert len(recs_3) <= 3

    recs_1 = recommend_products(sample_transactions, "Bob", top_n=1)
    assert len(recs_1) <= 1


def test_recommend_products_unknown_customer(sample_transactions: pd.DataFrame):
    """Verify recommending for a non-existent customer raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        recommend_products(sample_transactions, "GhostCustomer")
