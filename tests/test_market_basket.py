"""Tests for Market Basket Analysis, frequent itemsets, and association rules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.market_basket import (
    encode_baskets,
    get_basket_cross_sell,
    mine_association_rules,
)


def test_encode_baskets():
    """Verify transaction encoding produces boolean incidence matrix."""
    baskets = [
        ["Milk", "Bread"],
        ["Bread", "Butter"],
        ["Milk", "Bread", "Butter"],
    ]
    df = encode_baskets(baskets)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"Milk", "Bread", "Butter"}
    # Milk is in row 0 and 2
    assert df.loc[0, "Milk"]
    assert not df.loc[1, "Milk"]
    assert df.loc[2, "Milk"]


def test_mine_association_rules(sample_transactions: pd.DataFrame):
    """Verify association rule mining extracts valid rules with positive lift."""
    rules = mine_association_rules(
        sample_transactions,
        min_support=0.01,
        min_confidence=0.05,
        min_lift=1.0,
    )
    assert isinstance(rules, pd.DataFrame)
    expected_cols = ["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Affinity Insight"]
    for col in expected_cols:
        assert col in rules.columns

    if not rules.empty:
        # Check rule metric invariants
        assert (rules["Lift"] >= 1.0).all()
        assert (rules["Confidence"] > 0.0).all()
        assert (rules["Support"] > 0.0).all()


def test_get_basket_cross_sell():
    """Verify cross-sell recommendations match antecedents and rank by lift."""
    mock_rules = pd.DataFrame([
        {
            "Antecedent": "Coffee",
            "Consequent": "Donut",
            "Support": 0.05,
            "Confidence": 0.6,
            "Lift": 3.2,
            "Affinity Insight": "Mock insight",
        },
        {
            "Antecedent": "Coffee",
            "Consequent": "Bagel",
            "Support": 0.03,
            "Confidence": 0.4,
            "Lift": 2.1,
            "Affinity Insight": "Mock insight",
        },
    ])

    recs = get_basket_cross_sell(mock_rules, current_items=["Coffee"], top_n=5)
    assert len(recs) == 2
    assert recs.iloc[0]["Cross-Sell Product"] == "Donut"
    assert recs.iloc[0]["Lift Multiplier"] == 3.2

    # If customer already bought Donut, don't recommend Donut again
    recs_existing = get_basket_cross_sell(mock_rules, current_items=["Coffee", "Donut"], top_n=5)
    assert len(recs_existing) == 1
    assert recs_existing.iloc[0]["Cross-Sell Product"] == "Bagel"
