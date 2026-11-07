"""Product recommendation engine using basket co-occurrence and popularity.

The score combines personal repeat behaviour, products bought with the latest
basket, contextual popularity, and overall product popularity. It deliberately
does not invent a product preference for customers with only one purchase.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from src.models.purchase_predictor import customer_feature_row
from src.utils.config import (
    REC_WEIGHT_ASSOCIATION,
    REC_WEIGHT_CONTEXT,
    REC_WEIGHT_PERSONAL,
    REC_WEIGHT_POPULARITY,
    RECOMMENDATION_TOP_N,
)


def _normalise(counter: Counter[str]) -> dict[str, float]:
    largest = max(counter.values(), default=0)
    return {key: value / largest for key, value in counter.items()} if largest else {}


def recommend_products(transactions: pd.DataFrame, customer_name: str, top_n: int = RECOMMENDATION_TOP_N) -> pd.DataFrame:
    """Create transparent hybrid recommendations from basket and customer history.

    The score combines personal repeat behaviour, products bought with the latest
    basket, contextual popularity, and overall product popularity. It deliberately
    does not invent a product preference for customers with only one purchase.
    """
    _, history = customer_feature_row(transactions, customer_name)
    latest = history.iloc[-1]
    product_popularity: Counter[str] = Counter()
    product_context: Counter[str] = Counter()
    co_occurrence: dict[str, Counter[str]] = defaultdict(Counter)

    for row in transactions.itertuples(index=False):
        basket = list(dict.fromkeys(row.Product_List))
        product_popularity.update(basket)
        if (
            row.City == latest.City
            and row.Store_Type == latest.Store_Type
            and row.Customer_Category == latest.Customer_Category
        ):
            product_context.update(basket)
        for product in basket:
            co_occurrence[product].update(other for other in basket if other != product)

    personal = Counter(product for basket in history["Product_List"] for product in basket)
    latest_basket = set(latest.Product_List)
    association: Counter[str] = Counter()
    for product in latest_basket:
        association.update(co_occurrence[product])

    popularity_score = _normalise(product_popularity)
    context_score = _normalise(product_context)
    personal_score = _normalise(personal)
    association_score = _normalise(association)
    candidates = set(product_popularity)
    rows: list[dict[str, Any]] = []
    for product in candidates:
        score = (
            REC_WEIGHT_POPULARITY * popularity_score.get(product, 0)
            + REC_WEIGHT_CONTEXT * context_score.get(product, 0)
            + REC_WEIGHT_PERSONAL * personal_score.get(product, 0)
            + REC_WEIGHT_ASSOCIATION * association_score.get(product, 0)
        )
        if product in personal and personal_score.get(product, 0) > 0:
            reason = "Likely repeat purchase based on this customer's history"
        elif association_score.get(product, 0) > 0:
            reason = "Often purchased with items in the customer's latest basket"
        elif context_score.get(product, 0) > 0:
            reason = "Popular with customers in a similar store and customer context"
        else:
            reason = "Popular product across the available transaction history"
        rows.append(
            {
                "Recommended product": product,
                "Recommendation score": round(float(score), 3),
                "Why it is recommended": reason,
            }
        )

    recommendations = pd.DataFrame(rows)
    return recommendations.sort_values(["Recommendation score", "Recommended product"], ascending=[False, True]).head(top_n)
