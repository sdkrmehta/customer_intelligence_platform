"""Market Basket Analysis using association rule mining (Apriori / FP-Growth).

Discovers cross-selling patterns, complementary product affinities, and bundle
opportunities with Support, Confidence, and Lift metrics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def encode_baskets(baskets: list[list[str]]) -> pd.DataFrame:
    """Transform a list of product baskets into a boolean item incidence matrix.

    Parameters
    ----------
    baskets : list of list of str
        List where each element is a list of product names in a single transaction.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with one column per unique product.
    """
    clean_baskets = [[str(item).strip() for item in b if str(item).strip()] for b in baskets]
    clean_baskets = [b for b in clean_baskets if len(b) > 0]

    te = TransactionEncoder()
    te_ary = te.fit(clean_baskets).transform(clean_baskets)
    return pd.DataFrame(te_ary, columns=te.columns_)


def mine_association_rules(
    transactions: pd.DataFrame,
    min_support: float = 0.005,
    min_confidence: float = 0.1,
    min_lift: float = 1.05,
    algorithm: str = "fpgrowth",
    max_baskets: int = 30000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Mine high-affinity association rules from transaction baskets.

    Parameters
    ----------
    transactions : pd.DataFrame
        Transactions DataFrame containing 'Product_List'.
    min_support : float
        Minimum fraction of transactions that must contain the itemset.
    min_confidence : float
        Minimum P(Consequent | Antecedent).
    min_lift : float
        Minimum lift ratio (> 1 indicates positive affinity / complementary purchase).
    algorithm : str
        'fpgrowth' (recommended, faster) or 'apriori'.
    max_baskets : int
        Sample cap if dataset has hundreds of thousands of transactions.

    Returns
    -------
    pd.DataFrame
        Formatted rules with Antecedents, Consequents, Support, Confidence, Lift.
    """
    if "Product_List" not in transactions.columns:
        return pd.DataFrame()

    baskets_series = transactions["Product_List"]
    if len(baskets_series) > max_baskets:
        baskets_series = baskets_series.sample(n=max_baskets, random_state=random_state)

    baskets = baskets_series.tolist()
    # Filter multi-item baskets for meaningful association mining
    multi_item_baskets = [b for b in baskets if isinstance(b, list) and len(b) >= 2]

    if len(multi_item_baskets) < 5:
        # Fall back to all non-empty baskets if multi-item baskets are scarce
        multi_item_baskets = [b for b in baskets if isinstance(b, list) and len(b) >= 1]

    if not multi_item_baskets:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Affinity Insight"])

    encoded_df = encode_baskets(multi_item_baskets)

    try:
        if algorithm == "fpgrowth":
            frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
        else:
            frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            # Retry with lower support if initial threshold was too stringent
            frequent_itemsets = fpgrowth(encoded_df, min_support=max(0.001, min_support / 3), use_colnames=True)

        if frequent_itemsets.empty:
            return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Affinity Insight"])

        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=min_lift,
        )

        if rules.empty:
            return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Affinity Insight"])

        # Filter by minimum confidence
        rules = rules[rules["confidence"] >= min_confidence].copy()

        # Format rules into clean readable table
        formatted_rules: list[dict[str, Any]] = []
        for _, row in rules.iterrows():
            ant = ", ".join(sorted(list(row["antecedents"])))
            con = ", ".join(sorted(list(row["consequents"])))
            lift_val = round(float(row["lift"]), 2)
            conf_val = round(float(row["confidence"]), 3)
            supp_val = round(float(row["support"]), 4)

            insight = f"Shoppers who buy {ant} are {lift_val}x more likely to also purchase {con}."
            formatted_rules.append({
                "Antecedent": ant,
                "Consequent": con,
                "Support": supp_val,
                "Confidence": conf_val,
                "Lift": lift_val,
                "Affinity Insight": insight,
            })

        rules_df = pd.DataFrame(formatted_rules)
        return rules_df.sort_values(["Lift", "Confidence"], ascending=[False, False]).reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Affinity Insight"])


def get_basket_cross_sell(
    rules_df: pd.DataFrame,
    current_items: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend complementary cross-sell items based on mined association rules.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Table of association rules mined from :func:`mine_association_rules`.
    current_items : list of str
        Products currently in customer's basket or recent purchases.
    top_n : int
        Maximum number of cross-sell products to return.
    """
    if rules_df.empty or not current_items:
        return pd.DataFrame(columns=["Cross-Sell Product", "Lift Multiplier", "Confidence", "Why Recommended"])

    current_set = set(current_items)
    matches: list[dict[str, Any]] = []

    for _, rule in rules_df.iterrows():
        ant_items = {item.strip() for item in rule["Antecedent"].split(",")}
        con_items = {item.strip() for item in rule["Consequent"].split(",")}

        # If customer has all or part of the antecedent, and hasn't bought consequent
        if ant_items.intersection(current_set) and not con_items.issubset(current_set):
            for con in con_items:
                if con not in current_set:
                    matches.append({
                        "Cross-Sell Product": con,
                        "Lift Multiplier": rule["Lift"],
                        "Confidence": rule["Confidence"],
                        "Why Recommended": f"Strong association with {rule['Antecedent']} (Lift: {rule['Lift']}x)",
                    })

    if not matches:
        return pd.DataFrame(columns=["Cross-Sell Product", "Lift Multiplier", "Confidence", "Why Recommended"])

    df = pd.DataFrame(matches).drop_duplicates(subset=["Cross-Sell Product"])
    return df.sort_values(["Lift Multiplier", "Confidence"], ascending=[False, False]).head(top_n).reset_index(drop=True)
