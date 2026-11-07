"""Feature engineering and data transformation for model training.

Builds history-aware features from cleaned transaction data and prepares
labelled training frames with the purchase-within-horizon target.
Includes temporal train/test splitting for realistic evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import DataQualityError
from src.utils.config import FEATURE_COLUMNS


def add_history_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create features using information available by each transaction date only."""
    data = transactions.sort_values(["Customer_Name", "Timestamp", "Transaction_ID"], kind="stable").copy()
    grouped = data.groupby("Customer_Name", sort=False)

    data["Transactions_So_Far"] = grouped.cumcount() + 1
    data["Average_Spend_So_Far"] = grouped["Total_Cost"].transform(lambda series: series.expanding().mean())
    data["Average_Items_So_Far"] = grouped["Total_Items"].transform(lambda series: series.expanding().mean())
    data["Discount_Rate_So_Far"] = grouped["Discount_Flag"].transform(lambda series: series.expanding().mean())
    previous_timestamp = grouped["Timestamp"].shift(1)
    data["Days_Since_Previous_Purchase"] = (data["Timestamp"] - previous_timestamp).dt.total_seconds() / 86_400

    unique_counts: list[int] = []
    seen_products: set[str] = set()
    active_customer: str | None = None
    for row in data[["Customer_Name", "Product_List"]].itertuples(index=False):
        if row.Customer_Name != active_customer:
            active_customer = row.Customer_Name
            seen_products = set()
        seen_products.update(row.Product_List)
        unique_counts.append(len(seen_products))
    data["Unique_Products_So_Far"] = unique_counts
    return data


def compute_future_spend(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Compute total forward spend ($) for each transaction within the future horizon."""
    horizon_seconds = float(horizon_days * 86400.0)
    future_spend = np.zeros(len(df), dtype=np.float64)

    cust_col = df["Customer_Name"].values
    time_col = df["Timestamp"].values.astype("datetime64[s]").astype("int64")
    cost_col = df["Total_Cost"].values

    n = len(df)
    i = 0
    while i < n:
        cust = cust_col[i]
        start_idx = i
        while i < n and cust_col[i] == cust:
            i += 1
        end_idx = i

        if end_idx - start_idx > 1:
            times = time_col[start_idx:end_idx]
            costs = cost_col[start_idx:end_idx]
            m = end_idx - start_idx
            for p in range(m - 1):
                t_p = times[p]
                spend_sum = 0.0
                for q in range(p + 1, m):
                    if times[q] - t_p <= horizon_seconds:
                        spend_sum += costs[q]
                    else:
                        break
                future_spend[start_idx + p] = spend_sum

    return pd.Series(future_spend, index=df.index).round(2)


def build_training_frame(transactions: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Label a row with purchase probability and forward spend targets within horizon.

    To avoid claiming that an unobserved future is a negative, records within the
    final *horizon_days* of the dataset are excluded from supervised training.
    """
    if horizon_days < 30:
        raise ValueError("The prediction horizon must be at least 30 days.")

    data = add_history_features(transactions)
    next_purchase = data.groupby("Customer_Name", sort=False)["Timestamp"].shift(-1)
    data["Next_Purchase_Date"] = next_purchase
    data["Will_Purchase_In_Horizon"] = (
        ((next_purchase - data["Timestamp"]).dt.total_seconds() <= horizon_days * 86_400)
        & next_purchase.notna()
    ).astype(int)

    # Compute forward monetary spend (CLV target)
    data["Future_Spend_In_Horizon"] = compute_future_spend(data, horizon_days)

    observation_end = data["Timestamp"].max()
    label_cutoff = observation_end - pd.Timedelta(days=horizon_days)
    labelled = data.loc[data["Timestamp"] <= label_cutoff].copy()
    if labelled.empty:
        raise DataQualityError(
            "The selected prediction horizon leaves no labelled rows. Choose a shorter horizon or provide more history."
        )
    return labelled


def dashboard_data(transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare small, reusable aggregates for the dashboard charts."""
    monthly = (
        transactions.groupby("Transaction_Month", as_index=False)
        .agg(Revenue=("Total_Cost", "sum"), Transactions=("Transaction_ID", "count"))
        .sort_values("Transaction_Month")
    )
    city = (
        transactions.groupby("City", as_index=False)
        .agg(Revenue=("Total_Cost", "sum"), Transactions=("Transaction_ID", "count"))
        .sort_values("Revenue", ascending=False)
    )
    exploded = transactions[["Product_List"]].explode("Product_List").dropna()
    products = (
        exploded.groupby("Product_List", as_index=False)
        .size()
        .rename(columns={"Product_List": "Product", "size": "Times purchased"})
        .sort_values("Times purchased", ascending=False)
        .head(15)
    )
    return {"monthly": monthly, "city": city, "products": products}


def temporal_train_test_split(
    labelled: pd.DataFrame,
    test_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled data by date for realistic evaluation.

    Unlike a random split, this trains on older transactions and tests on
    newer ones — matching how a model would be used in production.

    Parameters
    ----------
    labelled : pd.DataFrame
        Output of :func:`build_training_frame` (must have ``Timestamp``).
    test_ratio : float
        Approximate fraction of rows to place in the test set (by date cutoff).

    Returns
    -------
    (train_df, test_df)
    """
    sorted_data = labelled.sort_values("Timestamp")
    split_idx = int(len(sorted_data) * (1 - test_ratio))
    cutoff_date = sorted_data.iloc[split_idx]["Timestamp"]

    train = sorted_data[sorted_data["Timestamp"] < cutoff_date].copy()
    test = sorted_data[sorted_data["Timestamp"] >= cutoff_date].copy()

    if train.empty or test.empty:
        raise DataQualityError(
            "Temporal split produced an empty train or test set. "
            "The dataset may not span enough time."
        )

    return train, test
