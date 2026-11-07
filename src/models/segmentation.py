"""Unsupervised customer segmentation using RFM (Recency, Frequency, Monetary) and K-Means.

Groups customers into distinct behavioral segments (Champions, At-Risk, Regular, New)
to enable targeted marketing, retention strategies, and segment-level lifetime value analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def compute_rfm_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary (RFM) metrics per customer.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cleaned transactions DataFrame containing 'Customer_Name', 'Timestamp',
        'Transaction_ID', and 'Total_Cost'.

    Returns
    -------
    pd.DataFrame
        Index: 'Customer_Name', Columns: ['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']
    """
    reference_date = transactions["Timestamp"].max()

    rfm = (
        transactions.groupby("Customer_Name")
        .agg(
            Last_Purchase=("Timestamp", "max"),
            Frequency=("Transaction_ID", "count"),
            Monetary=("Total_Cost", "sum"),
            Avg_Order_Value=("Total_Cost", "mean"),
            Total_Items=("Total_Items", "sum"),
        )
        .reset_index()
    )

    rfm["Recency"] = (reference_date - rfm["Last_Purchase"]).dt.total_seconds() / 86400.0
    rfm["Recency"] = rfm["Recency"].round(1)
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["Avg_Order_Value"] = rfm["Avg_Order_Value"].round(2)

    return rfm


def _assign_segment_labels(cluster_centers: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """Map raw K-Means cluster IDs to human-readable business personas based on centroids.

    Higher Monetary & Frequency + Lower Recency = Champions.
    High Recency + Moderate/High Monetary = At-Risk.
    Low Recency + Low Frequency = New Customers.
    Balanced = Regular Shoppers.
    """
    r_idx = feature_names.index("Recency")
    f_idx = feature_names.index("Frequency")
    m_idx = feature_names.index("Monetary")

    k = len(cluster_centers)
    scores = []
    for c_id, center in enumerate(cluster_centers):
        # Value score: higher F and M is better, lower R is better
        value_score = center[m_idx] + center[f_idx] - center[r_idx]
        scores.append((c_id, value_score, center[r_idx], center[f_idx], center[m_idx]))

    # Rank clusters by overall value score
    sorted_by_val = sorted(scores, key=lambda x: x[1], reverse=True)

    labels: dict[int, str] = {}
    assigned_names: set[str] = set()

    for rank, (c_id, val_score, rec, freq, mon) in enumerate(sorted_by_val):
        if rank == 0:
            name = "Champions / High Value"
        elif rec > 0.3 and (freq > -0.2 or mon > -0.2):
            name = "At Risk / Churning"
        elif rec < 0.0 and freq <= 0.0:
            name = "New / Promising"
        else:
            name = f"Regular Shoppers (Tier {rank})"

        # Guarantee unique persona names
        orig = name
        counter = 2
        while name in assigned_names:
            name = f"{orig} #{counter}"
            counter += 1

        labels[c_id] = name
        assigned_names.add(name)

    return labels


def perform_customer_segmentation(
    transactions: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    max_customers: int = 15000,
) -> dict[str, Any]:
    """Segment customers into behavioral clusters using RFM and K-Means.

    Parameters
    ----------
    transactions : pd.DataFrame
        Transactions history.
    n_clusters : int
        Number of clusters to partition into (default 4).
    random_state : int
        Random seed for reproducibility.
    max_customers : int
        Subsample cap for clustering if customer count is extremely large.

    Returns
    -------
    dict
        Contains:
        - 'rfm_df': pd.DataFrame with Customer_Name, RFM metrics, Cluster, Segment, PCA1, PCA2
        - 'segment_summary': aggregate stats (count, avg spend, avg frequency) per segment
        - 'silhouette_score': float quality metric
        - 'cluster_labels': dict mapping cluster ID -> persona name
    """
    rfm = compute_rfm_features(transactions)

    if len(rfm) > max_customers:
        rfm_sample = rfm.sample(n=max_customers, random_state=random_state).reset_index(drop=True)
    else:
        rfm_sample = rfm.copy()

    features_to_cluster = ["Recency", "Frequency", "Monetary"]
    X = rfm_sample[features_to_cluster].copy()

    # Log-transform positive heavily skewed values
    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    rfm_sample["Cluster"] = cluster_ids

    # Assign personas
    cluster_labels = _assign_segment_labels(kmeans.cluster_centers_, features_to_cluster)
    rfm_sample["Segment"] = rfm_sample["Cluster"].map(cluster_labels)

    # 2D PCA for visual mapping
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    rfm_sample["PCA1"] = pca_coords[:, 0].round(3)
    rfm_sample["PCA2"] = pca_coords[:, 1].round(3)

    # Silhouette score (compute on at most 2,000 samples for sub-second performance)
    sil_sample_size = min(2000, len(X_scaled))
    try:
        sil_score = float(
            silhouette_score(
                X_scaled[:sil_sample_size],
                cluster_ids[:sil_sample_size],
            )
        )
    except Exception:
        sil_score = 0.0

    # Aggregate Segment Profiles
    summary = (
        rfm_sample.groupby("Segment")
        .agg(
            Customer_Count=("Customer_Name", "count"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Total_Spend=("Monetary", "mean"),
            Avg_Order_Value=("Avg_Order_Value", "mean"),
        )
        .reset_index()
    )
    summary["Avg_Recency_Days"] = summary["Avg_Recency_Days"].round(1)
    summary["Avg_Frequency"] = summary["Avg_Frequency"].round(2)
    summary["Avg_Total_Spend"] = summary["Avg_Total_Spend"].round(2)
    summary["Avg_Order_Value"] = summary["Avg_Order_Value"].round(2)

    return {
        "rfm_df": rfm_sample,
        "segment_summary": summary,
        "silhouette_score": round(sil_score, 3),
        "cluster_labels": cluster_labels,
    }
