"""Evaluation metric computation, cross-validation, and reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.utils.config import PROBABILITY_THRESHOLD, RANDOM_STATE


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels (after thresholding).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of metric name to float value.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }

    if pd.Series(y_true).nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


def cross_validate_model(
    model_builder,
    features: pd.DataFrame,
    target: pd.Series,
    n_folds: int = 5,
    probability_fn=None,
) -> dict[str, Any]:
    """Run stratified K-fold cross-validation and return per-fold + summary metrics.

    Parameters
    ----------
    model_builder : callable
        A zero-argument callable that returns a fresh, unfitted sklearn Pipeline.
    features : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Binary target vector.
    n_folds : int
        Number of CV folds (default 5).
    probability_fn : callable, optional
        Function ``(model, X) -> np.ndarray`` returning P(class=1).
        Defaults to extracting column 1 from ``predict_proba``.

    Returns
    -------
    dict with keys:
        ``fold_metrics``  – list of per-fold metric dicts
        ``mean``          – dict of mean metric values across folds
        ``std``           – dict of std metric values across folds
        ``n_folds``       – number of folds used
    """
    if probability_fn is None:
        def probability_fn(model, X):
            proba = model.predict_proba(X)
            classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
            idx = classes.index(1) if 1 in classes else -1
            return proba[:, idx] if idx >= 0 else np.zeros(len(X))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, target), start=1):
        x_train = features.iloc[train_idx]
        x_val = features.iloc[val_idx]
        y_train = target.iloc[train_idx]
        y_val = target.iloc[val_idx]

        model = model_builder()
        model.fit(x_train, y_train)

        probabilities = probability_fn(model, x_val)
        predictions = (probabilities >= PROBABILITY_THRESHOLD).astype(int)
        metrics = compute_classification_metrics(y_val, predictions, probabilities)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    # Aggregate across folds
    metric_names = [k for k in fold_metrics[0] if k != "fold"]
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for name in metric_names:
        values = [m[name] for m in fold_metrics if m[name] is not None]
        if values:
            mean_metrics[name] = float(np.mean(values))
            std_metrics[name] = float(np.std(values))
        else:
            mean_metrics[name] = None
            std_metrics[name] = None

    return {
        "fold_metrics": fold_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
        "n_folds": n_folds,
    }
