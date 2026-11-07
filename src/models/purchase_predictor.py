"""Purchase prediction model: training, saving, loading, and inference.

Supports multiple classifiers (Logistic Regression, Random Forest, XGBoost,
LightGBM) selected via the ``model_name`` parameter. The model factory
handles construction; this module handles the training loop, evaluation,
artifact persistence, and customer-level inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_transactions
from src.data.preprocessing import build_training_frame
from src.evaluation.explainability import compute_shap_summary, explain_local_customer
from src.evaluation.metrics import compute_classification_metrics, cross_validate_model
from src.evaluation.tracking import log_training_run
from src.models.clv_predictor import predict_customer_clv, train_clv_model
from src.models.market_basket import mine_association_rules
from src.models.model_factory import build_model_pipeline, get_display_name
from src.models.segmentation import perform_customer_segmentation
from src.utils.config import (
    BAND_HIGH_THRESHOLD,
    BAND_MEDIUM_THRESHOLD,
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL,
    FEATURE_COLUMNS,
    LOW_CONFIDENCE_MIN_POSITIVES,
    PROBABILITY_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
)


class ConstantProbabilityModel:
    """Safe fallback used only when a dataset contains one target class."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.positive_probability)
        return np.column_stack([1 - positive, positive])


def _model_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return the probability of target class 1 for sklearn and fallback models."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    positive_index = classes.index(1) if 1 in classes else -1
    if positive_index == -1:
        return np.zeros(len(features))
    return probabilities[:, positive_index]


def train_purchase_model(
    transactions: pd.DataFrame,
    horizon_days: int = 365,
    model_name: str = DEFAULT_MODEL,
    cv_folds: int = 0,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Train a purchase-within-horizon classifier.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cleaned transaction data from :func:`load_transactions`.
    horizon_days : int
        Number of days defining the future purchase window.
    model_name : str
        Model key: ``logistic_regression``, ``random_forest``,
        ``xgboost``, or ``lightgbm``.
    cv_folds : int
        If > 1, run stratified K-fold cross-validation and include
        per-fold metrics in the summary. The final model is always
        retrained on the full train split for production use.
    """
    training = build_training_frame(transactions, horizon_days)
    features = training[FEATURE_COLUMNS]
    target = training["Will_Purchase_In_Horizon"]
    class_counts = target.value_counts().to_dict()

    summary: dict[str, Any] = {
        "labelled_transactions": int(len(training)),
        "positive_examples": int(class_counts.get(1, 0)),
        "negative_examples": int(class_counts.get(0, 0)),
        "positive_rate": float(target.mean()),
    }

    if target.nunique() < 2 or min(class_counts.values()) < 2:
        model: Any = ConstantProbabilityModel(float(target.mean()))
        summary.update(
            {
                "model_type": "Dataset-rate fallback",
                "evaluation_note": "Too few repeat-purchase examples for a reliable supervised classifier.",
                "is_low_confidence": True,
            }
        )
        return model, summary, training

    # --- Cross-validation (optional) ---
    if cv_folds > 1:
        cv_results = cross_validate_model(
            model_builder=lambda: build_model_pipeline(model_name),
            features=features,
            target=target,
            n_folds=cv_folds,
            probability_fn=_model_probability,
        )
        summary["cv"] = cv_results

    # --- Final holdout evaluation ---
    model = build_model_pipeline(model_name)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model.fit(x_train, y_train)
    probabilities = _model_probability(model, x_test)
    predictions = (probabilities >= PROBABILITY_THRESHOLD).astype(int)

    metrics = compute_classification_metrics(y_test, predictions, probabilities)

    eval_note = "Metrics calculated on a stratified 25% holdout set."
    if cv_folds > 1:
        cv_mean = summary["cv"]["mean"]
        eval_note += (
            f" {cv_folds}-fold CV mean: "
            f"Accuracy={cv_mean['accuracy']:.3f} +/-{summary['cv']['std']['accuracy']:.3f}, "
            f"F1={cv_mean['f1_score']:.3f} +/-{summary['cv']['std']['f1_score']:.3f}, "
            f"ROC-AUC={cv_mean.get('roc_auc', 0):.3f} +/-{summary['cv']['std'].get('roc_auc', 0):.3f}."
        )

    summary.update(
        {
            "model_type": get_display_name(model_name),
            "model_key": model_name,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            **metrics,
            "is_low_confidence": bool(class_counts.get(1, 0) < LOW_CONFIDENCE_MIN_POSITIVES),
            "evaluation_note": "Holdout metrics are indicative only; sparse repeat-customer history limits reliability."
            if class_counts.get(1, 0) < LOW_CONFIDENCE_MIN_POSITIVES
            else eval_note,
        }
    )
    return model, summary, training


def train_and_save(
    source: str | Path | BinaryIO | TextIO,
    artifact_path: str | Path,
    horizon_days: int = 365,
    model_name: str = DEFAULT_MODEL,
    cv_folds: int = 0,
    sample_frac: float | None = None,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train the model and write one reusable artifact for the web application."""
    transactions, quality = load_transactions(source, sample_frac=sample_frac)
    model, summary, training = train_purchase_model(
        transactions, horizon_days=horizon_days, model_name=model_name,
        cv_folds=cv_folds,
    )

    shap_results = None
    if compute_shap and not isinstance(model, ConstantProbabilityModel):
        try:
            plots_dir = Path(artifact_path).parent / "plots"
            shap_results = compute_shap_summary(
                model,
                training[FEATURE_COLUMNS],
                output_dir=plots_dir,
                max_samples=250,
            )
        except Exception as e:
            shap_results = {"error": str(e)}

    # Customer Segmentation via K-Means and RFM
    try:
        segmentation = perform_customer_segmentation(transactions, n_clusters=4)
    except Exception:
        segmentation = None

    # Market Basket Analysis (Association Rules)
    try:
        market_basket = mine_association_rules(transactions, min_support=0.005, min_lift=1.02)
    except Exception:
        market_basket = None

    # Customer Lifetime Value (CLV Regression)
    try:
        clv_reg_type = "xgboost" if model_name in ("xgboost", "lightgbm") else "ridge"
        clv_model, clv_summary = train_clv_model(training, model_name=clv_reg_type)
    except Exception:
        clv_model, clv_summary = None, None

    artifact = {
        "version": "2.0",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "horizon_days": int(horizon_days),
        "model_name": model_name,
        "model": model,
        "clv_model": clv_model,
        "clv_summary": clv_summary,
        "transactions": transactions,
        "quality": quality,
        "training_summary": summary,
        "shap": shap_results,
        "segmentation": segmentation,
        "market_basket": market_basket,
        "background_sample": training[FEATURE_COLUMNS].head(100) if not training.empty else None,
    }
    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_file)

    # MLflow experiment tracking
    try:
        run_id = log_training_run(artifact)
        artifact["mlflow_run_id"] = run_id
    except Exception:
        pass

    return artifact


def load_artifact(artifact_path: str | Path) -> dict[str, Any]:
    """Load a model artifact made by :func:`train_and_save`."""
    return joblib.load(artifact_path)


def customer_feature_row(transactions: pd.DataFrame, customer_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the current feature row for one customer's most recent transaction."""
    history = transactions.loc[transactions["Customer_Name"] == customer_name].sort_values("Timestamp").copy()
    if history.empty:
        raise KeyError(f"Customer '{customer_name}' was not found.")

    latest = history.iloc[-1]
    previous_timestamp = history["Timestamp"].iloc[-2] if len(history) > 1 else pd.NaT
    days_since_previous = (latest["Timestamp"] - previous_timestamp).total_seconds() / 86_400 if pd.notna(previous_timestamp) else np.nan
    purchased_products = {product for basket in history["Product_List"] for product in basket}

    values: dict[str, Any] = {column: latest[column] for column in CATEGORICAL_FEATURES}
    values.update(
        {
            "Total_Items": latest["Total_Items"],
            "Total_Cost": latest["Total_Cost"],
            "Discount_Flag": latest["Discount_Flag"],
            "Transactions_So_Far": len(history),
            "Average_Spend_So_Far": history["Total_Cost"].mean(),
            "Average_Items_So_Far": history["Total_Items"].mean(),
            "Discount_Rate_So_Far": history["Discount_Flag"].mean(),
            "Days_Since_Previous_Purchase": days_since_previous,
            "Unique_Products_So_Far": len(purchased_products),
        }
    )
    return pd.DataFrame([values])[FEATURE_COLUMNS], history


def predict_customer_purchase(artifact: dict[str, Any], customer_name: str) -> dict[str, Any]:
    """Return a human-readable purchase estimate for a selected customer."""
    feature_row, history = customer_feature_row(artifact["transactions"], customer_name)
    probability = float(_model_probability(artifact["model"], feature_row)[0])
    if probability >= BAND_HIGH_THRESHOLD:
        band = "High"
    elif probability >= BAND_MEDIUM_THRESHOLD:
        band = "Medium"
    else:
        band = "Low"

    local_factors = None
    if not isinstance(artifact["model"], ConstantProbabilityModel):
        try:
            background = artifact.get("background_sample")
            local_factors = explain_local_customer(
                artifact["model"],
                feature_row,
                background_df=background,
                top_n=5,
            )
        except Exception:
            local_factors = None

    # Predict forward Customer Lifetime Value ($) if CLV model exists
    predicted_clv = 0.0
    if artifact.get("clv_model") is not None:
        predicted_clv = predict_customer_clv(artifact["clv_model"], feature_row)

    return {
        "probability": probability,
        "band": band,
        "predicted_clv": round(float(predicted_clv), 2),
        "last_purchase": history["Timestamp"].max(),
        "transaction_count": int(len(history)),
        "total_spend": float(history["Total_Cost"].sum()),
        "local_factors": local_factors,
    }
