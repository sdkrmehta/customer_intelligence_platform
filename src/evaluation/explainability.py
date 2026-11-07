"""Model interpretability and explainability using SHAP.

Computes global feature importances (beeswarm summary, bar plot) and local
prediction explanations (per-customer feature contributions) for tree-based
and linear classifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _get_feature_names_from_preprocessor(preprocessor, original_feature_names: list[str]) -> list[str]:
    """Retrieve output feature names from an sklearn ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        return [f"feature_{i}" for i in range(len(original_feature_names))]


def compute_shap_summary(
    pipeline,
    features: pd.DataFrame,
    output_dir: str | Path = "models",
    max_samples: int = 300,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute SHAP values and generate summary plots for a fitted pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing a 'preprocessor' step and a 'classifier' step.
    features : pd.DataFrame
        Input features (untransformed).
    output_dir : str or Path
        Directory where generated plot PNGs will be saved.
    max_samples : int
        Number of background samples to evaluate to keep computation fast.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    dict
        Dictionary containing:
        - 'importance_df': pd.DataFrame with features ranked by mean |SHAP|
        - 'summary_plot_path': str path to saved beeswarm plot
        - 'bar_plot_path': str path to saved bar plot
        - 'expected_value': baseline expected value
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Subsample if dataset is large to ensure fast SHAP computation
    if len(features) > max_samples:
        sample_df = features.sample(n=max_samples, random_state=random_state)
    else:
        sample_df = features.copy()

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform features
    transformed_matrix = preprocessor.transform(sample_df)
    if hasattr(transformed_matrix, "toarray"):
        transformed_matrix = transformed_matrix.toarray()

    feature_names = _get_feature_names_from_preprocessor(preprocessor, list(features.columns))

    # Clean up feature names for cleaner plots (e.g. categorical__City_Los Angeles -> City: Los Angeles)
    clean_names = [
        name.replace("categorical__", "").replace("numeric__", "").replace("_", " ")
        for name in feature_names
    ]

    classifier_type = type(classifier).__name__

    try:
        if classifier_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_matrix)
            # Handle binary classification multi-output shapes
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Class 1 (positive class) SHAP values
                shap_matrix = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_matrix = shap_values[:, :, 1]
            else:
                shap_matrix = shap_values
            expected_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1
                else explainer.expected_value
            )
        elif classifier_type == "LogisticRegression":
            explainer = shap.LinearExplainer(classifier, transformed_matrix)
            shap_values = explainer.shap_values(transformed_matrix)
            shap_matrix = shap_values
            expected_val = explainer.expected_value
        else:
            # Universal fallback using KernelExplainer on a small sample
            background = transformed_matrix[: min(50, len(transformed_matrix))]
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_vals = explainer.shap_values(transformed_matrix[: min(50, len(transformed_matrix))])
            shap_matrix = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            expected_val = 0.5

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        importance_df = pd.DataFrame(
            {"Feature": clean_names, "Mean_Abs_SHAP": mean_abs_shap}
        ).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        # 1. Bar Plot of Top 15 Features
        top_15 = importance_df.head(15).sort_values("Mean_Abs_SHAP", ascending=True)
        fig_bar, ax_bar = plt.subplots(figsize=(8, 6), tight_layout=True)
        ax_bar.barh(top_15["Feature"], top_15["Mean_Abs_SHAP"], color="#1f77b4", edgecolor="none")
        ax_bar.set_xlabel("Mean |SHAP Value| (Impact on Purchase Likelihood)")
        ax_bar.set_title(f"Top Features Driving Purchase Likelihood ({classifier_type})")
        bar_plot_path = output_path / "shap_bar_plot.png"
        fig_bar.savefig(bar_plot_path, dpi=150)
        plt.close(fig_bar)

        # 2. Beeswarm Summary Plot
        fig_summary, ax_summary = plt.subplots(figsize=(9, 6), tight_layout=True)
        shap.summary_plot(
            shap_matrix,
            features=transformed_matrix,
            feature_names=clean_names,
            max_display=15,
            show=False,
        )
        summary_plot_path = output_path / "shap_summary_plot.png"
        plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        return {
            "importance_df": importance_df.head(20),
            "summary_plot_path": str(summary_plot_path),
            "bar_plot_path": str(bar_plot_path),
            "expected_value": float(np.ravel(expected_val)[0]) if expected_val is not None else 0.5,
        }

    except Exception as err:
        # Fallback if SHAP computation encounters any format mismatch
        plt.close("all")
        return {
            "error": str(err),
            "importance_df": pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP"]),
            "summary_plot_path": None,
            "bar_plot_path": None,
            "expected_value": None,
        }


def explain_local_customer(
    pipeline,
    customer_feature_df: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
    top_n: int = 6,
) -> pd.DataFrame:
    """Explain why a specific customer received their purchase likelihood score.

    Returns the top positive and negative contributing factors for that individual.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform
    sample_df = background_df.head(100) if background_df is not None and not background_df.empty else customer_feature_df
    bg_transformed = preprocessor.transform(sample_df)
    if hasattr(bg_transformed, "toarray"):
        bg_transformed = bg_transformed.toarray()

    cust_transformed = preprocessor.transform(customer_feature_df)
    if hasattr(cust_transformed, "toarray"):
        cust_transformed = cust_transformed.toarray()

    feature_names = _get_feature_names_from_preprocessor(preprocessor, list(customer_feature_df.columns))
    clean_names = [
        name.replace("categorical__", "").replace("numeric__", "").replace("_", " ")
        for name in feature_names
    ]

    classifier_type = type(classifier).__name__

    try:
        if classifier_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
            explainer = shap.TreeExplainer(classifier)
            shap_val = explainer.shap_values(cust_transformed)
            if isinstance(shap_val, list) and len(shap_val) == 2:
                row_shap = shap_val[1][0]
            elif isinstance(shap_val, np.ndarray) and shap_val.ndim == 3:
                row_shap = shap_val[0, :, 1]
            else:
                row_shap = shap_val[0]
        else:
            explainer = shap.LinearExplainer(classifier, bg_transformed)
            shap_val = explainer.shap_values(cust_transformed)
            row_shap = shap_val[0] if shap_val.ndim == 2 else shap_val

        local_df = pd.DataFrame({
            "Factor": clean_names,
            "Contribution": row_shap,
        })
        # Rank by magnitude of contribution
        local_df["Abs_Contribution"] = local_df["Contribution"].abs()
        top_factors = local_df.sort_values("Abs_Contribution", ascending=False).head(top_n).copy()

        top_factors["Impact"] = top_factors["Contribution"].apply(
            lambda v: "Pushes Likelihood Higher" if v > 0 else "Pushes Likelihood Lower"
        )
        top_factors["Magnitude"] = top_factors["Contribution"].apply(lambda v: f"{v:+.3f}")
        return top_factors[["Factor", "Impact", "Magnitude"]].reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=["Factor", "Impact", "Magnitude"])
