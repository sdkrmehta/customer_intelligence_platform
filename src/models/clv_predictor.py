"""Customer Lifetime Value (CLV) regression model.

Predicts the expected monetary spend ($) a customer will generate within
the future prediction horizon, enabling high-value customer identification
and ROI-based marketing budget allocation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def _build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing pipeline for regression."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _build_regressor(model_name: str) -> Any:
    """Construct a regression model instance."""
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    # Linear baseline
    return Ridge(alpha=1.0)


def build_clv_pipeline(model_name: str = "xgboost") -> Pipeline:
    """Create an end-to-end CLV regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("regressor", _build_regressor(model_name)),
        ]
    )


def train_clv_model(
    training_df: pd.DataFrame,
    model_name: str = "xgboost",
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a CLV regression model to predict forward horizon spend.

    Parameters
    ----------
    training_df : pd.DataFrame
        Labelled DataFrame from :func:`build_training_frame` containing
        features and 'Future_Spend_In_Horizon'.
    model_name : str
        Regressor algorithm key ('xgboost', 'lightgbm', 'random_forest', 'ridge').

    Returns
    -------
    (fitted_pipeline, metrics_summary)
    """
    X = training_df[FEATURE_COLUMNS]
    y = training_df["Future_Spend_In_Horizon"]

    pipeline = build_clv_pipeline(model_name)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_pred_clipped = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    mse = float(mean_squared_error(y_test, y_pred_clipped))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_clipped))

    summary = {
        "model_type": f"CLV Regressor ({model_name.replace('_', ' ').title()})",
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "mean_actual_spend": round(float(y_test.mean()), 2),
        "mean_predicted_spend": round(float(y_pred_clipped.mean()), 2),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }

    return pipeline, summary


def predict_customer_clv(clv_pipeline: Pipeline, feature_row: pd.DataFrame) -> float:
    """Predict expected future monetary spend ($) for a specific customer."""
    try:
        raw_pred = float(clv_pipeline.predict(feature_row)[0])
        return float(np.clip(raw_pred, 0.0, None))
    except Exception:
        return 0.0
