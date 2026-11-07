"""Model factory: build any supported classifier from a name and config.

Centralizes model construction so that the training pipeline, Streamlit app,
and CLI can select a model by name without knowing implementation details.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import (
    AVAILABLE_MODELS,
    CATEGORICAL_FEATURES,
    LIGHTGBM_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_DISPLAY_NAMES,
    NUMERIC_FEATURES,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    XGBOOST_PARAMS,
)


def _build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing pipeline for all models."""
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


def _build_classifier(model_name: str) -> Any:
    """Instantiate a classifier by name using config-driven hyperparameters."""
    if model_name == "logistic_regression":
        params = dict(LOGISTIC_REGRESSION_PARAMS)
        return LogisticRegression(
            max_iter=params.get("max_iter", 2_000),
            class_weight=params.get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":
        params = dict(RANDOM_FOREST_PARAMS)
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 12),
            min_samples_split=params.get("min_samples_split", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            class_weight=params.get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(XGBOOST_PARAMS)
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            eval_metric=params.get("eval_metric", "logloss"),
            verbosity=params.get("verbosity", 0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(LIGHTGBM_PARAMS)
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 8),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=params.get("num_leaves", 31),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            is_unbalance=params.get("is_unbalance", True),
            verbose=params.get("verbose", -1),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {', '.join(AVAILABLE_MODELS)}"
    )


def build_model_pipeline(model_name: str) -> Pipeline:
    """Build a full sklearn Pipeline (preprocessor + classifier) by model name.

    Parameters
    ----------
    model_name : str
        One of the keys in :data:`AVAILABLE_MODELS`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit pipeline with preprocessing and the chosen classifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("classifier", _build_classifier(model_name)),
        ]
    )


def get_display_name(model_name: str) -> str:
    """Return the human-readable name for a model key."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)
