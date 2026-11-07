"""Centralized configuration and constants for the retail prediction project.

Loads tunable settings from ``configs/model_config.yaml`` when available,
falling back to sensible defaults so the project works without the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Locate and load the YAML config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Read the YAML config file, returning an empty dict if missing."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


_CFG = _load_yaml_config()

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------

_data_cfg = _CFG.get("data", {})
RAW_DATA_PATH: str = _data_cfg.get("raw_path", "data/raw/Retail_Transactions_Dataset.csv")
PROCESSED_DIR: str = _data_cfg.get("processed_dir", "data/processed")
DATE_FORMAT_DAYFIRST: bool = _data_cfg.get("date_format_dayfirst", True)

# ---------------------------------------------------------------------------
# Column / feature constants (not in YAML — schema is fixed)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "Transaction_ID",
    "Date",
    "Customer_Name",
    "Product",
    "Total_Items",
    "Total_Cost",
    "Payment_Method",
    "City",
    "Store_Type",
    "Discount_Applied",
    "Customer_Category",
    "Season",
    "Promotion",
}

CATEGORICAL_FEATURES = [
    "City",
    "Store_Type",
    "Payment_Method",
    "Customer_Category",
    "Season",
    "Promotion",
    "Primary_Product",
]

NUMERIC_FEATURES = [
    "Total_Items",
    "Total_Cost",
    "Discount_Flag",
    "Transactions_So_Far",
    "Average_Spend_So_Far",
    "Average_Items_So_Far",
    "Discount_Rate_So_Far",
    "Days_Since_Previous_Purchase",
    "Unique_Products_So_Far",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

_training_cfg = _CFG.get("training", {})
DEFAULT_HORIZON_DAYS: int = _training_cfg.get("horizon_days", 365)
TEST_SIZE: float = _training_cfg.get("test_size", 0.25)
RANDOM_STATE: int = _training_cfg.get("random_state", 42)
DEFAULT_MODEL: str = _training_cfg.get("default_model", "logistic_regression")

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

_models_cfg = _CFG.get("models", {})
LOGISTIC_REGRESSION_PARAMS: dict[str, Any] = _models_cfg.get(
    "logistic_regression", {"max_iter": 2000, "class_weight": "balanced"}
)
RANDOM_FOREST_PARAMS: dict[str, Any] = _models_cfg.get(
    "random_forest", {
        "n_estimators": 200, "max_depth": 12, "min_samples_split": 10,
        "min_samples_leaf": 5, "class_weight": "balanced",
    }
)
XGBOOST_PARAMS: dict[str, Any] = _models_cfg.get(
    "xgboost", {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss",
        "verbosity": 0,
    }
)
LIGHTGBM_PARAMS: dict[str, Any] = _models_cfg.get(
    "lightgbm", {
        "n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
        "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
        "is_unbalance": True, "verbose": -1,
    }
)

AVAILABLE_MODELS: list[str] = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

_eval_cfg = _CFG.get("evaluation", {})
PROBABILITY_THRESHOLD: float = _eval_cfg.get("probability_threshold", 0.5)
LOW_CONFIDENCE_MIN_POSITIVES: int = _eval_cfg.get("low_confidence_min_positives", 20)

# ---------------------------------------------------------------------------
# Recommendation settings
# ---------------------------------------------------------------------------

_rec_cfg = _CFG.get("recommendation", {})
RECOMMENDATION_TOP_N: int = _rec_cfg.get("top_n", 8)
_weights = _rec_cfg.get("weights", {})
REC_WEIGHT_POPULARITY: float = _weights.get("popularity", 0.6)
REC_WEIGHT_CONTEXT: float = _weights.get("context", 0.8)
REC_WEIGHT_PERSONAL: float = _weights.get("personal", 2.5)
REC_WEIGHT_ASSOCIATION: float = _weights.get("association", 1.8)

# ---------------------------------------------------------------------------
# Prediction band thresholds
# ---------------------------------------------------------------------------

_bands_cfg = _CFG.get("bands", {})
BAND_HIGH_THRESHOLD: float = _bands_cfg.get("high_threshold", 0.60)
BAND_MEDIUM_THRESHOLD: float = _bands_cfg.get("medium_threshold", 0.30)
