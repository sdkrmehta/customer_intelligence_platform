"""Data preparation, prediction, and recommendation functions for the project.

The functions in this module are kept separate from the web interface so that the
same model can be trained from the command line or from the Streamlit app.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


class DataQualityError(ValueError):
    """Raised when an uploaded file cannot be used by this project."""


class ConstantProbabilityModel:
    """Safe fallback used only when a dataset contains one target class."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.positive_probability)
        return np.column_stack([1 - positive, positive])


def parse_products(value: Any) -> list[str]:
    """Convert the text representation of a product basket into product names."""
    if isinstance(value, list):
        products = value
    elif pd.isna(value) or not str(value).strip():
        products = []
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            products = parsed if isinstance(parsed, list) else [parsed]
        except (SyntaxError, ValueError):
            # This fallback keeps the app usable if a future CSV stores one product
            # name without the square-bracket list notation.
            products = [text.strip("[]'\" ")]

    return [str(product).strip() for product in products if str(product).strip()]


def load_transactions(source: str | Path | BinaryIO | TextIO) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate, and clean the source transaction CSV.

    The supplied dataset contains many blank rows after the valid records. Pandas
    ignores those rows, and the explicit checks below also protect uploaded files.
    """
    try:
        raw = pd.read_csv(source, skip_blank_lines=True, low_memory=False)
    except Exception as error:  # pragma: no cover - message is for app users
        raise DataQualityError(f"The file could not be read as a CSV: {error}") from error

    raw.columns = raw.columns.astype(str).str.strip()
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataQualityError(f"The CSV is missing required column(s): {missing}.")

    input_rows = len(raw)
    data = raw.copy()
    data["Transaction_ID"] = data["Transaction_ID"].astype("string").str.strip()
    data["Customer_Name"] = data["Customer_Name"].astype("string").str.strip()
    data["Timestamp"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")

    valid_rows = (
        data["Transaction_ID"].notna()
        & data["Transaction_ID"].ne("")
        & data["Customer_Name"].notna()
        & data["Customer_Name"].ne("")
        & data["Timestamp"].notna()
    )
    data = data.loc[valid_rows].copy()
    if data.empty:
        raise DataQualityError("No valid transaction rows were found in the CSV.")

    for column in ["Total_Items", "Total_Cost"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        median = data[column].median()
        data[column] = data[column].fillna(0 if pd.isna(median) else median).clip(lower=0)

    text_columns = [
        "Payment_Method",
        "City",
        "Store_Type",
        "Customer_Category",
        "Season",
        "Promotion",
    ]
    for column in text_columns:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    data["Discount_Flag"] = (
        data["Discount_Applied"].fillna(False).astype(str).str.upper().isin({"TRUE", "1", "YES", "Y"}).astype(int)
    )
    data["Product_List"] = data["Product"].apply(parse_products)
    data["Primary_Product"] = data["Product_List"].apply(lambda products: products[0] if products else "Unknown")
    data["Transaction_Month"] = data["Timestamp"].dt.to_period("M").astype(str)

    data = data.sort_values(["Customer_Name", "Timestamp", "Transaction_ID"], kind="stable").reset_index(drop=True)
    quality = {
        "input_rows": int(input_rows),
        "valid_rows": int(len(data)),
        "dropped_rows": int(input_rows - len(data)),
        "unique_customers": int(data["Customer_Name"].nunique()),
        "date_start": data["Timestamp"].min().strftime("%d %b %Y"),
        "date_end": data["Timestamp"].max().strftime("%d %b %Y"),
        "customers_with_repeat_purchase": int((data.groupby("Customer_Name").size() > 1).sum()),
    }
    return data, quality


def _add_history_features(transactions: pd.DataFrame) -> pd.DataFrame:
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


def build_training_frame(transactions: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Label a row 1 if that customer buys again within the chosen horizon.

    To avoid claiming that an unobserved future is a negative, records within the
    final *horizon_days* of the dataset are excluded from supervised training.
    """
    if horizon_days < 30:
        raise ValueError("The prediction horizon must be at least 30 days.")

    data = _add_history_features(transactions)
    next_purchase = data.groupby("Customer_Name", sort=False)["Timestamp"].shift(-1)
    data["Next_Purchase_Date"] = next_purchase
    data["Will_Purchase_In_Horizon"] = (
        ((next_purchase - data["Timestamp"]).dt.total_seconds() <= horizon_days * 86_400)
        & next_purchase.notna()
    ).astype(int)

    observation_end = data["Timestamp"].max()
    label_cutoff = observation_end - pd.Timedelta(days=horizon_days)
    labelled = data.loc[data["Timestamp"] <= label_cutoff].copy()
    if labelled.empty:
        raise DataQualityError(
            "The selected prediction horizon leaves no labelled rows. Choose a shorter horizon or provide more history."
        )
    return labelled


def _model_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return the probability of target class 1 for sklearn and fallback models."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    positive_index = classes.index(1) if 1 in classes else -1
    if positive_index == -1:
        return np.zeros(len(features))
    return probabilities[:, positive_index]


def train_purchase_model(transactions: pd.DataFrame, horizon_days: int = 365) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Train a logistic-regression purchase-within-horizon classifier."""
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
        # A normal classifier cannot learn from one class or one isolated positive
        # example. The app remains usable but labels the outcome as low confidence.
        model: Any = ConstantProbabilityModel(float(target.mean()))
        summary.update(
            {
                "model_type": "Dataset-rate fallback",
                "evaluation_note": "Too few repeat-purchase examples for a reliable supervised classifier.",
                "is_low_confidence": True,
            }
        )
        return model, summary, training

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    model.fit(x_train, y_train)
    probabilities = _model_probability(model, x_test)
    predictions = (probabilities >= 0.5).astype(int)

    summary.update(
        {
            "model_type": "Class-balanced logistic regression",
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
            "is_low_confidence": bool(class_counts.get(1, 0) < 20),
            "evaluation_note": "Holdout metrics are indicative only; sparse repeat-customer history limits reliability."
            if class_counts.get(1, 0) < 20
            else "Metrics calculated on a stratified 25% holdout set.",
        }
    )
    return model, summary, training


def train_and_save(
    source: str | Path | BinaryIO | TextIO,
    artifact_path: str | Path,
    horizon_days: int = 365,
) -> dict[str, Any]:
    """Train the model and write one reusable artifact for the web application."""
    transactions, quality = load_transactions(source)
    model, summary, _ = train_purchase_model(transactions, horizon_days=horizon_days)
    artifact = {
        "version": "1.0",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "horizon_days": int(horizon_days),
        "model": model,
        "transactions": transactions,
        "quality": quality,
        "training_summary": summary,
    }
    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_file)
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
    if probability >= 0.60:
        band = "High"
    elif probability >= 0.30:
        band = "Medium"
    else:
        band = "Low"
    return {
        "probability": probability,
        "band": band,
        "last_purchase": history["Timestamp"].max(),
        "transaction_count": int(len(history)),
        "total_spend": float(history["Total_Cost"].sum()),
    }


def _normalise(counter: Counter[str]) -> dict[str, float]:
    largest = max(counter.values(), default=0)
    return {key: value / largest for key, value in counter.items()} if largest else {}


def recommend_products(transactions: pd.DataFrame, customer_name: str, top_n: int = 8) -> pd.DataFrame:
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
            0.6 * popularity_score.get(product, 0)
            + 0.8 * context_score.get(product, 0)
            + 2.5 * personal_score.get(product, 0)
            + 1.8 * association_score.get(product, 0)
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
