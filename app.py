"""Streamlit interface for the Retail Customer Purchase Prediction project."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.retail_predictor import (
    DataQualityError,
    dashboard_data,
    load_artifact,
    predict_customer_purchase,
    recommend_products,
    train_and_save,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "Retail_Transactions_Dataset.csv"
MODEL_FILE = PROJECT_DIR / "models" / "purchase_predictor.joblib"

st.set_page_config(page_title="Retail Purchase Prediction", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner=False)
def get_artifact(model_file: str, modified_time: float) -> dict:
    """Cache the saved model but automatically refresh it after retraining."""
    return load_artifact(model_file)


def format_percent(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.1%}"


def main() -> None:
    st.title("🛒 Retail Customer Purchase Prediction")
    st.caption("Predict a customer's likelihood of returning and recommend products from transaction history.")

    with st.sidebar:
        st.header("Model setup")
        horizon = st.number_input(
            "Prediction horizon (days)", min_value=30, max_value=1_460, value=365, step=30,
            help="A positive label means the same customer made another recorded purchase within this number of days.",
        )
        upload = st.file_uploader("Optional: upload a transaction CSV", type=["csv"])
        st.caption("If no file is uploaded, the supplied project dataset is used.")
        train_clicked = st.button("Train / update model", type="primary", width="stretch")

    if train_clicked:
        source = BytesIO(upload.getvalue()) if upload is not None else DEFAULT_DATASET
        try:
            with st.spinner("Cleaning data and training the model..."):
                artifact = train_and_save(source, MODEL_FILE, int(horizon))
            get_artifact.clear()
            st.success(
                f"Model trained with {artifact['quality']['valid_rows']:,} valid transactions and saved successfully."
            )
        except (DataQualityError, ValueError, OSError) as error:
            st.error(f"Training could not be completed: {error}")

    if not MODEL_FILE.exists():
        st.info("Start by clicking **Train / update model** in the sidebar. It uses the supplied CSV by default.")
        st.stop()

    try:
        artifact = get_artifact(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
    except Exception as error:  # pragma: no cover - defensive UI handling
        st.error(f"The saved model could not be opened. Train it again. Details: {error}")
        st.stop()

    transactions = artifact["transactions"]
    quality = artifact["quality"]
    summary = artifact["training_summary"]
    dashboard = dashboard_data(transactions)
    dashboard_tab, customer_tab, model_tab, data_tab = st.tabs(
        ["Dashboard", "Customer prediction", "Model details", "Data quality"]
    )

    with dashboard_tab:
        total_revenue = transactions["Total_Cost"].sum()
        average_order = transactions["Total_Cost"].mean()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valid transactions", f"{len(transactions):,}")
        col2.metric("Customers", f"{transactions['Customer_Name'].nunique():,}")
        col3.metric("Recorded revenue", f"${total_revenue:,.2f}")
        col4.metric("Average order value", f"${average_order:,.2f}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.line(dashboard["monthly"], x="Transaction_Month", y="Revenue", markers=True, title="Revenue by month"),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.bar(dashboard["city"], x="City", y="Revenue", title="Revenue by city", color="Revenue", color_continuous_scale="Blues"),
                width="stretch",
            )
        st.plotly_chart(
            px.bar(dashboard["products"], x="Product", y="Times purchased", title="Most frequently purchased products"),
            width="stretch",
        )

    with customer_tab:
        st.subheader("Customer-level prediction and recommendations")
        customers = sorted(transactions["Customer_Name"].unique())
        selected_customer = st.selectbox("Choose a customer", customers)
        result = predict_customer_purchase(artifact, selected_customer)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Purchase likelihood within {artifact['horizon_days']} days", format_percent(result["probability"]))
        col2.metric("Risk / opportunity band", result["band"])
        col3.metric("Recorded purchases", result["transaction_count"])
        col4.metric("Recorded spend", f"${result['total_spend']:,.2f}")
        st.caption(
            f"Most recent recorded transaction: {result['last_purchase'].strftime('%d %b %Y')}. "
            "This is a decision-support estimate, not a guarantee of customer behavior."
        )
        st.subheader("Personalized product recommendations")
        recommendations = recommend_products(transactions, selected_customer)
        st.dataframe(recommendations, width="stretch", hide_index=True)
        st.download_button(
            "Download these recommendations as CSV",
            data=recommendations.to_csv(index=False).encode("utf-8"),
            file_name=f"recommendations_{selected_customer.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    with model_tab:
        st.subheader("How the prediction is made")
        st.write(
            f"The target is whether a customer makes another recorded purchase within **{artifact['horizon_days']} days**. "
            "The classifier uses transaction context, spend, discount use, purchase history, and product information."
        )
        metric_rows = [
            ("Model", summary["model_type"]),
            ("Labelled training transactions", f"{summary['labelled_transactions']:,}"),
            ("Positive examples", f"{summary['positive_examples']:,}"),
            ("Negative examples", f"{summary['negative_examples']:,}"),
            ("Positive rate", format_percent(summary["positive_rate"])),
        ]
        for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1_score"), ("Average precision", "average_precision"), ("ROC-AUC", "roc_auc")]:
            if key in summary:
                metric_rows.append((label, format_percent(summary[key])))
        st.dataframe(pd.DataFrame(metric_rows, columns=["Measure", "Value"]), hide_index=True, width="stretch")
        if summary.get("is_low_confidence"):
            st.warning(
                "Low-confidence notice: this dataset has very few customers with repeat purchases. "
                "Use the predictions as a demonstration and collect more repeat-customer history before using it for real decisions."
            )
        st.info(summary["evaluation_note"])

    with data_tab:
        st.subheader("Data validation summary")
        st.write(f"Model trained: {artifact['trained_at']}")
        details = pd.DataFrame(
            [
                ("Rows read", quality["input_rows"]),
                ("Valid transaction rows", quality["valid_rows"]),
                ("Rows removed", quality["dropped_rows"]),
                ("Unique customers", quality["unique_customers"]),
                ("Customers with more than one recorded purchase", quality["customers_with_repeat_purchase"]),
                ("Data period", f"{quality['date_start']} to {quality['date_end']}"),
            ],
            columns=["Check", "Result"],
        )
        # Streamlit/Arrow needs a uniform column type; the report deliberately
        # contains counts and a readable date range in the Result column.
        details["Result"] = details["Result"].astype(str)
        st.dataframe(details, hide_index=True, width="stretch")
        st.caption(
            "Blank rows are ignored. Numeric values are checked, dates are parsed in day-month-year format, and incomplete key rows are excluded."
        )


if __name__ == "__main__":
    main()
