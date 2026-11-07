"""Tests for data loading, parsing, and quality validation."""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import (
    DataQualityError,
    load_transactions,
    parse_products,
)


def test_parse_products():
    """Verify parse_products handles various formats robustly."""
    # List literal string
    assert parse_products("['Apple', 'Banana']") == ["Apple", "Banana"]
    # Single product string with quotes
    assert parse_products("['Apple']") == ["Apple"]
    # Direct list
    assert parse_products(["Milk", "Bread"]) == ["Milk", "Bread"]
    # Empty string or NaN
    assert parse_products("") == []
    assert parse_products("   ") == []
    assert parse_products(float("nan")) == []
    # Plain string fallback
    assert parse_products("SingleItem") == ["SingleItem"]


def test_load_transactions_valid(sample_csv_text: str):
    """Verify valid CSV is correctly parsed and cleaned."""
    buffer = io.StringIO(sample_csv_text)
    df, quality = load_transactions(buffer, use_cache=False)

    assert len(df) == 8
    assert quality["valid_rows"] == 8
    assert quality["unique_customers"] == 4
    assert "Timestamp" in df.columns
    assert "Product_List" in df.columns
    assert "Discount_Flag" in df.columns
    assert "Primary_Product" in df.columns
    # Check types
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
    assert pd.api.types.is_numeric_dtype(df["Total_Cost"])


def test_load_transactions_missing_columns():
    """Verify missing required columns triggers DataQualityError."""
    bad_csv = "Transaction_ID,Date\nTX01,01-01-2023"
    buffer = io.StringIO(bad_csv)
    with pytest.raises(DataQualityError, match="missing required column"):
        load_transactions(buffer, use_cache=False)


def test_load_transactions_empty():
    """Verify empty or header-only CSV raises DataQualityError."""
    empty_csv = (
        "Transaction_ID,Date,Customer_Name,Product,Total_Items,Total_Cost,"
        "Payment_Method,City,Store_Type,Discount_Applied,Customer_Category,Season,Promotion\n"
    )
    buffer = io.StringIO(empty_csv)
    with pytest.raises(DataQualityError, match="No valid transaction rows"):
        load_transactions(buffer, use_cache=False)


def test_load_transactions_sampling(sample_csv_text: str):
    """Verify sampling fraction limits the returned rows."""
    buffer = io.StringIO(sample_csv_text)
    df_sample, _ = load_transactions(buffer, sample_frac=0.5, use_cache=False)
    assert len(df_sample) < 8
    assert len(df_sample) > 0


def test_load_transactions_caching(tmp_path: Path, sample_csv_text: str):
    """Verify Parquet caching creates a valid cached artifact and reloads from it."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(sample_csv_text, encoding="utf-8")

    # First load creates cache
    df1, q1 = load_transactions(csv_file, use_cache=True)
    assert len(df1) == 8

    # Second load loads from cache
    df2, q2 = load_transactions(csv_file, use_cache=True)
    assert len(df2) == 8
    assert df1["Customer_Name"].tolist() == df2["Customer_Name"].tolist()


def test_load_transactions_uci_format():
    """Verify arbitrary real-world dataset formats (like UCI Online Retail) are auto-mapped."""
    uci_csv = (
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,01/12/2020 08:26,2.55,17850,United Kingdom\n"
        "536365,71053,WHITE METAL LANTERN,6,01/12/2020 08:26,3.39,17850,United Kingdom\n"
        "536366,22633,HAND WARMER UNION JACK,6,01/12/2020 08:28,1.85,17850,United Kingdom\n"
        "536367,84879B,ASSORTED COLOUR BIRD ORNAMENT,32,01/12/2020 08:34,1.69,13047,France\n"
    )
    buffer = io.StringIO(uci_csv)
    df, quality = load_transactions(buffer, use_cache=False)

    # 4 line-items aggregated into 3 distinct invoices
    assert len(df) == 3
    assert quality["valid_rows"] == 3
    assert quality["unique_customers"] == 2
    assert "Customer_Name" in df.columns
    assert "Total_Cost" in df.columns
    assert "Product_List" in df.columns
    # Check invoice 536365 aggregated spend: 6 * 2.55 + 6 * 3.39 = 35.64
    inv_536365 = df[df["Transaction_ID"] == "536365"].iloc[0]
    assert pytest.approx(inv_536365["Total_Cost"], 0.01) == 35.64
    assert len(inv_536365["Product_List"]) == 2


def test_load_transactions_minimal_format():
    """Verify minimal 3-column dataset (customer_id, order_date, spend) works seamlessly."""
    min_csv = (
        "customer_id,order_date,spend\n"
        "Cust_A,2023-01-15,45.50\n"
        "Cust_A,2023-03-20,30.00\n"
        "Cust_B,2023-02-10,15.20\n"
    )
    buffer = io.StringIO(min_csv)
    df, quality = load_transactions(buffer, use_cache=False)

    assert len(df) == 3
    assert quality["unique_customers"] == 2
    assert "Transaction_ID" in df.columns
    assert "Customer_Name" in df.columns
    assert "Timestamp" in df.columns
    assert "Season" in df.columns
    assert df["Total_Cost"].tolist() == [45.5, 30.0, 15.2]

