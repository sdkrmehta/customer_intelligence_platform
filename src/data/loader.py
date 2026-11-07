"""Data loading, validation, and cleaning for the retail transaction CSV.

The functions in this module read raw CSVs, validate required columns,
handle blank rows, and produce a clean DataFrame ready for feature engineering.
Supports Parquet caching for fast reloads and optional row sampling.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any, BinaryIO, TextIO

import pandas as pd

from src.utils.config import PROCESSED_DIR, REQUIRED_COLUMNS


class DataQualityError(ValueError):
    """Raised when an uploaded file cannot be used by this project."""


COLUMN_SYNONYMS: dict[str, list[str]] = {
    "Transaction_ID": [
        "transaction_id", "transactionid", "invoice_no", "invoiceno", "invoice", "order_id", "orderid",
        "id", "trans_id", "transaction_number", "transactionno", "order_number", "orderno"
    ],
    "Date": [
        "date", "invoicedate", "invoice_date", "order_date", "orderdate", "timestamp", "datetime",
        "created_at", "trans_date", "transaction_date", "purchase_date", "purchasedate"
    ],
    "Customer_Name": [
        "customer_name", "customername", "customer_id", "customerid", "client_id", "clientid",
        "user_id", "userid", "customer", "user", "buyer_id", "shopper_id", "account_id"
    ],
    "Product": [
        "product", "products", "description", "item", "items", "stockcode", "product_name",
        "product_title", "title", "sku", "item_name", "item_description"
    ],
    "Total_Cost": [
        "total_cost", "totalcost", "total_amount", "totalamount", "amount", "price", "sales",
        "revenue", "spend", "order_total", "total", "net_amount", "grand_total"
    ],
    "Total_Items": [
        "total_items", "totalitems", "quantity", "qty", "items_count", "item_count", "units",
        "item_qty", "order_quantity"
    ],
    "City": [
        "city", "country", "location", "region", "state", "store_location", "metro", "area"
    ],
    "Payment_Method": [
        "payment_method", "paymentmethod", "payment_type", "payment", "pay_mode", "tender", "payment_mode"
    ],
    "Store_Type": [
        "store_type", "storetype", "channel", "platform", "store", "store_name", "outlet", "retail_type"
    ],
    "Discount_Applied": [
        "discount_applied", "discountapplied", "discount", "is_discount", "has_discount", "discount_flag"
    ],
    "Customer_Category": [
        "customer_category", "customercategory", "segment", "customer_type", "member_type", "tier", "cohort"
    ],
    "Season": [
        "season"
    ],
    "Promotion": [
        "promotion", "promo", "campaign", "discount_code", "voucher", "offer"
    ],
}


def _normalize_name(name: str) -> str:
    """Normalize a column name by removing non-alphanumeric chars and lowercasing."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _map_columns_to_canonical(raw_columns: list[str]) -> dict[str, str]:
    """Map raw dataframe column names to canonical schema column names using synonym lists."""
    mapping = {}
    norm_map = {_normalize_name(c): c for c in raw_columns}
    for canon, syns in COLUMN_SYNONYMS.items():
        if canon in raw_columns:
            mapping[canon] = canon
            continue
        for s in syns:
            ns = _normalize_name(s)
            if ns in norm_map and norm_map[ns] not in mapping:
                mapping[norm_map[ns]] = canon
                break
    return mapping


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


def _compute_quality(data: pd.DataFrame, input_rows: int) -> dict[str, Any]:
    """Build the data quality summary dictionary."""
    return {
        "input_rows": int(input_rows),
        "valid_rows": int(len(data)),
        "dropped_rows": int(input_rows - len(data)),
        "unique_customers": int(data["Customer_Name"].nunique()),
        "date_start": data["Timestamp"].min().strftime("%d %b %Y"),
        "date_end": data["Timestamp"].max().strftime("%d %b %Y"),
        "customers_with_repeat_purchase": int((data.groupby("Customer_Name").size() > 1).sum()),
    }


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply universal schema adaptation, validation, and cleaning to a raw DataFrame."""
    raw.columns = raw.columns.astype(str).str.strip()

    # 1. Map columns to canonical names
    col_map = _map_columns_to_canonical(list(raw.columns))
    rename_dict = {raw_col: canon_col for raw_col, canon_col in col_map.items() if raw_col != canon_col}
    data = raw.rename(columns=rename_dict).copy()

    # 2. Check Core Requirements: Customer, Date, and Transaction_ID
    missing_core = []
    if "Customer_Name" not in data.columns:
        missing_core.append("Customer_Name")
    if "Date" not in data.columns:
        missing_core.append("Date")
    if "Transaction_ID" not in data.columns:
        if "Customer_Name" in data.columns and "Date" in data.columns and not data.empty:
            data["Transaction_ID"] = "TX_" + data.index.astype(str)
        else:
            missing_core.append("Transaction_ID")

    if missing_core:
        missing_str = ", ".join(sorted(missing_core))
        raise DataQualityError(f"The CSV is missing required column(s): {missing_str}.")

    data["Transaction_ID"] = data["Transaction_ID"].astype("string").str.strip()
    data["Customer_Name"] = data["Customer_Name"].astype("string").str.strip()
    data["Timestamp"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")

    # Filter out rows with invalid critical identifiers
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

    # 3. Handle Spend / Cost calculation
    if "Total_Cost" not in data.columns:
        # Check if UnitPrice or Price exists
        price_col = None
        for p_cand in ["UnitPrice", "unit_price", "price", "Price", "item_price", "cost"]:
            if p_cand in data.columns:
                price_col = p_cand
                break
        qty_col = "Total_Items" if "Total_Items" in data.columns else None
        if price_col and qty_col:
            data["Total_Cost"] = (
                pd.to_numeric(data[qty_col], errors="coerce").fillna(1)
                * pd.to_numeric(data[price_col], errors="coerce").fillna(0)
            )
        elif price_col:
            data["Total_Cost"] = pd.to_numeric(data[price_col], errors="coerce").fillna(0)
        else:
            data["Total_Cost"] = 10.0

    for column in ["Total_Items", "Total_Cost"]:
        if column not in data.columns:
            data[column] = 1.0 if column == "Total_Items" else 10.0
        data[column] = pd.to_numeric(data[column], errors="coerce")
        median = data[column].median()
        data[column] = data[column].fillna(0 if pd.isna(median) else median).clip(lower=0)

    # 4. Product parsing
    if "Product" not in data.columns:
        data["Product"] = "Item"
    data["Product_List"] = data["Product"].apply(parse_products)

    # 5. Multi-row line item aggregation (if dataset has multiple lines per transaction)
    if data["Transaction_ID"].duplicated().any() and data.groupby(["Transaction_ID", "Customer_Name"]).size().max() > 1:
        agg_spec = {
            "Timestamp": "first",
            "Date": "first",
            "Total_Cost": "sum",
            "Total_Items": "sum",
            "Product_List": lambda p_lists: [item for sub in p_lists for item in (sub if isinstance(sub, list) else [sub])],
        }
        for opt_c in ["City", "Store_Type", "Payment_Method", "Customer_Category", "Season", "Promotion", "Discount_Applied"]:
            if opt_c in data.columns:
                agg_spec[opt_c] = "first"
        data = data.groupby(["Transaction_ID", "Customer_Name"], as_index=False).agg(agg_spec).reset_index(drop=True)

    # 6. Synthesize optional context features if missing
    if "City" not in data.columns:
        data["City"] = "Online"
    if "Store_Type" not in data.columns:
        data["Store_Type"] = "Online"
    if "Payment_Method" not in data.columns:
        data["Payment_Method"] = "Card"
    if "Customer_Category" not in data.columns:
        data["Customer_Category"] = "General"
    if "Promotion" not in data.columns:
        data["Promotion"] = "None"
    if "Discount_Applied" not in data.columns:
        data["Discount_Applied"] = False

    if "Season" not in data.columns:
        def _infer_season(dt):
            if pd.isna(dt):
                return "Unknown"
            m = dt.month
            return "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5) else "Summer" if m in (6, 7, 8) else "Fall"
        data["Season"] = data["Timestamp"].apply(_infer_season)

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
    data["Primary_Product"] = data["Product_List"].apply(lambda products: products[0] if products else "Unknown")
    data["Transaction_Month"] = data["Timestamp"].dt.to_period("M").astype(str)

    data = data.sort_values(["Customer_Name", "Timestamp", "Transaction_ID"], kind="stable").reset_index(drop=True)
    return data


def _parquet_cache_path(csv_path: Path) -> Path:
    """Derive a deterministic cache path from the CSV file name and size."""
    stat = csv_path.stat()
    key = f"{csv_path.name}_{stat.st_size}_{stat.st_mtime}"
    short_hash = hashlib.md5(key.encode()).hexdigest()[:10]
    cache_dir = csv_path.parent.parent / PROCESSED_DIR.split("/")[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{csv_path.stem}_{short_hash}.parquet"


def load_transactions(
    source: str | Path | BinaryIO | TextIO,
    sample_frac: float | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate, and clean the source transaction CSV.

    Parameters
    ----------
    source : path or file-like
        CSV file to read.
    sample_frac : float, optional
        If set (0 < value <= 1), randomly sample this fraction of rows after
        cleaning. Useful for fast experimentation on large datasets.
    use_cache : bool
        If *source* is a file path, cache the cleaned result as Parquet for
        fast subsequent loads. Ignored for file-like uploads.
    """
    csv_path: Path | None = None
    if isinstance(source, (str, Path)):
        csv_path = Path(source)

    # Try loading from Parquet cache
    if use_cache and csv_path is not None and csv_path.exists():
        cache_path = _parquet_cache_path(csv_path)
        if cache_path.exists():
            data = pd.read_parquet(cache_path)
            # Reconstruct Product_List (stored as string in parquet)
            if "Product_List" in data.columns and data["Product_List"].dtype == object:
                data["Product_List"] = data["Product_List"].apply(
                    lambda v: ast.literal_eval(v) if isinstance(v, str) else v
                )
            quality = _compute_quality(data, len(data))
            if sample_frac is not None and 0 < sample_frac < 1:
                data = data.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            return data, quality

    # Fresh load from CSV
    try:
        raw = pd.read_csv(source, skip_blank_lines=True, low_memory=False)
    except Exception as error:  # pragma: no cover - message is for app users
        raise DataQualityError(f"The file could not be read as a CSV: {error}") from error

    input_rows = len(raw)
    data = _clean_raw(raw)

    # Save to Parquet cache
    if use_cache and csv_path is not None:
        try:
            cache_path = _parquet_cache_path(csv_path)
            # Convert Product_List to string for Parquet compatibility
            cache_data = data.copy()
            cache_data["Product_List"] = cache_data["Product_List"].apply(str)
            cache_data.to_parquet(cache_path, index=False)
        except Exception:
            pass  # Cache write failure is non-critical

    quality = _compute_quality(data, input_rows)

    if sample_frac is not None and 0 < sample_frac < 1:
        data = data.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    return data, quality
