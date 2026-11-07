"""Shared pytest fixtures providing realistic synthetic transaction datasets."""

from __future__ import annotations

import io
import pandas as pd
import pytest


@pytest.fixture
def sample_csv_text() -> str:
    """Provide a minimal, valid CSV string with multiple customer transactions over time."""
    return """Transaction_ID,Date,Customer_Name,Product,Total_Items,Total_Cost,Payment_Method,City,Store_Type,Discount_Applied,Customer_Category,Season,Promotion
TX001,01-01-2023 10:00,Alice,"['Milk', 'Bread']",2,20.0,Credit Card,New York,Supermarket,FALSE,Regular,Winter,None
TX002,15-01-2023 11:30,Alice,"['Apples', 'Milk']",3,35.0,Credit Card,New York,Supermarket,TRUE,Regular,Winter,Discount
TX003,05-02-2023 14:00,Bob,"['Coffee']",1,12.5,Cash,Chicago,Convenience,FALSE,New,Winter,None
TX004,20-03-2023 09:15,Alice,"['Bread', 'Butter']",2,18.0,Credit Card,New York,Supermarket,FALSE,Regular,Spring,None
TX005,10-04-2023 16:45,Charlie,"['Cereal', 'Milk']",2,22.0,Mobile Payment,Los Angeles,Supermarket,TRUE,Regular,Spring,BOGO
TX006,01-08-2023 12:00,Bob,"['Coffee', 'Donut']",2,15.0,Cash,Chicago,Convenience,TRUE,Regular,Summer,None
TX007,15-11-2023 17:30,Alice,"['Milk', 'Cookies']",2,25.0,Credit Card,New York,Supermarket,FALSE,VIP,Fall,None
TX008,10-12-2023 18:00,David,"['Tea']",1,8.0,Cash,Houston,Convenience,FALSE,New,Winter,None
"""


@pytest.fixture
def sample_transactions(sample_csv_text: str) -> pd.DataFrame:
    """Return a cleaned transactions DataFrame loaded from the sample CSV text."""
    from src.data.loader import load_transactions

    buffer = io.StringIO(sample_csv_text)
    df, _ = load_transactions(buffer, use_cache=False)
    return df
