"""Clean customers and create the lab's simple engineered features."""
import pandas as pd
from .config import SPEND_COLUMNS, PURCHASE_COLUMNS

def prepare_data(raw):
    """Clean known data errors and add age, tenure, spend, and purchase totals."""
    # Work on a copy, then remove known invalid rows.
    data = raw.copy()
    data["Income"] = pd.to_numeric(data["Income"], errors="coerce")
    data = data[data["Year_Birth"] >= 1920]
    data = data[data["Income"] != 666666]
    # Merge uncommon marital labels into the single category.
    data["Marital_Status"] = data["Marital_Status"].replace({"Absurd": "Single", "YOLO": "Single", "Alone": "Single"})
    # Parse dates in day-first order and derive the requested features.
    dates = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors="coerce")
    data["Age"] = 2014 - data["Year_Birth"]
    data["Tenure_Days"] = (pd.Timestamp("2014-12-31") - dates).dt.days
    data["Total_Spend"] = data[SPEND_COLUMNS].sum(axis=1)
    data["Total_Purchases"] = data[PURCHASE_COLUMNS].sum(axis=1)
    # Remove identifiers, raw date, and constant fields from modeling data.
    return data.drop(columns=["ID", "Dt_Customer", "Year_Birth", "Z_CostContact", "Z_Revenue"])
