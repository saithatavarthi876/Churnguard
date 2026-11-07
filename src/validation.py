import pandas as pd

REQUIRED_COLUMNS = [
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_data(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["customerID"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Invalid Churn values detected.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    return True