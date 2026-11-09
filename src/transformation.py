import pandas as pd


def transform_data(df):
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"].astype("string").str.strip(),
        errors="coerce",
    )

    zero_tenure = cleaned["tenure"].eq(0)
    blank_charges = cleaned["TotalCharges"].isna()

    cleaned.loc[
        zero_tenure & blank_charges, "TotalCharges"
    ] = 0

    if cleaned["TotalCharges"].isna().any():
        raise ValueError(
            "Unresolved missing or invalid TotalCharges values."
        )

    if cleaned["MonthlyCharges"].isna().any():
        raise ValueError("Missing MonthlyCharges values detected.")

    if (cleaned["MonthlyCharges"] < 0).any():
        raise ValueError("Negative monthly charges detected.")

    if (cleaned["TotalCharges"] < 0).any():
        raise ValueError("Negative total charges detected.")

    return cleaned 