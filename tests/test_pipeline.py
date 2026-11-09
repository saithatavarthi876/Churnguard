
import pandas as pd
import pytest

from src.validation import validate_data
from src.transformation import transform_data


def sample_data():
    return pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "tenure": [1, 0, 12],
        "MonthlyCharges": [50.0, 25.0, 70.0],
        "TotalCharges": ["50.0", " ", "840.0"],
        "Churn": ["No", "Yes", "No"],
    })


def test_valid_data_passes():
    assert validate_data(sample_data()) is True


def test_missing_column_fails():
    df = sample_data().drop(columns=["Churn"])

    with pytest.raises(ValueError):
        validate_data(df)


def test_invalid_churn_value_fails():
    df = sample_data()
    df.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError):
        validate_data(df)


def test_duplicate_customer_ids_fail():
    df = sample_data()
    df.loc[1, "customerID"] = "C001"

    with pytest.raises(ValueError):
        validate_data(df)


def test_zero_tenure_blank_charges_become_zero():
    cleaned = transform_data(sample_data())

    assert cleaned.loc[1, "TotalCharges"] == 0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_invalid_charge_fails():
    df = sample_data()
    df.loc[2, "TotalCharges"] = "invalid"

    with pytest.raises(ValueError):
        transform_data(df)


def test_duplicate_rows_are_removed():
    df = sample_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = transform_data(df)

    assert len(cleaned) == 3
