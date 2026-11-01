
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORT_DIR = ROOT / "reports"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


def inspect_data(df):
    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names and data types:")
    print(df.dtypes)

    print("\nDescriptive statistics:")
    print(df.describe(include="all").transpose())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df.describe(include="all").transpose().to_csv(
        REPORT_DIR / "descriptive_statistics.csv"
    )

def analyze_data_quality(df):
    print("\nMissing values by column:")
    print(df.isnull().sum())

    print("\nBlank strings by column:")
    print(
        df.astype("string")
        .apply(lambda column: column.str.strip().eq("").sum())
    )

    print("\nExact duplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate customer IDs:")
    print(df["customerID"].duplicated().sum())

    print("\nChurn distribution:")
    print(df["Churn"].value_counts(dropna=False))

    print("\nChurn proportions:")
    print(df["Churn"].value_counts(normalize=True, dropna=False))
    
if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    analyze_data_quality(df)
