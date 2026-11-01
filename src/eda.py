
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


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
