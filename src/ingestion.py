from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def ingest_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df