
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def create_eda_charts(df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Work on a copy so the raw dataset remains unchanged.
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    # 1. Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "churn_distribution.png")
    plt.close()

    # 2. Churn by contract type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "churn_by_contract.png")
    plt.close()

    # 3. Tenure by churn status
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=data,
        x="tenure",
        hue="Churn",
        bins=25,
        element="step",
    )
    plt.title("Customer Tenure by Churn Status")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "tenure_by_churn.png")
    plt.close()

    # 4. Correlations between numeric features
    numeric_data = data.select_dtypes(include="number")

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png")
    plt.close()

    print("EDA charts saved in the reports folder.")

    
if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    analyze_data_quality(df)
    create_eda_charts(df)
    