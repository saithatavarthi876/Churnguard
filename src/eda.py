
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


# 
def create_eda_charts(df):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = df.copy()

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    create_churn_overview(data)
    print("Churn overview chart generated.")
    create_service_pie_charts(data)
    print("Service distribution charts generated.")
    create_charges_charts(data)
    print("Charges analysis charts generated.")
    create_relationship_charts(data)
    print("Relationship analysis charts generated.")
    create_tenure_group_chart(data)
    print("Churn by tenure group chart generated.")
    print("All EDA charts generated successfully!")
# IMAGE 1: CUSTOMER CHURN OVERVIEW
def add_percentage_labels(ax):

    for bar in ax.patches:

        height = bar.get_height()

        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points"
        )

    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
def create_churn_overview(data):

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    fig.suptitle(
        "CUSTOMER CHURN - OVERALL ANALYSIS",
        fontsize=19,
        fontweight="bold"
    )

    # 1. Overall churn distribution - Pie chart
    churn_counts = data["Churn"].value_counts()

    axes[0, 0].pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#66b3ff", "#ff9999"],
        explode=[0, 0.08],
        textprops={"fontsize": 11}
    )

    axes[0, 0].set_title("1. Overall Customer Churn")

    # 2. Churn percentage by contract
    contract_rate = (
        data.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    sns.barplot(
        data=contract_rate,
        x="Contract",
        y="ChurnRate",
        hue="Contract",
        palette="Set2",
        legend=False,
        ax=axes[0, 1]
    )

    axes[0, 1].set_title("2. Churn Rate by Contract Type")
    axes[0, 1].set_ylabel("Churn Rate (%)")
    axes[0, 1].set_xlabel("Contract Type")
    add_percentage_labels(axes[0, 1])
    # 3. Churn percentage by gender
    gender_rate = (
        data.groupby("gender")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    sns.barplot(
        data=gender_rate,
        x="gender",
        y="ChurnRate",
        hue="gender",
        palette="Set2",
        legend=False,
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("3. Churn Rate by Gender")
    axes[1, 0].set_ylabel("Churn Rate (%)")
    axes[1, 0].set_xlabel("Gender")
    add_percentage_labels(axes[1, 0])
    # 4. Churn percentage by senior citizen
    senior_data = data.copy()

    senior_data["SeniorCitizen"] = (
        senior_data["SeniorCitizen"]
        .map({0: "Non-Senior", 1: "Senior"})
    )

    senior_rate = (
        senior_data.groupby("SeniorCitizen")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    sns.barplot(
        data=senior_rate,
        x="SeniorCitizen",
        y="ChurnRate",
        hue="SeniorCitizen",
        palette="Set2",
        legend=False,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title("4. Churn Rate by Senior Citizen Status")
    axes[1, 1].set_ylabel("Churn Rate (%)")
    axes[1, 1].set_xlabel("Customer Category")
    add_percentage_labels(axes[1, 1])
    # 5. Customer tenure by churn
    sns.histplot(
        data=data,
        x="tenure",
        hue="Churn",
        bins=25,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=axes[2, 0]
    )

    axes[2, 0].set_title("5. Customer Tenure by Churn Status")
    axes[2, 0].set_xlabel("Tenure (Months)")
    axes[2, 0].set_ylabel("Density")

    # 6. Overall customer summary
    total = len(data)
    churned = (data["Churn"] == "Yes").sum()
    retained = (data["Churn"] == "No").sum()
    churn_percentage = (churned / total) * 100

    axes[2, 1].axis("off")

    summary = (
        f"TOTAL CUSTOMERS\n{total:,}\n\n"
        f"CHURNED CUSTOMERS\n{churned:,}\n\n"
        f"RETAINED CUSTOMERS\n{retained:,}\n\n"
        f"OVERALL CHURN RATE\n{churn_percentage:.2f}%"
    )

    axes[2, 1].text(
        0.5, 0.5,
        summary,
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="lightblue",
            alpha=0.3
        )
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(
        REPORT_DIR / "01_customer_churn_overview.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# IMAGE 2: SERVICE DISTRIBUTION PIE CHARTS

def create_service_pie_charts(data):

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 15))

    fig.suptitle(
        "CUSTOMER SERVICE DISTRIBUTION",
        fontsize=19,
        fontweight="bold"
    )

    for ax, column in zip(axes.flatten(), service_columns):

        counts = data[column].value_counts()

        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.72,
            textprops={"fontsize": 8}
        )

        ax.set_title(
            column.replace("_", " "),
            fontsize=12,
            fontweight="bold"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        REPORT_DIR / "02_service_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# IMAGE 3: CHARGES AND CUSTOMER SPENDING

def create_charges_charts(data):

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    fig.suptitle(
        "CUSTOMER CHARGES AND SPENDING ANALYSIS",
        fontsize=18,
        fontweight="bold"
    )

    # 1. Monthly charges distribution
    sns.histplot(
        data=data,
        x="MonthlyCharges",
        bins=30,
        kde=True,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("Monthly Charges Distribution")
    axes[0, 0].set_xlabel("Monthly Charges")

    # 2. Total charges distribution
    sns.histplot(
        data=data,
        x="TotalCharges",
        bins=30,
        kde=True,
        ax=axes[0, 1]
    )

    axes[0, 1].set_title("Total Charges Distribution")
    axes[0, 1].set_xlabel("Total Charges")

    # 3. Monthly charges by churn
    sns.boxplot(
        data=data,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        legend=False,
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("Monthly Charges by Churn")
    axes[1, 0].set_xlabel("Churn Status")

    # 4. Total charges by churn
    sns.boxplot(
        data=data,
        x="Churn",
        y="TotalCharges",
        hue="Churn",
        legend=False,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title("Total Charges by Churn")
    axes[1, 1].set_xlabel("Churn Status")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        REPORT_DIR / "03_charges_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    

 


# IMAGE 5: CORRELATION AND CUSTOMER RELATIONSHIPS

def create_relationship_charts(data):

    plt.figure(figsize=(12, 8))

    plt.title(
        "CUSTOMER FEATURE CORRELATION ANALYSIS",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    # Select only numerical columns
    numeric_data = data.select_dtypes(include="number")

    # Calculate correlation
    correlation_matrix = numeric_data.corr()

    # Create correlation heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "04_relationship_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# IMAGE 6: CHURN RATE BY TENURE GROUP

def create_tenure_group_chart(data):

    chart_data = data.copy()

    chart_data["TenureGroup"] = pd.cut(
        chart_data["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-36 Months",
            "37-48 Months",
            "49-60 Months",
            "61-72 Months"
        ]
    )

    churn_rate = (
        chart_data.groupby(
            "TenureGroup", observed=False
        )["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=churn_rate,
        x="TenureGroup",
        y="ChurnRate",
        hue="TenureGroup",
        palette="viridis",
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=3
        )

    plt.title("Customer Churn Rate by Tenure Group")
    plt.xlabel("Customer Tenure")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "05_churn_by_tenure_group.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    analyze_data_quality(df)
    create_eda_charts(df)
   
