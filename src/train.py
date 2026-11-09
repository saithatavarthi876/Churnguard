
from pathlib import Path
import json
from unicodedata import name

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "customers_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def build_preprocessor(X):

    # Identify numerical and categorical columns
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Numerical data preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical data preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Apply appropriate preprocessing to each column type
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def prepare_data(df):

    # Convert target values into numbers
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Remove target and customer identifier from features
    X = df.drop(columns=["Churn", "customerID"])

    if y.isna().any():
        raise ValueError("Unexpected values in the Churn target.")

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def main():

    # Check whether processed dataset exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}. "
            "Run the data pipeline first."
        )

    # Create output directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed dataset...")
    df = pd.read_csv(DATA_FILE)

    print("Preparing training and testing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Define classification models
    models = {
        "Dummy Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}
    all_classification_reports = {}
    # Train and evaluate each model
    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        model = Pipeline([
            ("preprocessing", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])

        # Evaluate using cross-validation on training data
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )

        # Train the model on the complete training dataset
        model.fit(X_train, y_train)

        # Predict on unseen test data
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Generate classification report for the current model
        model_report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Stayed", "Churned"],
        zero_division=0,
)

        all_classification_reports[name] = model_report
        # Calculate evaluation metrics
        result = {
            "model": name,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "test_f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "test_roc_auc": roc_auc_score(
                y_test, probabilities
            ),
        }

        results.append(result)
        fitted_models[name] = model

        print(f"\n{name} results:")
        print(pd.Series(result).to_string())

    # Create model comparison DataFrame
    results_df = pd.DataFrame(results)

    
    # Save model comparison results
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Select the best model using cross-validation F1 score
    best_name = results_df.loc[
        results_df["cv_f1_mean"].idxmax(),
        "model",
    ]

    best_model = fitted_models[best_name]

    # Save the selected model
    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.joblib",
    )

    # Create a complete evaluation report
    report_content = []

    report_content.append("=" * 65)
    report_content.append("CHURNGUARD - MODEL EVALUATION REPORT")
    report_content.append("=" * 65)

    # Include classification reports for all models
    for model_name, model_report in all_classification_reports.items():

        report_content.append("\n")
        report_content.append("-" * 65)
        report_content.append(f"MODEL: {model_name}")
        report_content.append("-" * 65)
        report_content.append(model_report)

    # Include model comparison table
    report_content.append("\n")
    report_content.append("=" * 65)
    report_content.append("MODEL COMPARISON")
    report_content.append("=" * 65)

    report_content.append(
        results_df.round(4).to_string(index=False)
    )

    # Put selected model at the very end
    report_content.append("\n")
    report_content.append("=" * 65)
    report_content.append("FINAL SELECTED MODEL")
    report_content.append("=" * 65)

    report_content.append(f"Selected Model: {best_name}")
    report_content.append("Selection Metric: Mean 5-fold Cross-Validation F1 Score")
    report_content.append(
        f"Cross-validation F1 Score: "
        f"{results_df.loc[results_df['model'] == best_name, 'cv_f1_mean'].iloc[0]:.4f}"
    )

    # Save complete report
    (REPORT_DIR / "classification_report.txt").write_text(
        "\n".join(report_content),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
