
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "customers_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

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

def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def prepare_data(df):
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # customerID is an identifier, not a useful predictor here.
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
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run the data pipeline first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X_train, X_test, y_train, y_test = prepare_data(df)

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}

    for name, classifier in models.items():
        model = Pipeline([
            ("preprocessing", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

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

        print(f"\n{name}")
        print(pd.Series(result).to_string())

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Choose the model using training cross-validation, not test scores.
    best_name = results_df.loc[
        results_df["cv_f1_mean"].idxmax(), "model"
    ]
    best_model = fitted_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0,
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n{report}",
        encoding="utf-8",
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Stayed", "Churned"],
    )
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "best_model_confusion_matrix.png")
    plt.close()

    summary = {
        "selected_model": best_name,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "target": "Churn",
        "positive_class": "Yes",
        "selection_metric": "Mean 5-fold CV F1",
    }

    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nSelected model:", best_name)


if __name__ == "__main__":
    main()
