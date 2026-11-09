# ChurnGuard: Customer Churn Prediction and Data Pipeline

## Overview

ChurnGuard is a machine learning and data engineering project that predicts customer churn using the IBM Telco Customer
Churn sample dataset.

The project demonstrates data ingestion, validation, transformation, exploratory data analysis, machine learning,
SQLite storage, logging, automated tests, and CI.

## Problem Statement

Customer churn can affect revenue and long-term customer relationships. This project explores customer characteristics
and builds classification models to identify customers who may leave a service provider.

## Dataset
This project uses the IBM Telco Customer Churn dataset.
The dataset contains customer information, account details, subscribed services, billing information, and a churn label indicating whether a customer left the company.

**Dataset source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The raw dataset is not committed to this repository.

## Technologies

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Database:** SQLite, SQL
- **Testing:** Pytest
- **Version Control:** Git and GitHub
- **Automation:** GitHub Actions

## Project Workflow

1. Ingest the raw customer dataset.
2. Validate required columns and customer identifiers.
3. Clean and transform the data.
4. Save the processed dataset.
5. Store customer records and pipeline history in SQLite.
6. Perform EDA and generate charts.
7. Train a baseline and three classification models.
8. Compare models using cross-validation and test metrics.
9. Run automated data pipeline tests.
10. Run the workflow through a single command.

## Setup

Create a virtual environment and install dependencies:

    python -m venv .venv
    python -m pip install -r requirements.txt

Activate the environment using the command appropriate
for your operating system.

## Run the Project

    python -m src.run_all

Run the pipeline independently:

    python -m src.pipeline

Run SQL queries:

    python -m src.database

Run tests:

    python -m pytest -v

## Evaluation

The project compares a Dummy Classifier baseline, Logistic Regression, Decision Tree, and Random Forest.

Metrics include accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

Actual model results are generated during execution and saved in the reports directory.

## Limitations

The dataset is a static sample. This project demonstrates a repeatable local pipeline, not a real-time production
system. Predictions do not guarantee that a customer will leave.

## Future Improvements

- Model hyperparameter tuning
- Data drift monitoring
- Scheduled pipeline execution
- API-based predictions
- Containerization and deployment
