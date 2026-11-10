# ChurnGuard: Customer Churn Prediction and Data Pipeline

Click this to view my deployed app--> https://churnguard-dsml-devops.streamlit.app/


## Overview

ChurnGuard is an end-to-end machine learning and data engineering project that predicts customer churn using the IBM Telco Customer Churn dataset.

The project combines data ingestion, validation, transformation, exploratory data analysis (EDA), machine learning, SQLite storage, automated testing, Continuous Integration (CI), and web-based deployment.

A Streamlit application allows users to enter customer details and receive churn predictions with probability scores. Jenkins automates dependency installation and test execution to validate code changes before they are merged into the main branch.

## Problem Statement

Customer churn can affect revenue and long-term customer relationships. This project explores customer characteristics and builds classification models to identify customers who may be likely to leave a service provider.

The objective is to develop a repeatable data pipeline and an interactive machine learning application that demonstrates the complete workflow, from raw data processing to model-based predictions.

## Dataset

This project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information, account details, subscribed services, billing information, and a churn label indicating whether a customer left the company.

**Dataset source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The raw dataset is not committed to this repository.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Database:** SQLite, SQL
- **Web Application:** Streamlit
- **Testing:** Pytest
- **Version Control:** Git and GitHub
- **Continuous Integration:** Jenkins
- **Deployment:** Streamlit Community Cloud

## Project Workflow

1. Ingest the raw customer dataset.
2. Validate required columns, customer identifiers, and data quality.
3. Clean and transform the data.
4. Save the processed dataset.
5. Store customer records and pipeline information in SQLite.
6. Perform exploratory data analysis and generate visualizations.
7. Train a baseline model and classification models.
8. Compare models using cross-validation and evaluation metrics.
9. Save the trained machine learning model.
10. Run automated data pipeline tests using Pytest.
11. Develop an interactive Streamlit application for customer churn prediction.
12. Use Jenkins to install dependencies and execute automated tests.
13. Deploy the Streamlit application through Streamlit Community Cloud.

## Machine Learning

The project compares the following classification models:

- Dummy Classifier (baseline)
- Logistic Regression
- Decision Tree
- Random Forest

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The selected trained model is saved using Joblib and loaded by the Streamlit application for making predictions.

## Streamlit Application

ChurnGuard provides an interactive web interface built using Streamlit.

Application features include:

- Customer information input through a web form.
- Customer churn prediction.
- Churn probability scores.


**Live Application:** [ChurnGuard – Streamlit App](YOUR_STREAMLIT_APP_URL)

## Continuous Integration with Jenkins

Jenkins is used to automate the testing process.

The CI pipeline performs the following stages:

1. **Checkout:** Retrieves the source code from the GitHub repository.
2. **Python Check:** Verifies Python availability in the Jenkins environment.
3. **Dependency Installation:** Installs the required packages from `requirements.txt`.
4. **Automated Testing:** Executes the test suite using Pytest.
5. **Build Status:** Reports whether the pipeline completed successfully or failed.

The project follows a feature-branch workflow. Changes are developed in a separate branch, tested through Jenkins, and merged into `main` after successful validation.

## Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

The deployment is connected to the GitHub repository and uses the `main` branch and `app/main.py` as its application entry point.

Changes merged into the deployed branch can trigger application redeployment.

## Setup

Clone the repository:

```bash
git clone https://github.com/saithatavarthi876/Churnguard.git
cd Churnguard
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

Run the complete data pipeline:

```bash
python -m src.run_all
```

Run the pipeline independently:

```bash
python -m src.pipeline
```

Run SQL operations:

```bash
python -m src.database
```

Run automated tests:

```bash
python -m pytest -v
```

Launch the Streamlit application locally:

```bash
python -m streamlit run app/main.py
```

## Limitations

- The project uses a static sample dataset rather than continuously updated customer data.
- The machine learning model provides predictions based on historical patterns and does not guarantee actual customer behaviour.
- SQLite prediction history is stored locally and may not persist across Streamlit Cloud restarts or redeployments.
- Jenkins currently performs CI testing; Streamlit Community Cloud handles application deployment separately.

## Future Improvements

- Model hyperparameter tuning and optimization.
- Data drift detection and model monitoring.
- Scheduled data pipeline execution.
- API-based prediction services.
- Docker-based containerization.
- Persistent cloud database integration.
- Automated deployment protection based on CI test results.

## Project Goal

ChurnGuard demonstrates an end-to-end workflow combining data engineering, machine learning, software testing, CI automation, and cloud-based application deployment in a single project.



