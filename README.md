# ChurnGuard – Customer Churn Prediction

## About the Project
ChurnGuard is a machine learning and data engineering project that aims to predict whether a customer is likely to leave a telecommunications company.
Customer churn can affect a company's revenue and long-term growth. By studying customer information such as contract type, monthly charges, tenure, and services used, this project explores how machine learning can help identify customers who may be at risk of leaving.
I am building this project to understand how a complete data workflow works, from collecting and preparing data to training a machine learning model and storing processed information.

## Project Objectives
- Understand the customer churn dataset through exploratory data analysis (EDA).
- Clean the data and handle missing or inconsistent values.
- Transform the data into a format suitable for machine learning.
- Train and evaluate classification models.
- Store processed data and results using SQLite.
- Build a repeatable pipeline for data processing and model training.
- Add basic testing, logging, and automation.

## Dataset
This project uses the IBM Telco Customer Churn dataset.
The dataset contains customer information, account details, subscribed services, billing information, and a churn label indicating whether a customer left the company.

**Dataset source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

## Technologies Used
- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Database:** SQLite, SQL
- **Testing:** Pytest
- **Version Control:** Git and GitHub
- **Automation:** GitHub Actions

## Project Workflow
1. Load the raw customer dataset.
2. Explore the data and identify patterns.
3. Clean and preprocess the dataset.
4. Store and query structured data using SQLite.
5. Train classification models to predict customer churn.
6. Evaluate model performance using suitable classification metrics.
7. Automate the workflow and add tests to check important functionality.

## Evaluation
The models will be evaluated using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.
Since identifying customers who may leave is important, recall and F1-score will also be considered rather than relying on accuracy alone.

## Project Status
This project is being developed incrementally. The goal is to build a working end-to-end pipeline while learning and applying machine learning, data engineering, SQL, testing, and automation concepts.

## What I Hope to Learn
Through this project, I aim to gain practical experience in preparing real-world data, building classification models, organizing Python code into reusable modules, managing data with SQL, testing code, and maintaining a project using Git and GitHub.
