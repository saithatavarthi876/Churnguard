
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = ROOT / "data" / "churnguard.db"


def load_customers(df):
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_FILE) as connection:
        df.to_sql(
            "customers",
            connection,
            if_exists="replace",
            index=False,
        )

    print(f"Loaded {len(df)} customers into SQLite.")


def record_pipeline_run(status, rows_processed=0, error_message=None):
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                rows_processed INTEGER,
                error_message TEXT
            )
        """)

        connection.execute("""
            INSERT INTO pipeline_runs
                (status, rows_processed, error_message)
            VALUES (?, ?, ?)
        """, (status, rows_processed, error_message))


def query_database():
    with sqlite3.connect(DATABASE_FILE) as connection:
        print("\nCustomer count:")
        result = connection.execute(
            "SELECT COUNT(*) FROM customers"
        )
        print(result.fetchone()[0])

        print("\nCustomers by churn:")
        result = connection.execute("""
            SELECT Churn, COUNT(*)
            FROM customers
            GROUP BY Churn
        """)
        for row in result.fetchall():
            print(row)

        print("\nAverage monthly charges by contract:")
        result = connection.execute("""
            SELECT Contract, ROUND(AVG(MonthlyCharges), 2)
            FROM customers
            GROUP BY Contract
            ORDER BY AVG(MonthlyCharges) DESC
        """)
        for row in result.fetchall():
            print(row)

        print("\nRecent pipeline runs:")
        result = connection.execute("""
            SELECT run_id, run_time, status, rows_processed,
                   error_message
            FROM pipeline_runs
            ORDER BY run_id DESC
            LIMIT 10
        """)
        for row in result.fetchall():
            print(row)


if __name__ == "__main__":
    query_database()
