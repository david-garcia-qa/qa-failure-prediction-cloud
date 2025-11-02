import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/qa_runs.db")
CSV_PATH = Path("data/seed_test_run_history.csv")

def init_database():
    # 1. Load CSV data
    df = pd.read_csv(CSV_PATH)

    # 2. Open / Create SQLite base
    conn = sqlite3.connect(DB_PATH)

    # 3. Write the history of tests in a SQL table
    df.to_sql("test_run_history", conn, if_exists="replace", index=False)

    # 4. Create a table for the futurs predictions
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predicted_risks (
            test_name TEXT,
            module TEXT,
            bug_severity TEXT,
            failed_last_run INTEGER,
            failed_last_3_runs INTEGER,
            flaky INTEGER,
            predicted_fail_probability REAL,
            risk_level TEXT
        )
        """
    )

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH.absolute()}")

if __name__ == "__main__":
    init_database()
