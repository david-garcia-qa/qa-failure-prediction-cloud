import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from pathlib import Path

DB_PATH = Path("data/qa_runs.db")

def load_raw_data_from_sql():
    """
    Read the history of tests from SQLite base.
    Return DataFrame Pandas.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM test_run_history", conn)
    conn.close()
    return df

def build_feature_pipeline():
    """
    - One-hot encode categorical features
    - Keep numeric features
    """
    categorical_cols = ["module", "bug_severity", "last_status"]
    numeric_cols = ["execution_time_ms", "failed_last_run", "failed_last_3_runs", "flaky"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return preprocessor, categorical_cols + numeric_cols

def prepare_datasets(test_size=0.2, random_state=42):
    df = load_raw_data_from_sql()

    # Target
    y = df["will_fail_next_run"].astype(int)

    # Features
    feature_cols = [
        "module",
        "bug_severity",
        "last_status",
        "execution_time_ms",
        "failed_last_run",
        "failed_last_3_runs",
        "flaky",
    ]
    X = df[feature_cols]

    # Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        df_train,
        df_test,
    ) = train_test_split(
        X,
        y,
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, _ = build_feature_pipeline()

    # Fit on train only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Sparse -> dense
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()

    return (
        X_train_processed.astype(np.float32),
        X_test_processed.astype(np.float32),
        y_train.to_numpy().astype(np.float32),
        y_test.to_numpy().astype(np.float32),
        preprocessor,
        df_train,
        df_test,
    )

if __name__ == "__main__":
    Xtr, Xte, ytr, yte, prep, dftr, dfte = prepare_datasets()
    print("Train shape:", Xtr.shape)
    print("Test shape:", Xte.shape)
    print("Positives in train:", ytr.sum(), "/", len(ytr))
