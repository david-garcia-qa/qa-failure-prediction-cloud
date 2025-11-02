import sqlite3
import torch
import pandas as pd
import joblib
from pathlib import Path
from model import FailurePredictor

DB_PATH = Path("data/qa_runs.db")

def load_latest_tests_from_sql():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM test_run_history", conn)
    conn.close()
    return df

def predict_risk(checkpoint="model_checkpoint.pt", preproc_path="preprocessor.joblib"):
    # 1. Load data (all currents tests)
    df = load_latest_tests_from_sql()

    # 2. Load preprocessor and model
    preprocessor = joblib.load(preproc_path)

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
    X_proc = preprocessor.transform(X)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    X_t = torch.tensor(X_proc).float()

    model = FailurePredictor(input_dim=X_proc.shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = model(X_t).numpy().flatten()

    # 3. Build the dataframe result
    df_out = df.copy()
    df_out["predicted_fail_probability"] = probs
    df_out["risk_level"] = pd.cut(
        probs,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )

    # 4. Save in DB (table predicted_risks)
    conn = sqlite3.connect(DB_PATH)
    df_out_to_save = df_out[[
        "test_name",
        "module",
        "bug_severity",
        "failed_last_run",
        "failed_last_3_runs",
        "flaky",
        "predicted_fail_probability",
        "risk_level",
    ]]
    df_out_to_save.to_sql("predicted_risks", conn, if_exists="replace", index=False)
    conn.close()

    return df_out_to_save.sort_values(
        by="predicted_fail_probability", ascending=False
    )

if __name__ == "__main__":
    report_df = predict_risk()
    print(report_df.head(10))
    print("\nPredictions saved back into SQLite table 'predicted_risks'.")
