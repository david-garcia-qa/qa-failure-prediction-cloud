from infer import predict_risk

def export_report(
    output_csv="predicted_risk_report.csv",
    output_md="predicted_risk_report.md",
):
    df = predict_risk()

    df.to_csv(output_csv, index=False)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# QA Failure Risk Report\n\n")
        f.write("This report ranks automated tests by predicted probability of failure in the next run.\n\n")
        for _, row in df.iterrows():
            f.write(
                f"- {row['test_name']} ({row['module']}) "
                f"=> risk {row['risk_level']} "
                f"({row['predicted_fail_probability']:.2f})\n"
            )
        f.write("\nGenerated automatically from historical execution data.\n")

    print(f"Saved {output_csv} and {output_md}")

if __name__ == "__main__":
    export_report()
