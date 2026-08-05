"""
Command-line inference script for the customer churn model.

Usage:
    python predict.py --input new_customers.csv --output predictions.csv
"""

import argparse
import sys

import joblib
import pandas as pd

MODEL_PATH = "model/customer_churn_model.pkl"

# Raw columns the pipeline's ColumnTransformer expects (order does not matter,
# but every column below must be present in the input CSV).
REQUIRED_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    X = df[REQUIRED_COLUMNS]
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, list(model.classes_).index("Yes")]
    out = df.copy()
    out["Churn_Prediction"] = preds
    out["Churn_Probability"] = proba.round(4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the .pkl model file.")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        df = pd.read_csv(args.input)
        result = predict(model, df)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} predictions to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
