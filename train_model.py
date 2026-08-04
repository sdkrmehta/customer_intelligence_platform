"""Train the customer purchase prediction model from the supplied CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.retail_predictor import DataQualityError, train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Retail Customer Purchase Prediction model.")
    parser.add_argument("--data", default="Retail_Transactions_Dataset.csv", help="Path to the input transaction CSV.")
    parser.add_argument("--output", default="models/purchase_predictor.joblib", help="Where to save the trained model.")
    parser.add_argument("--horizon", type=int, default=365, help="Days used to define a future purchase (minimum: 30).")
    args = parser.parse_args()

    try:
        artifact = train_and_save(args.data, args.output, args.horizon)
    except (DataQualityError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}") from error

    summary = artifact["training_summary"]
    print("Training completed successfully.")
    print(f"Model saved to: {Path(args.output).resolve()}")
    print(f"Valid transactions: {artifact['quality']['valid_rows']}")
    print(f"Positive examples: {summary['positive_examples']}")
    print(f"Model type: {summary['model_type']}")
    print(f"Note: {summary['evaluation_note']}")


if __name__ == "__main__":
    main()
