from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import DataQualityError
from src.models.purchase_predictor import train_and_save
from src.utils.config import AVAILABLE_MODELS, DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Customer Intelligence Platform ML pipeline.")
    parser.add_argument("--data", default="data/raw/Retail_Transactions_Dataset.csv", help="Path to the input transaction CSV.")
    parser.add_argument("--output", default="models/purchase_predictor.joblib", help="Where to save the trained model.")
    parser.add_argument("--horizon", type=int, default=365, help="Days used to define a future purchase (minimum: 30).")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Which classifier to train (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        metavar="FOLDS",
        help="Run K-fold cross-validation before final training (e.g. --cv 5). 0 = skip CV.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help="Sample this fraction of data for quick experimentation (e.g. --sample 0.1 for 10%%).",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Disable SHAP explainability plot generation.",
    )
    args = parser.parse_args()

    try:
        artifact = train_and_save(
            args.data, args.output, args.horizon,
            model_name=args.model, cv_folds=args.cv, sample_frac=args.sample,
            compute_shap=not args.no_shap,
        )
    except (DataQualityError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}") from error

    summary = artifact["training_summary"]
    print("Training completed successfully.")
    print(f"Model saved to: {Path(args.output).resolve()}")
    if artifact.get("mlflow_run_id"):
        print(f"MLflow Run ID: {artifact['mlflow_run_id']}")
    print(f"Valid transactions: {artifact['quality']['valid_rows']}")
    print(f"Model type: {summary['model_type']}")
    print(f"Positive examples: {summary['positive_examples']}")

    if "accuracy" in summary:
        print(f"\nHoldout metrics:")
        print(f"  Accuracy:  {summary['accuracy']:.4f}")
        print(f"  Precision: {summary['precision']:.4f}")
        print(f"  Recall:    {summary['recall']:.4f}")
        print(f"  F1 score:  {summary['f1_score']:.4f}")
        print(f"  ROC-AUC:   {summary.get('roc_auc', 'N/A')}")

    if "cv" in summary:
        cv = summary["cv"]
        print(f"\n{cv['n_folds']}-fold cross-validation results:")
        print(f"  {'Metric':<20s} {'Mean':>8s}  {'Std':>8s}")
        print(f"  {'-'*38}")
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            mean_val = cv["mean"].get(metric)
            std_val = cv["std"].get(metric)
            if mean_val is not None:
                print(f"  {metric:<20s} {mean_val:>8.4f}  {std_val:>8.4f}")

    if artifact.get("shap") and "importance_df" in artifact["shap"]:
        imp = artifact["shap"]["importance_df"]
        if not imp.empty:
            print("\nTop 5 Most Influential Features (SHAP):")
            for _, r in imp.head(5).iterrows():
                print(f"  - {r['Feature']:<30s} (mean |SHAP|: {r['Mean_Abs_SHAP']:.4f})")
            if artifact["shap"].get("summary_plot_path"):
                print(f"  Plots saved to: {Path(artifact['shap']['summary_plot_path']).parent.resolve()}")

    print(f"\nNote: {summary['evaluation_note']}")


if __name__ == "__main__":
    main()