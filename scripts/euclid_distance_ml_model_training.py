import argparse
import os

import torch

from utils.euclid_distance_ml_model_training_utils import (
    LinearRegressor,
    convert_pickle_to_parquet,
    evaluate_model,
    load_features,
    split_data,
    train_mlp,
    train_xgboost,
)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Accelerated ML Training for JEPA Heuristics (torch + XGBoost)"
    )

    parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to the source .pkl file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["linear", "xgb", "mlp", "all"],
        help="Which model to train"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Training set proportion"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs for MLP"
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=4096,
        help="Batch size for MLP training and inference"
    )
    parser.add_argument(
        "--device",
        type=str, default=None,
        choices=["cuda", "cpu"],
        help="Training device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Optional seed for a reproducible train/test split"
    )
    parser.add_argument(
        "--xgb_max_rows",
        type=int, default=0,
        help="Train XGBoost on at most N randomly sampled rows (0 = all rows)"
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    print("Starting GPU Training Pipeline...")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Convert Pickle to Parquet (if not already done)
    # This ensures we use float32 and optimized storage
    parquet_path = args.input.replace(".pkl", ".parquet")
    if os.path.exists(parquet_path):
        print(f"Using existing parquet file: {parquet_path}")
    else:
        parquet_path = convert_pickle_to_parquet(args.input)
        print(f"Data converted and saved to: {parquet_path}")

    # 2. Load features as tensors on the target device
    print("Loading data...")
    X, y = load_features(parquet_path, target_col="heuristic", device=device)
    print(f"Features: {tuple(X.shape)} "
          f"({X.numel() * X.element_size() / 1e6:.1f} MB on {X.device})")

    # 3. Split Data (on the device)
    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = split_data(
        X, y, train_size=args.train_size, seed=args.seed
    )
    del X, y
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    # 4. Model Training & Evaluation
    results_summary = {}

    if args.model in ["linear", "all"]:
        print("\nTraining Linear Regression...")
        model_lr = LinearRegressor().fit(x_train, y_train)
        results_summary["Linear"] = evaluate_model(model_lr, x_test, y_test)
        print(f"Linear Results: {results_summary['Linear']}")

    if args.model in ["xgb", "all"]:
        print("\nTraining XGBoost...")
        model_xgb = train_xgboost(
            x_train, y_train, device=device, max_rows=args.xgb_max_rows
        )
        results_summary["XGBoost"] = evaluate_model(
            model_xgb, x_test, y_test, batch_size=args.batch_size
        )
        print(f"XGBoost Results: {results_summary['XGBoost']}")

    if args.model in ["mlp", "all"]:
        print("\nTraining MLP Neural Net...")
        model_mlp = train_mlp(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results_summary["MLP"] = evaluate_model(
            model_mlp, x_test, y_test, batch_size=args.batch_size
        )
        print(f"MLP Results: {results_summary['MLP']}")

    print("\n" + "=" * 30)
    print("FINAL SUMMARY")
    print("=" * 30)
    for model, metrics in results_summary.items():
        print(
            f"{model}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}, "
            f"Pearson={metrics['Pearson']:.4f}, Spearman={metrics['Spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
