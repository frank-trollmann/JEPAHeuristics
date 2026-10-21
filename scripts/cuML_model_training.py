import argparse
import cudf
from utils.cuML_model_training_utils import (
    convert_pickle_to_parquet,
    split_gpu_data,
    train_gpu_linear_regression,
    train_gpu_random_forest,
    train_gpu_mlp,
    evaluate_gpu_model
)

def main():
    parser = argparse.ArgumentParser(description="GPU Accelerated ML Training for JEPA Heuristics")

    parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to the source .pkl file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["linear", "rf", "mlp", "all"],
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
        type=int, default=1024,
        help="Batch size for MLP"
    )
    
    args = parser.parse_args()

    print(f"Starting GPU Training Pipeline...")

    # 1. Convert Pickle to Parquet (if not already done)
    # This ensures we use float32 and optimized storage
    parquet_path = convert_pickle_to_parquet(args.input)
    print(f"Data converted and saved to: {parquet_path}")

    # 2. Load Data into VRAM via cuDF
    print("Loading data into GPU memory...")
    df_gpu = cudf.read_parquet(parquet_path)

    # 3. Split Data (Native GPU Split)
    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = split_gpu_data(
        df_gpu, 
        target_col='heuristic', 
        train_size=args.train_size
    )
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    # 4. Model Training & Evaluation
    results_summary = {}

    if args.model in ["linear", "all"]:
        print("\n Training Linear Regression...")
        model_lr = train_gpu_linear_regression(x_train, y_train)
        results_summary["Linear"] = evaluate_gpu_model(model_lr, x_test, y_test, model_type='cuml')
        print(f"Linear Results: {results_summary['Linear']}")

    if args.model in ["rf", "all"]:
        print("\n Training Random Forest...")
        model_rf = train_gpu_random_forest(x_train, y_train)
        results_summary["RandomForest"] = evaluate_gpu_model(model_rf, x_test, y_test, model_type='cuml')
        print(f"RandomForest Results: {results_summary['RandomForest']}")

    if args.model in ["mlp", "all"]:
        print("\n Training MLP Neural Net...")
        model_mlp = train_gpu_mlp(
            x_train, 
            y_train, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        results_summary["MLP"] = evaluate_gpu_model(model_mlp, x_test, y_test, model_type='torch')
        print(f"MLP Results: {results_summary['MLP']}")

    print("\n" + "="*30)
    print("FINAL SUMMARY")
    print("="*30)
    for model, metrics in results_summary.items():
        print(f"{model}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
