from utils.true_distance.data.splitter import heuristic_dataset_to_xy
from utils.true_distance.training import log_experiment, train_and_compare_models
import argparse
import pandas as pd


def main(dataset_path, n_trials, device, models_dir, log_path, seed):
    dataset = pd.read_pickle(dataset_path)
    X, y = heuristic_dataset_to_xy(dataset)

    results = train_and_compare_models(
        X, y,
        dataset_size=len(dataset),
        n_trials=n_trials,
        device=device,
        models_dir=models_dir,
        seed=seed,
    )

    log_experiment(results.to_dict("records"), path=log_path)

    print(results)
    print(f"Logged results to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare heuristic-distance models on a generated dataset"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the .pkl dataset file (from scripts/true_distance/generate_dataset.py)",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials per tunable model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run XGBoost/MLP on",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained models into",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="experiment_log.csv",
        help="Path to the persistent experiment-results CSV log",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split and model training",
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        n_trials=args.n_trials,
        device=args.device,
        models_dir=args.models_dir,
        log_path=args.log_path,
        seed=args.seed,
    )
