from utils.true_distance.data.splitter import position_dataset_to_xy
from utils.true_distance.training import log_experiment, train_and_compare_models
import argparse
import pandas as pd


def main(dataset_path, n_trials, device, models_dir, log_path, seed, n_jobs, tuning_sample_size):
    dataset = pd.read_pickle(dataset_path)
    X, y = position_dataset_to_xy(dataset)

    results = train_and_compare_models(
        X, y,
        dataset_size=len(X),
        n_trials=n_trials,
        device=device,
        models_dir=models_dir,
        seed=seed,
        n_jobs=n_jobs,
        tuning_sample_size=tuning_sample_size,
    )

    log_experiment(results.to_dict("records"), path=log_path)

    print(results)
    print(f"Logged results to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare models that predict (x, y) position from a single "
                     "JEPA embedding - diagnostic to check whether position is recoverable "
                     "from the embedding at all, independent of the distance-prediction task."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the .pkl dataset file (same format as the distance task, "
             "from scripts/true_distance/generate_dataset.py) - agent/goal embeddings "
             "and positions are extracted and deduplicated into single-embedding rows.",
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
        default="models/position_probing",
        help="Directory to save trained models into",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="experiment_log_position_probing.csv",
        help="Path to the persistent experiment-results CSV log "
             "(kept separate from the distance-prediction log since the metrics "
             "measure position error, not distance error)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split and model training",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="CPU cores for Random Forest (None=1 core, -1=all cores). "
             "Safe default of 1 core - pass a higher number explicitly on a shared machine with headroom.",
    )

    parser.add_argument(
        "--tuning_sample_size",
        type=int,
        default=None,
        help="If set, each Optuna trial trains on a random subset of this many "
             "Train rows instead of the full Train set - keeps tuning cost roughly "
             "constant regardless of dataset size (useful at e.g. 1M samples). "
             "The final fit after tuning always uses the full Train set.",
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        n_trials=args.n_trials,
        device=args.device,
        models_dir=args.models_dir,
        log_path=args.log_path,
        seed=args.seed,
        n_jobs=args.n_jobs,
        tuning_sample_size=args.tuning_sample_size,
    )
