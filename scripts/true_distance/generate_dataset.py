from utils.true_distance.generate_true_distance_dataset import (
    generate_true_distance_dataset,
    DistanceAlgorithm,
)
from datetime import datetime
import argparse
import os


def main(num_samples: int, device: str, seed: int, batch_size: int, algorithm: DistanceAlgorithm):
    dataset = generate_true_distance_dataset(
        num_samples=num_samples,
        seed=seed,
        device=device,
        batch_size=batch_size,
        algorithm=algorithm,
    )

    os.makedirs("data", exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    safe_path = f"data/{current_datetime}_{num_samples}_true_distance_dataset.pkl"

    dataset.to_pickle(safe_path)
    print(f"Saved true-distance dataset to {safe_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a true-distance (wall-aware) JEPA heuristic dataset"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the encoder on",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Samples per encoder batch (2x this many images per call) - raise this to make better use of a GPU",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=DistanceAlgorithm.MAZE_FORMULA.value,
        choices=[a.value for a in DistanceAlgorithm],
        help="Distance algorithm for the heuristic label - maze_formula (fast, default) or dijkstra (slow, reference/comparison only)",
    )

    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        algorithm=DistanceAlgorithm(args.algorithm),
    )
