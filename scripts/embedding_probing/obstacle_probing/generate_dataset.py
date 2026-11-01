from utils.embedding_probing.experiments.obstacle_probing.generate_random_obstacles import (
    generate_random_obstacles_dataset,
)
from datetime import datetime
import argparse
import os


def main(num_samples: int, device: str, seed: int, batch_size: int):
    dataset = generate_random_obstacles_dataset(
        num_samples=num_samples,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )

    os.makedirs("data", exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    safe_path = f"data/{current_datetime}_{num_samples}_random_obstacles_dataset.pkl"

    dataset.to_pickle(safe_path)
    print(f"Saved random-obstacles dataset to {safe_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset with per-sample randomized door "
                     "position/size, for the obstacle-probing diagnostic "
                     "(can a single JEPA embedding be decoded back into the "
                     "door's position/size on the dividing wall?)"
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
        default=42,
        help="Random seed for reproducibility - controls both the door "
             "geometry resampled per sample and the agent/goal positions",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Samples per encoder batch (2x this many images per call) - raise this to make better use of a GPU",
    )

    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
    )
