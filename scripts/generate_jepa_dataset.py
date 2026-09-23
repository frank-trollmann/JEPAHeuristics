from utils.generate_jepa_heuristic_dataset import generate_jepa_heuristic_dataset
from datetime import datetime
import argparse

def main(num_samples: int, model: str):

  match str(model):
    case "tworoom":
      dataset = generate_jepa_heuristic_dataset(num_samples = num_samples, checkpoint_path=None)
      current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
      safe_path = "data/" + current_datetime +"_heuristic_dataset.pkl"

      dataset.to_pickle(safe_path)
      print("Saved heuristic dataset")

    case _:
      raise NotImplementedError(f"Unsupported model: {model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a JEPA heuristic dataset"
    )

    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "model",
        type=str,
        choices=["tworoom"],
        help="Model used for dataset generation",
    )

    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        model=args.model,
    )
