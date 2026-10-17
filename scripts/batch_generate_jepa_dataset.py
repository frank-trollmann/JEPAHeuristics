from utils.batchpredict_generate_jepa_heuristic_dataset import batchpredict_generate_jepa_heuristic_dataset
from datetime import datetime
import argparse
import concurrent.futures
import os

def generate_worker(num_samples: int, model: str, batch_size: int):
    """
    Worker function for parallel execution.
    Each worker handles one specific dataset size using batch encoding.
    """
    try:
        print(f"Starting batch generation for {num_samples} samples (batch_size={batch_size})...")
        
        match str(model):
            case "tworoom":
                # Use the new batch prediction utility
                dataset = batchpredict_generate_jepa_heuristic_dataset(
                    num_samples=num_samples, 
                    batch_size=batch_size, 
                    checkpoint_path=None
                )
                
                # Timestamp + size in the path to ensure files are distinguishable
                # Example: 2023-10-27_14-30-05_10000_batch_heuristic_dataset.pkl
                current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                os.makedirs("data", exist_ok=True)
                safe_path = f"data/{current_datetime}_{num_samples}_batch_heuristic_dataset.pkl"
                
                dataset.to_pickle(safe_path)
                return f"Successfully saved {num_samples} samples to {safe_path}"
            
            case _:
                return f"Error: Unsupported model {model}"
                
    except Exception as e:
        # Catch exceptions so a single failed dataset generation 
        # doesn't abort the entire parallel batch.
        return f"Error generating {num_samples} samples: {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple JEPA heuristic datasets in parallel using batch prediction"
    )

    # --sizes: List of sample sizes to generate
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of sample sizes to generate (e.g. --sizes 1000 10000 100000)",
    )

    # --model: Model type
    parser.add_argument(
        "--model",
        type=str,
        choices=["tworoom"],
        required=True,
        help="Model used for dataset generation",
    )

    # --batch_size: How many samples the GPU processes at once
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="GPU batch size (increase for speed, decrease if OOM)",
    )

    # --workers: Number of parallel processes
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel processes (keep low to avoid GPU OOM)",
    )

    args = parser.parse_args()

    print("========================================================")
    print(f"JEPA Batch Dataset Generator")
    print(f"Model:       {args.model}")
    print(f"Sizes:       {args.sizes}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Workers:     {args.workers}")
    print("========================================================")

    # Use ProcessPoolExecutor for true parallelism (bypasses Python GIL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Create a list of futures for each requested size
        futures = [
            executor.submit(generate_worker, size, args.model, args.batch_size) 
            for size in args.sizes
        ]
        
        # Collect and print results as they complete
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
