from utils.generate_jepa_heuristic_dataset import generate_jepa_heuristic_dataset
from datetime import datetime
import argparse
import concurrent.futures
import os

def generate_worker(num_samples: int, model: str):
    """
    Worker function for parallel execution.
    Loads the model in a separate process and generates the dataset.
    """
    try:
        print(f"Starting generation for {num_samples} samples...")
        
        match str(model):
            case "tworoom":
                # Generate the dataset
                dataset = generate_jepa_heuristic_dataset(num_samples=num_samples, checkpoint_path=None)
                
                # Timestamp + size in the path to ensure files are distinguishable
                # Example: 2023-10-27_14-30-05_10000_heuristic_dataset.pkl
                current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                os.makedirs("data", exist_ok=True)
                safe_path = f"data/{current_datetime}_{num_samples}_heuristic_dataset.pkl"
                
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
        description="Generate multiple JEPA heuristic datasets in parallel"
    )

    # --sizes: Accepts a list of integers (e.g. --sizes 1000 10000 100000)
    # --sizes: This flag allows passing multiple integers. 
    # The 'nargs="+"' tells argparse to gather all remaining values into a list.
    # Example: --sizes 100 1000 10000
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of sample sizes to generate (e.g. --sizes 1000 10000 100000)",
    )

    # --model: Required flag for the model type
    # --model: Specifies which environment/model configuration to use.
    # We use 'choices' to restrict input to supported models only.
    parser.add_argument(
        "--model",
        type=str,
        choices=["tworoom"],
        required=True,
        help="Model used for dataset generation",
    )

    # --workers: Defines the number of concurrent processes.
    # Since each worker loads the model into GPU memory, this is the 
    # primary lever to prevent Out-of-Memory (OOM) crashes.
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel processes (keep low to avoid GPU OOM)",
    )

    args = parser.parse_args()

    print("========================================================")
    print(f"JEPA Dataset Generator")
    print(f"Model:    {args.model}")
    print(f"Sizes:     {args.sizes}")
    print(f"Workers:   {args.workers}")
    print("========================================================")

    # Use ProcessPoolExecutor for true parallelism (bypasses Python GIL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Create a list of futures for each requested size
        futures = [executor.submit(generate_worker, size, args.model) for size in args.sizes]
        
        # Collect and print results as they complete
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()

