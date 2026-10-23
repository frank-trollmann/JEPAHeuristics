import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.utils.dlpack
from cuml.linear_model import LinearRegression as GPULinearRegression
from cuml.ensemble import RandomForestRegressor as GPURandomForest
from datetime import datetime
import json
import os


def _to_cupy(data):
    """
    Converts cuDF/pandas/numpy data into a cupy array (GPU).
    """
    if hasattr(data, "to_cupy"):
        return data.to_cupy()
    if hasattr(data, "get"):  # already a cupy array
        return data
    return np.asarray(data)


def _to_numpy(data):
    """
    Converts cuDF/cupy/pandas data into a numpy array (CPU).
    """
    if hasattr(data, "to_numpy"):  # cudf.Series / pandas.Series
        return data.to_numpy()
    if hasattr(data, "get"):  # cupy-Array
        return data.get()
    return np.asarray(data)


def _to_torch_tensor(data, dtype=torch.float32, device="cuda"):
    """
    Converts cuDF/cupy/pandas data into a torch tensor losslessly via DLPack.

    torch.as_tensor() cannot handle cupy arrays (NumPy/tensors only),
    so we go through DLPack, which additionally avoids a host copy.
    """
    # torch dtype -> numpy dtype (for astype/comparisons)
    np_dtype = torch.empty(0, dtype=dtype).numpy().dtype

    array = _to_cupy(data)

    if array.dtype != np_dtype:
        array = array.astype(np_dtype)

    # DLPack requires contiguous memory
    if not array.flags.c_contiguous:
        import cupy
        array = cupy.ascontiguousarray(array)

    try:
        # prefer the DLPack protocol (zero-copy, no host round-trip)
        tensor = torch.from_dlpack(array)
    except (AttributeError, RuntimeError, TypeError):
        # older CuPy versions without __dlpack__: legacy capsule
        tensor = torch.utils.dlpack.from_dlpack(array.toDlpack())

    return tensor.to(device)

def split_gpu_data(df, target_col='heuristic', train_size=0.8, random_state=None):
    """
    Splits a cuDF DataFrame into training and testing sets for X and y.
    
    Args:
        df: The cuDF DataFrame containing features and target.
        target_col: The name of the column to be used as target (y).
        train_size: Proportion of the dataset to include in the train split.
        random_state: Optional seed for a reproducible shuffle.
        
    Returns:
        x_train, x_test, y_train, y_test as cuDF objects.
   """
    # Reset the index: .loc with a shuffled label array breaks on duplicate labels
    df = df.reset_index(drop=True)

    # cuDF doesn't have a direct train_test_split like sklearn,
    # so we shuffle the frame and slice by position.
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(len(shuffled) * train_size)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]

    x_train = train_df.drop(columns=[target_col])
    x_test = test_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return x_train, x_test, y_train, y_test

def convert_pickle_to_parquet(pickle_path: str) -> str:
  """
  Converts a JEPA heuristic dataset from pickle (.pkl) to parquet (.parquet).
  Optimizes memory by casting float64 to float32.

  Args:
      pickle_path: Path to the source .pkl file (e.g., 'data/2023-10-27_14-30-05_10000_batch_heuristic_dataset.pkl')

  Returns:
      The path to the created .parquet file.
  """
  if not pickle_path.endswith(".pkl"):
    raise ValueError("The provided file must be a .pkl file.")

  # Generate parquet path by replacing extension
  parquet_path = pickle_path.replace(".pkl", ".parquet")

  # 1. Load pickle using pandas (most reliable for .pkl)
  df_pandas = pd.read_pickle(pickle_path)

  # 2. Convert to Polars for efficient casting and writing
  df = pl.DataFrame(df_pandas)

  # 3. Optimize: Cast all float64 to float32 to save 50% space and speed up GPU loading
  # We use a direct expression without the list to avoid some Linter confusion
  df = df.with_columns(
      pl.col(pl.Float64).cast(pl.Float32)
  )

  # 4. Save as Parquet
  df.write_parquet(parquet_path, compression="snappy")

  return parquet_path


def train_gpu_linear_regression(X, y):
    """
    Trains a Linear Regression model using cuML on GPU.
    X: cudf.DataFrame or cupy array
    y: cudf.Series or cupy array
    """
    model = GPULinearRegression()
    model.fit(X, y)
    return model

def train_gpu_random_forest(X, y, n_estimators=100):
    """
    Trains a Random Forest model using cuML on GPU.
    """
    model = GPURandomForest(n_estimators=n_estimators)
    model.fit(X, y)
    return model

def train_gpu_mlp(X, y, hidden_layers=(64, 32), epochs=100, batch_size=1024, lr=0.001):
    """
    Trains a Multi-Layer Perceptron using PyTorch on GPU.
    Handles the conversion from cuDF to Torch Tensors.
    """
    # Convert cuDF/cupy to Torch (via DLPack, torch.as_tensor cannot handle cupy arrays)
    X_torch = _to_torch_tensor(X, dtype=torch.float32)
    y_torch = _to_torch_tensor(y, dtype=torch.float32).view(-1, 1)

    # Define simple MLP
    layers = []
    input_dim = X_torch.shape[1]
    prev_dim = input_dim
    for h_dim in hidden_layers:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = h_dim
    layers.append(torch.nn.Linear(prev_dim, 1))
    
    model = torch.nn.Sequential(*layers).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mini-batch training loop for memory efficiency with 10^7 points
    model.train()
    for epoch in range(epochs):
        # Simple shuffling and batching
        permutation = torch.randperm(X_torch.size(0))
        for i in range(0, X_torch.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_torch[indices], y_torch[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_gpu_model(model, X_test, y_test, model_type='cuml'):
    """
    Evaluates a model and returns standard metrics.
    """
    if model_type == 'cuml':
        y_pred = model.predict(X_test)
    else: # PyTorch
        model.eval()
        with torch.no_grad():
            X_t = _to_torch_tensor(X_test, dtype=torch.float32)
            y_pred = model(X_t).cpu().numpy().flatten()

    # cuML returns cupy arrays, sklearn metrics need real numpy arrays
    y_pred_np = _to_numpy(y_pred)
    y_test_np = _to_numpy(y_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Convert metrics to float because numpy types aren't JSON serializable
    results = {
        "MAE": float(mean_absolute_error(y_test_np, y_pred_np)),
        "MSE": float(mean_squared_error(y_test_np, y_pred_np)),
        "R2": float(r2_score(y_test_np, y_pred_np))
    }

    # Save results to /data/MLresults
    results_dir = "data/MLresults"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else 'model'
    filename = f"{timestamp}_{model_name}_results.json"
    save_path = os.path.join(results_dir, filename)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results



