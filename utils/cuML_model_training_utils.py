import polars as pl
import pandas as pd
import numpy as np
import torch
from cuml.linear_model import LinearRegression as GPULinearRegression
from cuml.ensemble import RandomForestRegressor as GPURandomForest
from datetime import datetime
import json
import os


def _to_array(data):
    """
    Returns the data as a cupy array where possible (cuDF), otherwise as a
    numpy array (pandas/numpy). The name deliberately does not promise cupy:
    for CPU inputs nothing is moved to the GPU.
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


_NP_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
}


def _to_torch_tensor(data, dtype=torch.float32, device="cuda"):
    """
    Converts cuDF/cupy/pandas data into a torch tensor losslessly via DLPack.

    torch.as_tensor() cannot handle cupy arrays (NumPy/tensors only),
    so we go through DLPack, which additionally avoids a host copy.
    """
    # torch dtype -> numpy dtype (for astype/comparisons)
    np_dtype = _NP_DTYPES.get(dtype)
    if np_dtype is None:
        np_dtype = torch.empty(0, dtype=dtype).numpy().dtype

    array = _to_array(data)

    if array.dtype != np_dtype:
        array = array.astype(np_dtype)

    # DLPack requires contiguous memory (works for numpy and cupy alike)
    if not array.flags.c_contiguous:
        array = array.copy(order="C")

    # zero-copy: the returned tensor shares memory with `array`
    tensor = torch.from_dlpack(array)

    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        # plain "cuda" means "current device" - keep data where it already is
        return tensor if tensor.is_cuda else tensor.to(target)

    return tensor if tensor.device == target else tensor.to(target)

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
    # Only positional access (.iloc) is used below, so no index reset is needed.
    # The single required copy is the shuffle; the slices and drop() are views
    # in cuDF and share the underlying buffers.
    shuffled = df.sample(frac=1, random_state=random_state)

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
  del df_pandas  # avoid holding both copies in host RAM

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
        # Simple shuffling and batching (indices stay on the GPU)
        permutation = torch.randperm(X_torch.size(0), device=X_torch.device)
        for i in range(0, X_torch.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_torch[indices], y_torch[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_gpu_model(model, X_test, y_test, model_type='cuml', batch_size=4096):
    """
    Evaluates a model and returns standard metrics.

    PyTorch models are evaluated in batches so the whole test split is never
    materialised twice in VRAM.
    """
    if model_type == 'cuml':
        y_pred = model.predict(X_test)
    else: # PyTorch
        model.eval()
        device = next(model.parameters()).device
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_test), batch_size):
                batch_x = _to_torch_tensor(
                    X_test.iloc[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                preds.append(model(batch_x).squeeze(1))
        y_pred = torch.cat(preds).cpu().numpy()

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

    # Save results to data/MLresults (relative to the working directory)
    results_dir = "data/MLresults"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = type(model).__name__
    filename = f"{timestamp}_{model_name}_results.json"
    save_path = os.path.join(results_dir, filename)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results



