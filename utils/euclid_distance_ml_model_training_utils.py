def train_xgboost(X, y, n_estimators: int = 500, max_depth: int = 6,
                  learning_rate: float = 0.1, device: str = "cuda",
                  max_rows: int = 0, standardize: bool = True,
                  subsample: float = 0.8, colsample_bytree: float = 0.8):
    """
    Gradient boosted trees on the GPU (XGBoost `tree_method="hist"`).

    XGBoost takes host/NumPy input here, so the training matrix is copied back
    once; prediction happens on the GPU.

    `max_rows > 0` trains on a random subset. Tree models plateau early, and
    for multi-million row datasets the host copy of X is by far the largest
    allocation of the whole pipeline.
    """
    # On macOS, torch and xgboost ship two libomp runtimes; spawning more than
    # one OpenMP thread then crashes with "OMP: Error #179". Linux is unaffected.
    n_jobs = 1 if sys.platform == "darwin" else -1

    num_rows = X.shape[0]
    if 0 < max_rows < num_rows:
        print(f"XGBoost: training on a random subset of {max_rows:,} "
              f"of {num_rows:,} rows")
        subset = torch.randperm(num_rows, device=X.device)[:max_rows]
        X, y = X[subset], y[subset]

    # XGBoost benefits from standardization (tree models are invariant to monotonic
    # transformations, but hist-based splitting is numerically more stable with
    # zero-mean, unit-variance features).
    standardizer = Standardizer() if standardize else None
    X_np = standardizer.fit_transform(X) if standardizer else _to_numpy(X)
    y_np = _to_numpy(y)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
    )
    model.fit(X_np, y_np)
    model.standardizer = standardizer  # attach for later transform in predict()
    return model
