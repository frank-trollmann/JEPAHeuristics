import optuna

from utils.true_distance.models.mlp.train import train_torch_mlp


def mlp_params_to_kwargs(params):
  """
  Optuna output to MLP input 
  """
  n_layers = params["n_layers"]
  hidden_dims = tuple(params[f"hidden_dim_{i}"] for i in range(n_layers))
  return {
    "hidden_dims": hidden_dims,
    "dropout": params["dropout"],
    "lr": params["lr"],
    "weight_decay": params["weight_decay"],
    "batch_size": params["batch_size"],
  }


def _objective(trial, X_train, y_train, X_val, y_val, device, seed):
  n_layers = trial.suggest_int("n_layers", 1, 3)
  hidden_dims = tuple(
    trial.suggest_int(f"hidden_dim_{i}", 32, 512, log=True) for i in range(n_layers)
  )
  dropout = trial.suggest_float("dropout", 0.0, 0.5)
  lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
  weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
  batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

  _, best_val_mae = train_torch_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dims=hidden_dims,
    dropout=dropout,
    lr=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
    device=device,
    seed=seed,
  )
  return best_val_mae


def tune_mlp(X_train, y_train, X_val, y_val, n_trials=50, device="cpu", seed=None):
  """
  Optuna hyperparameter search for the TorchMLP, scored on the held-out
  validation set (MAE, lower is better).

  Returns: (best_params, study) - pass best_params through
  mlp_params_to_kwargs() before calling train_torch_mlp() with it.
  """
  sampler = optuna.samplers.TPESampler(seed=seed)
  study = optuna.create_study(direction="minimize", sampler=sampler)
  study.optimize(
    lambda trial: _objective(trial, X_train, y_train, X_val, y_val, device, seed or 42),
    n_trials=n_trials,
  )
  return study.best_params, study
