import numpy as np
import optuna
import torch

from utils.true_distance.models.mlp.model import TorchMLP
from utils.true_distance.models.mlp.train import train_torch_mlp, predict_torch_mlp
from utils.true_distance.models.mlp.tune import tune_mlp, mlp_params_to_kwargs

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _synthetic_data(n=100, dim=5, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, dim)).astype(np.float32)
  y = (X.sum(axis=1) * 10 + rng.normal(scale=0.1, size=n)).astype(np.float32)
  return X, y


def test_torch_mlp_output_shape():
  model = TorchMLP(input_dim=5, hidden_dims=(8, 4))
  x = torch.randn(10, 5)
  out = model(x)
  assert out.shape == (10,)


def test_train_torch_mlp_returns_model_and_val_mae():
  X_train, y_train = _synthetic_data(n=80, seed=1)
  X_val, y_val = _synthetic_data(n=20, seed=2)

  model, best_val_mae = train_torch_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims=(16, 8), max_epochs=20, patience=5, device="cpu",
  )

  assert isinstance(model, TorchMLP)
  assert best_val_mae >= 0
  assert np.isfinite(best_val_mae)


def test_train_torch_mlp_learns_something_on_easy_data():
  # y is a simple linear function of X -> should be easy to fit well
  X_train, y_train = _synthetic_data(n=200, dim=3, seed=1)
  X_val, y_val = _synthetic_data(n=50, dim=3, seed=2)

  model, best_val_mae = train_torch_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims=(32, 16), max_epochs=100, patience=15, device="cpu",
  )

  # mean |y_val| is roughly 10*sqrt(3)*mean|N(0,1)| ~ a few tens;
  # a trained model should clearly beat predicting zero/noise
  naive_mae = np.mean(np.abs(y_val))
  assert best_val_mae < naive_mae


def test_predict_torch_mlp_matches_model_output_shape():
  X_train, y_train = _synthetic_data(n=50, seed=1)
  X_val, y_val = _synthetic_data(n=10, seed=2)

  model, _ = train_torch_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims=(8,), max_epochs=5, patience=5, device="cpu",
  )

  y_pred = predict_torch_mlp(model, X_val, device="cpu")
  assert y_pred.shape == (10,)


def test_tune_mlp_returns_valid_params_and_study():
  X_train, y_train = _synthetic_data(n=60, dim=3, seed=1)
  X_val, y_val = _synthetic_data(n=20, dim=3, seed=2)

  best_params, study = tune_mlp(
    X_train, y_train, X_val, y_val, n_trials=2, device="cpu", seed=0
  )

  assert "n_layers" in best_params
  assert "dropout" in best_params
  assert len(study.trials) == 2
  assert study.best_value >= 0


def test_mlp_params_to_kwargs_converts_flat_params():
  params = {
    "n_layers": 2,
    "hidden_dim_0": 64,
    "hidden_dim_1": 32,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
  }
  kwargs = mlp_params_to_kwargs(params)
  assert kwargs["hidden_dims"] == (64, 32)
  assert kwargs["dropout"] == 0.2
  assert kwargs["lr"] == 1e-3
  assert kwargs["weight_decay"] == 1e-5
  assert kwargs["batch_size"] == 64


def test_tune_mlp_best_params_are_usable_via_conversion():
  X_train, y_train = _synthetic_data(n=60, dim=3, seed=1)
  X_val, y_val = _synthetic_data(n=20, dim=3, seed=2)

  best_params, _ = tune_mlp(X_train, y_train, X_val, y_val, n_trials=2, device="cpu", seed=0)
  kwargs = mlp_params_to_kwargs(best_params)

  model, _ = train_torch_mlp(
    X_train, y_train, X_val, y_val, max_epochs=5, patience=5, device="cpu", **kwargs
  )
  y_pred = predict_torch_mlp(model, X_val, device="cpu")
  assert y_pred.shape == (20,)
