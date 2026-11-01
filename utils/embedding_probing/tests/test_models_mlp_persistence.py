import json

import numpy as np
import torch

from utils.embedding_probing.models.mlp.model import TorchMLP
from utils.embedding_probing.models.mlp.persistence import load_torch_mlp, save_torch_mlp


def test_save_and_load_torch_mlp_roundtrip(tmp_path):
  torch.manual_seed(0)
  model = TorchMLP(input_dim=5, hidden_dims=(8, 4), dropout=0.1)
  model.eval()

  x = torch.randn(10, 5)
  with torch.no_grad():
    original_output = model(x)

  path = tmp_path / "mlp_model"
  save_torch_mlp(model, path, input_dim=5, hidden_dims=(8, 4), dropout=0.1)

  loaded = load_torch_mlp(path, device="cpu")
  with torch.no_grad():
    loaded_output = loaded(x)

  np.testing.assert_allclose(original_output.numpy(), loaded_output.numpy(), atol=1e-6)


def test_save_torch_mlp_writes_both_files(tmp_path):
  model = TorchMLP(input_dim=3, hidden_dims=(4,), dropout=0.0)
  path = tmp_path / "mlp_model"

  save_torch_mlp(model, path, input_dim=3, hidden_dims=(4,), dropout=0.0)

  assert (tmp_path / "mlp_model.pt").exists()
  assert (tmp_path / "mlp_model.json").exists()


def test_save_and_load_torch_mlp_roundtrip_multioutput(tmp_path):
  torch.manual_seed(0)
  model = TorchMLP(input_dim=5, hidden_dims=(8, 4), output_dim=2, dropout=0.1)
  model.eval()

  x = torch.randn(10, 5)
  with torch.no_grad():
    original_output = model(x)

  path = tmp_path / "mlp_model"
  save_torch_mlp(model, path, input_dim=5, hidden_dims=(8, 4), output_dim=2, dropout=0.1)

  loaded = load_torch_mlp(path, device="cpu")
  with torch.no_grad():
    loaded_output = loaded(x)

  assert loaded_output.shape == (10, 2)
  np.testing.assert_allclose(original_output.numpy(), loaded_output.numpy(), atol=1e-6)


def test_load_torch_mlp_defaults_output_dim_for_configs_saved_before_it_existed(tmp_path):
  model = TorchMLP(input_dim=4, hidden_dims=(8,), dropout=0.0)
  path = tmp_path / "mlp_model"
  save_torch_mlp(model, path, input_dim=4, hidden_dims=(8,), dropout=0.0)

  # simulate an old config file saved before output_dim was tracked
  with open(f"{path}.json") as f:
    config = json.load(f)
  del config["output_dim"]
  with open(f"{path}.json", "w") as f:
    json.dump(config, f)

  loaded = load_torch_mlp(path, device="cpu")
  out = loaded(torch.randn(2, 4))
  assert out.shape == (2,)


def test_load_torch_mlp_reconstructs_correct_architecture(tmp_path):
  model = TorchMLP(input_dim=6, hidden_dims=(16, 8, 4), dropout=0.3)
  path = tmp_path / "mlp_model"
  save_torch_mlp(model, path, input_dim=6, hidden_dims=(16, 8, 4), dropout=0.3)

  loaded = load_torch_mlp(path, device="cpu")

  assert isinstance(loaded, TorchMLP)
  # architecture check: predicting on correctly-shaped input shouldn't raise
  out = loaded(torch.randn(2, 6))
  assert out.shape == (2,)
