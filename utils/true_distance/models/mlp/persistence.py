import json

import torch

from utils.true_distance.models.mlp.model import TorchMLP


def save_torch_mlp(model, path, input_dim, hidden_dims, dropout):
  torch.save(model.state_dict(), f"{path}.pt")
  with open(f"{path}.json", "w") as f:
    json.dump(
      {"input_dim": input_dim, "hidden_dims": list(hidden_dims), "dropout": dropout},
      f,
    )


def load_torch_mlp(path, device="cpu"):
  with open(f"{path}.json") as f:
    config = json.load(f)

  model = TorchMLP(
    input_dim=config["input_dim"],
    hidden_dims=config["hidden_dims"],
    dropout=config["dropout"],
  )
  model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
  model.to(device)
  model.eval()
  return model
