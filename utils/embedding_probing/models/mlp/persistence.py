import json

import torch

from utils.embedding_probing.models.mlp.model import TorchMLP


def save_torch_mlp(model, path, input_dim, hidden_dims, dropout, output_dim=1):
  torch.save(model.state_dict(), f"{path}.pt")
  with open(f"{path}.json", "w") as f:
    json.dump(
      {
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "output_dim": output_dim,
        "dropout": dropout,
      },
      f,
    )


def load_torch_mlp(path, device="cpu"):
  with open(f"{path}.json") as f:
    config = json.load(f)

  model = TorchMLP(
    input_dim=config["input_dim"],
    hidden_dims=config["hidden_dims"],
    # .get() default keeps this loadable for configs saved before output_dim existed
    output_dim=config.get("output_dim", 1),
    dropout=config["dropout"],
  )
  model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
  model.to(device)
  model.eval()
  return model
