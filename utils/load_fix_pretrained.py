"""
    This is a fixed implementationversion of the load_pretrained function in stable_worldmodel.wm.utils
    The issue solved here is that the model checkpoints on huggingface contain unexpected keys (for example "encoder.encoder.layer.0.attention.attention.query.weight" instead of "encoder.layer.0.attention.attention.query.weight")
    This class implements the load_pretrained function in a way that checks for this issue and fixes it.
"""
import torch
import stable_worldmodel.wm.utils as utils
from hydra.utils import instantiate


def load_fix_pretrained(name: str, cache_dir: str = None, extra_args=None):
    """Load a model from a local checkpoint or a HuggingFace repository.

    Supported formats for `name`:

    1. **`.pt` file** — path to a specific checkpoint file.
       A `config.json` must live in the same directory.

        ```python
        model = load_pretrained('my_run/weights_epoch_10.pt')
        ```

    2. **Folder** — path to a directory containing exactly one `.pt` file
       and a `config.json`.

        ```python
        model = load_pretrained('my_run/')
        ```

    3. **HuggingFace repo** (`<user>/<repo>`) — loaded from the local cache
       if already present, otherwise fetched from HF.

        ```python
        model = load_pretrained('nice-user/my-worldmodel')
        ```

    All local paths are resolved relative to `<cache_dir>/checkpoints/`.
    Some common known issues with the huggingface repository are automatically fixed.
    """

    cache_dir = utils.get_cache_dir(cache_dir, sub_folder='checkpoints')
    utils.ensure_dir_exists(cache_dir)
    checkpoint_path, config = utils._resolve(name, cache_dir)
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    state_dict = _fix_state_dict(state_dict)

    # assume keys with the dotted notation
    if extra_args is not None:
        for key, value in extra_args.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    model = instantiate(config)
    model.load_state_dict(state_dict)
    return model


def _fix_state_dict(state_dict):
    """
        fix the state dictionary by..
        unwrapping the state dict in case it is wrapped in an extra "model" or "state_dict" property
        removing dublicate ".encoder" in keys.
    """
    # Handle cases where the dictionary is nested under 'state_dict' or 'model'
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # 2. Fix the nested "encoder.encoder" prefix
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("encoder.encoder.", "encoder.", 1)
        new_key = new_key.replace("attention.attention.", "attention.",1)
        new_key = new_key.replace("layer.", "layers.",1)
        new_key = new_key.replace("query.", "q_proj.",1)
        new_key = new_key.replace("key.", "k_proj.",1)
        new_key = new_key.replace("value.", "v_proj.",1)
        new_key = new_key.replace("value.", "v_proj.",1)
        new_key = new_key.replace("attention.output.dense.", "attention.o_proj.",1)
        new_key = new_key.replace("intermediate.dense.", "mlp.fc1.",1)
        new_key = new_key.replace("output.dense.", "mlp.fc2.",1)


        """
            if new_key.startswith("encoder.encoder."):
                new_key = new_key.replace("encoder.encoder.", "encoder.", 1)
            if "attention.attention." in new_key:
                new_key = new_key.replace("attention.attention.","attention.",1)
        """
        fixed_state_dict[new_key] = value
    return fixed_state_dict


__all__ = ['load_fix_pretrained']