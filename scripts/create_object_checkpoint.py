"""
script for creating an object snapshop checkpoint from a weights checkpoint (.pt)
"""
import sys
import json, torch, stable_pretraining as spt
from pathlib import Path
from lewm.jepa import JEPA
from lewm.module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm



def main():

    if len(sys.argv) != 5:
        print("Please supply exactly four arguments in this order: \n - Source Dir: the directory the source weights are located \n - Source Name: the name of the weights file (without .pt) \n - Out Dir: the name of the outpuut directory \n - Out Name: the name of the output file (without _object.ckpt)")
        return

    SOURCE_DIR = sys.argv[1]
    SOURCE_CHECKPOINT_NAME = sys.argv[2]
    OUT_DIR = sys.argv[3]
    OUT_MODEL_NAME = sys.argv[4]

    src = Path(swm.data.utils.get_cache_dir(), "checkpoints/"+ SOURCE_DIR)
    out = Path(swm.data.utils.get_cache_dir(), "checkpoints/" + OUT_DIR, OUT_MODEL_NAME + "_object.ckpt")

    cfg = json.loads((src / "config.json").read_text())
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False, use_mask_token=False,
    )
    mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                        hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**cfg["predictor"]),
        action_encoder=Embedder(**cfg["action_encoder"]),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    sd = torch.load(src / (SOURCE_CHECKPOINT_NAME + ".pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)



if __name__ == "__main__":
    main()
