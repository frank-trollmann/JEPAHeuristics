# The LeWM encoder expects scaled and normalised images

**Short answer.** The LeWM encoder was trained on images scaled to [0, 1] and then normalised with the ImageNet mean and std per colour channel. The environment renders values 0-255, and `LeWM.encode` does not rescale them. Feeding raw 0-255 values is off the distribution the model was trained on, and it changes the output a lot (numbers below).

## Where this is defined (authors' code, `lucas-maes/le-wm`, commit `8edfeb3`)

| What | Link |
|---|---|
| Training builds its image preparation with `get_img_preprocessor` | [`train.py` L59](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py#L59) |
| `get_img_preprocessor` = `ToImage(**ImageNet stats)` then resize | [`utils.py` L6-L11](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py#L6-L11) |
| Evaluation: `ToDtype(float32, scale=True)` then `Normalize(**ImageNet)` | [`eval.py` L17-L24](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/eval.py#L17-L24) |

- `ToImage(scale=True, mean=..., std=...)` in `stable_pretraining` (0.1.8, `data/transforms.py`) casts to float, scales uint8 values to [0, 1] (`ToDtype(scale=True)`), and then applies `Normalize(mean, std)`.
- `ImageNet = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (`stable_pretraining/data/dataset_stats.py`).
- `LeWM.encode` (`stable_worldmodel` 0.1.1, `wm/lewm/lewm.py`) only casts the dtype and passes the pixels to the ViT.

`utils/image_preprocessing.normalise_images` does the same computation, and `utils/tests/test_image_preprocessing.py` checks it against the transform of `eval.py` on a random frame.

## Does it matter? Measured with the real model

Run `python -m scripts.check_encoder_preprocessing` (about one minute). Same pictures, two preparations:

| | Prepared (as in training) | Raw 0-255 |
|---|---|---|
| Encoder output change when the agent moves 1 / 5 / 20 px | 1.54 / 8.26 / 22.16 | 0.01 / 0.05 / 0.06 |
| Predictor error relative to "nothing moves" (median of 150 random moves) | 0.21 (better than "nothing moves" in 100% of trials) | 247 (0%) |

Further measurements from my setup (all valid positions, split by position, not part of this script): the position read from one encoding has a mean error of 0.9 px with prepared input and 14.3 px with raw input; a learned wall-aware distance has a mean error of 0.64 px and 15.7 px.

## What this does not show

- It does not prove that the exact ImageNet statistics are the only good choice; it shows that the training and evaluation code uses them and that raw 0-255 input breaks the model.
- It does not measure the effect on the models trained on the existing datasets. A position probe on a regenerated set would.
- Existing datasets store encodings, not images, so they cannot be corrected afterwards; they must be regenerated.
