"""
Shows what skipping the image preparation does to the LeWM encoder and its predictor.
The same pictures are encoded twice: prepared like in the LeWM training and evaluation code
(utils.image_preprocessing.normalise_images) and as raw 0-255 values, which is what the generators passed before.

Run: python -m scripts.check_encoder_preprocessing   (about one minute)
"""
import numpy as np
import torch
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv

import utils.load_fix_pretrained as load_util
from utils.image_preprocessing import normalise_images

BLOCK = 5  # environment steps per model step
# mean and std of the actions in the LeWM training data (tworoom.h5), used to z-score actions like the LeWM evaluation
ACTION_MEAN, ACTION_STD = np.array([0.0031, -0.0530]), np.array([0.8675, 0.8678])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_util.load_fix_pretrained(snapshot_download(
        repo_id="quentinll/lewm-tworooms", revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec")).to(device).eval()
    env = TwoRoomEnv()
    env.reset()

    def render(x, y):
        env.reset(options={"state": [x, y], "target_state": [x, y]})
        return env.render()

    def encode(frames, prepared):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().unsqueeze(1).to(device)
        if prepared:
            x = normalise_images(x)
        with torch.no_grad():
            return model.encode({"pixels": x})["emb"][:, 0]

    print("1) Encoder output change when the agent moves d px to the right (same picture, two preparations)")
    frames = [render(60, 150)] + [render(60 + d, 150) for d in (1, 5, 20)]
    for name, prepared in (("prepared (as in LeWM training)", True), ("raw 0-255 values", False)):
        z = encode(frames, prepared)
        print(f"   {name:32s} d=1: {(z[0] - z[1]).norm():6.2f}   d=5: {(z[0] - z[2]).norm():6.2f}   d=20: {(z[0] - z[3]).norm():6.2f}")

    print("\n2) Does the model's own predictor work? ratio = |predicted next - real next| / |current - real next|")
    print("   (below 1: it predicts the movement; 1 or more: no better than guessing that nothing moves)")
    rng = np.random.default_rng(0)
    before, after, blocks = [], [], []
    for _ in range(150):
        x, y = int(rng.integers(30, 90)), int(rng.integers(40, 200))  # free space left of the wall
        action = rng.choice([-1.0, 1.0], 2).astype(np.float32)
        before.append(render(x, y))
        for _ in range(BLOCK):
            env.step(action)
        after.append(env.render())
        blocks.append(np.tile((action - ACTION_MEAN) / ACTION_STD, BLOCK))
    action_blocks = torch.tensor(np.stack(blocks), dtype=torch.float32, device=device)[:, None]  # (trials, 1, 2 * BLOCK)
    history = model.predictor.num_frames
    for name, prepared in (("prepared (as in LeWM training)", True), ("raw 0-255 values", False)):
        z0, z1 = encode(before, prepared), encode(after, prepared)
        with torch.no_grad():
            predicted = model.predict(z0[:, None][:, -history:], model.action_encoder(action_blocks)[:, -history:])[:, -1]
        ratio = (predicted - z1).norm(dim=1) / (z0 - z1).norm(dim=1)
        print(f"   {name:32s} median ratio {ratio.median().item():7.2f}   trials below 1: {100 * (ratio < 1).float().mean().item():3.0f}%")


if __name__ == "__main__":
    main()
