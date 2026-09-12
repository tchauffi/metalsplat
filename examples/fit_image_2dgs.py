"""Overfits a set of 2D gaussian splats ("surfels") to a target image using
Adam -- the 2DGS counterpart of `fit_image.py`.

Same "fit gaussians to an image" smoke test, but exercising the 2DGS
pipeline end to end: project_2dgs -> tile-bin (reused from 3DGS) ->
rasterize_2dgs's exact ray-splat intersection, with the distortion loss
added on top of plain image MSE. Proves gradients flow correctly through
the whole 2DGS pipeline (including into `means`/`raw_scales`/`raw_quats`
via the ray-splat `transform`, not just through the screen-space
fallback) and that training converges. Saves before/after PNGs next to
this script.

Usage: uv run python examples/fit_image_2dgs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from metalsplat import Camera, Gaussian2DModel, render_2dgs
from metalsplat.losses import distortion_loss

DEVICE = "mps"
IMG_SIZE = 128
NUM_GAUSSIANS = 2000
NUM_ITERS = 300
LR = 0.01
DISTORTION_WEIGHT = 0.01


def make_target_image(size: int) -> torch.Tensor:
    img = Image.new("RGB", (size, size), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [size * 0.15, size * 0.15, size * 0.65, size * 0.65], fill=(220, 80, 60)
    )
    draw.rectangle(
        [size * 0.45, size * 0.35, size * 0.9, size * 0.8], fill=(60, 160, 200)
    )
    draw.ellipse(
        [size * 0.3, size * 0.55, size * 0.75, size * 0.95], fill=(240, 210, 60)
    )
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr  # (H, W, 3)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

    target = make_target_image(IMG_SIZE).to(DEVICE)

    # Same depth=fx trick as fit_image.py: magnification fx/z == 1, so
    # world x/y and scale map 1:1 onto pixel offsets/sizes.
    fx = fy = float(IMG_SIZE)
    depth = fx

    torch.manual_seed(0)
    means = torch.zeros(NUM_GAUSSIANS, 3, device=DEVICE)
    means[:, 0] = (torch.rand(NUM_GAUSSIANS, device=DEVICE) - 0.5) * IMG_SIZE * 0.9
    means[:, 1] = (torch.rand(NUM_GAUSSIANS, device=DEVICE) - 0.5) * IMG_SIZE * 0.9
    means[:, 2] = depth

    scales = torch.full((NUM_GAUSSIANS, 2), 4.0, device=DEVICE)
    colors = torch.rand(NUM_GAUSSIANS, 3, device=DEVICE)
    model = Gaussian2DModel(means, scales=scales, colors=colors).to(DEVICE)

    camera = Camera.identity(
        fx=fx,
        fy=fy,
        cx=IMG_SIZE / 2,
        cy=IMG_SIZE / 2,
        img_width=IMG_SIZE,
        img_height=IMG_SIZE,
    ).to(DEVICE)

    out_dir = Path(__file__).parent
    save_image(target, out_dir / "fit_image_2dgs_target.png")
    with torch.no_grad():
        initial = render_2dgs(model, camera)
        torch.mps.synchronize()
    save_image(initial, out_dir / "fit_image_2dgs_before.png")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()
        aux = render_2dgs(model, camera, return_aux=True)
        loss = (aux.image - target).pow(2).mean() + DISTORTION_WEIGHT * distortion_loss(
            aux.distortion
        )
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == 1:
            torch.mps.synchronize()
            print(f"step {step:4d}  loss {loss.item():.5f}")

    with torch.no_grad():
        final = render_2dgs(model, camera)
        torch.mps.synchronize()
    save_image(final, out_dir / "fit_image_2dgs_after.png")
    print(f"Saved before/target/after images to {out_dir}")


if __name__ == "__main__":
    main()
