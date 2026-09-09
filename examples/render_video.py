"""Renders a smooth orbit video (and a depth pass) of a trained scene.

Loads a trained scene from a 3DGS .ply and flies an *analytic* circular
orbit around it: fit the plane the capture rig moved in (SVD on the camera
positions), take the median orbit radius and height, then sweep 360
degrees at constant angular velocity with a fixed look-at target.

Deliberately not interpolated from the capture poses. This is a dome
capture -- positions span a similar range vertically as horizontally -- so
ordering the real poses by azimuth puts cameras at very different heights
next to each other, and the resulting video jumps violently frame to
frame. An analytic circle is smooth by construction: no ordering, no
interpolation, no jitter. It stays near the training manifold by taking
its radius and height from the capture's own median.

Usage: uv run python examples/render_video.py
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metalsplat import Camera, render
from metalsplat.cleanup import damp_view_dependence, prune_isolated
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.export import load_ply

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
PLY_PATH = Path(__file__).parent / "garden.ply"
OUT_DIR = Path(__file__).parent / "video_frames"
VIDEO_PATH = Path(__file__).parent / "garden_orbit.mp4"
DEPTH_VIDEO_PATH = Path(__file__).parent / "garden_orbit_depth.mp4"
NUM_FRAMES = 720  # one full 360-degree revolution (24s at 30fps)
FPS = 30
RENDER_SCALE = 0.75  # render at a fraction of capture resolution, for speed
# Sub-frames averaged per output frame, i.e. shutter integration.
#
# The bulk of the orbit's flicker was a projection bug (a missing FOV clamp
# on the EWA Jacobian -- see reference/project_ref.py); fixing it cut the
# metric 40% on its own. What remains is genuine *temporal* aliasing: at 450
# frames/revolution the camera sweeps 0.8 deg = ~10 px per frame, enough to
# decorrelate fine texture (grass, paving, the table's radial planks)
# between consecutive frames. Averaging sub-frames is the physical cure --
# it is what a real shutter does.
#
# Measured after the projection fix, as mean |I[t+1] - 2I[t] + I[t-1]| over
# a 10-frame window -- the second difference, which cancels smooth motion
# and leaves flicker (0.188 before the fix, at 450 frames and no shutter):
#
#   450 frames, no shutter : 0.112      720 frames, no shutter : 0.092
#   450 frames, shutter 4  : 0.081      720 frames, shutter 4  : 0.070
#
# 720 + 4 lands 63% below where this started. Spatial anti-aliasing does
# *not* help here -- 3x supersampling and eps2d up to 2.0 each bought under
# 4% -- because the frames are decorrelated in time, not undersampled in
# space. Costs 4x the renders.
SHUTTER_SAMPLES = 4
# Post-training cleanup (metalsplat.cleanup): floaters and overfit
# view-dependence are far more visible in a moving camera than in a
# still, and both of these improve held-out PSNR too. 0 / None to skip.
PRUNE_CELL = 0.4
PRUNE_MIN_PER_CELL = 8
SH_DAMP = 0.75
BOB_AMPLITUDE = 0.06  # gentle vertical drift over the orbit, as a fraction of radius
# Sphere around the camera inside which gaussians are faded out, as a
# fraction of the orbit radius: (fully suppressed, fully visible). None to
# skip.
#
# Off because nothing on this scene is actually near the camera: the closest
# gaussian *centre* to any orbit camera is 1.78 world units away against an
# orbit radius of 3.68, and setting a fade band measured as an exact no-op.
# The splats that looked like they were floating in front of the camera were
# not near it at all -- they were far off-axis gaussians whose 2D covariance
# blew up and smeared across the frame, which is the projection bug fixed in
# kernels/project.metal. Kept because a capture with genuine near-camera
# floaters is a real and common failure and this is the knob for it.
NEAR_FADE = None  # e.g. (0.15, 0.30) to fade everything within 0.15*radius


def fit_orbit_frame(
    cameras: list[Camera],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Fits the plane the capture rig moved in.

    Returns (centre, up, right, radius, height): an orthonormal in-plane
    basis plus the median orbit radius and the median signed offset along
    `up`. The plane normal comes from an SVD of the centred camera
    positions (smallest singular direction = the direction they vary least
    in), sign-flipped to agree with where the cameras themselves think up
    is, so the orbit doesn't run upside down.
    """
    positions = torch.stack([c.position.cpu() for c in cameras])
    centre = positions.mean(dim=0)
    centred = positions - centre

    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    up = vh[-1]
    camera_up = torch.stack([-c.R_wc[1].cpu() for c in cameras]).mean(dim=0)
    if torch.dot(up, camera_up) < 0:
        up = -up
    up = up / up.norm()

    in_plane = centred - torch.outer(centred @ up, up)
    radius = in_plane.norm(dim=1).median().item()
    height = (centred @ up).median().item()

    right = in_plane[in_plane.norm(dim=1).argmax()]
    right = right - torch.dot(right, up) * up
    right = right / right.norm()
    return centre, up, right, radius, height


def build_camera_path(
    cameras: list[Camera], points: torch.Tensor, scale: float, subframes: int = 1
) -> tuple[list[Camera], float]:
    """A constant-radius, constant-speed circular orbit with a fixed look-at.

    Emits `NUM_FRAMES * subframes` cameras evenly around the circle, so
    consecutive groups of `subframes` span exactly one output frame's worth
    of motion and can be averaged into it. Returns the path and the fitted
    orbit radius (used to scale `NEAR_FADE`).
    """
    centre, up, right, radius, height = fit_orbit_frame(cameras)
    forward = torch.linalg.cross(up, right)  # completes the right-handed in-plane basis
    target = points.cpu().median(dim=0).values  # robust to SfM outliers

    template = cameras[0]
    width = int(template.img_width * scale)
    img_height = int(template.img_height * scale)
    fx, fy = template.fx * scale, template.fy * scale
    cx, cy = template.cx * scale, template.cy * scale

    path: list[Camera] = []
    n = NUM_FRAMES * subframes
    for i in range(n):
        theta = 2 * math.pi * i / n
        # one slow sine over the full revolution keeps the loop seamless
        bob = BOB_AMPLITUDE * radius * math.sin(theta)
        eye = (
            centre
            + up * (height + bob)
            + radius * (math.cos(theta) * right + math.sin(theta) * forward)
        )
        path.append(
            Camera.look_at(
                eye=eye,
                target=target,
                up=up,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_width=width,
                img_height=img_height,
            )
        )
    return path, radius


DEPTH_COVERAGE_THRESH = 0.1
"""Alpha below which a pixel is treated as empty in the depth pass.

Not 0.5: on this scene 69% of pixels reach alpha 0.5 but 99.4% reach 0.1
(median alpha is 0.58), so a half-coverage cut punches speckled black holes
through solid ground and foliage. The scene is genuinely semi-transparent
in places -- plenty of surfaces are built from stacked low-opacity
gaussians -- so the threshold has to sit near "any coverage at all" rather
than "mostly opaque" to visualise depth honestly.
"""


def depth_to_rgb(depth: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    """Normalised inverse-depth, turned into a viridis-ish false-colour map."""
    covered = alpha > DEPTH_COVERAGE_THRESH
    normalised = torch.where(
        covered, depth / alpha.clamp_min(1e-6), torch.zeros_like(depth)
    )
    if covered.any():
        near = torch.quantile(normalised[covered], 0.02)
        far = torch.quantile(normalised[covered], 0.98)
    else:
        near, far = torch.tensor(0.0), torch.tensor(1.0)
    t = ((normalised - near) / (far - near).clamp_min(1e-6)).clamp(0, 1)
    t = torch.where(covered, 1.0 - t, torch.zeros_like(t))  # near = bright

    # cheap perceptual ramp: dark blue -> teal -> yellow
    r = (t.clamp(0.5, 1.0) - 0.5) * 2
    g = t
    b = (1 - t).clamp(0, 1) * 0.8 + 0.2 * t
    rgb = torch.stack([r, g, b], dim=-1) * covered[..., None]
    return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def encode(pattern: str, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_path),
        ],
        check=True,
    )


def main() -> None:
    if not PLY_PATH.exists():
        raise FileNotFoundError(
            f"No trained scene at {PLY_PATH}; run train_garden.py first."
        )

    print(f"Loading {PLY_PATH.name}...", flush=True)
    model = load_ply(PLY_PATH, device=DEVICE)
    print(f"{model.num_points} gaussians, sh_degree={model.sh_degree}", flush=True)

    if PRUNE_MIN_PER_CELL:
        model, n_pruned = prune_isolated(model, PRUNE_CELL, PRUNE_MIN_PER_CELL)
        print(f"pruned {n_pruned} isolated floaters -> {model.num_points}", flush=True)
    if SH_DAMP and SH_DAMP != 1.0:
        model = damp_view_dependence(model, SH_DAMP)
        print(f"damped non-DC SH by {SH_DAMP}", flush=True)

    scene = load_colmap_scene(DATA_ROOT, device=DEVICE)
    shutter = max(1, SHUTTER_SAMPLES)
    path, radius = build_camera_path(scene.cameras, scene.points, RENDER_SCALE, shutter)
    print(
        f"{NUM_FRAMES} frame analytic orbit ({shutter} sub-frames each) "
        f"fitted to {len(scene.cameras)} capture poses",
        flush=True,
    )
    near_fade = (
        None if NEAR_FADE is None else (NEAR_FADE[0] * radius, NEAR_FADE[1] * radius)
    )
    if near_fade:
        print(
            f"fading gaussians within {near_fade[0]:.2f}..{near_fade[1]:.2f} of the camera",
            flush=True,
        )

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    background = torch.zeros(3, device=DEVICE)
    for i in range(NUM_FRAMES):
        # Average the sub-frames spanning this output frame: shutter
        # integration, the temporal low-pass that removes strobing.
        accum = None
        mid = None
        with torch.no_grad():
            for s in range(shutter):
                aux = render(
                    model,
                    path[i * shutter + s].to(DEVICE),
                    background=background,
                    return_aux=True,
                    near_fade=near_fade,
                )
                accum = aux.image if accum is None else accum + aux.image
                if s == shutter // 2:
                    mid = aux  # depth comes from one instant, not a blur of several
            torch.mps.synchronize()
        rgb = ((accum / shutter).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(OUT_DIR / f"rgb_{i:04d}.png")
        Image.fromarray(depth_to_rgb(mid.depth, 1.0 - mid.final_T)).save(
            OUT_DIR / f"depth_{i:04d}.png"
        )
        if i % 50 == 0:
            print(f"  frame {i}/{NUM_FRAMES}", flush=True)

    print("Encoding...", flush=True)
    encode(str(OUT_DIR / "rgb_%04d.png"), VIDEO_PATH)
    encode(str(OUT_DIR / "depth_%04d.png"), DEPTH_VIDEO_PATH)
    duration = NUM_FRAMES / FPS
    print(
        f"Wrote {VIDEO_PATH} and {DEPTH_VIDEO_PATH} ({duration:.1f}s @ {FPS}fps)",
        flush=True,
    )


if __name__ == "__main__":
    main()
