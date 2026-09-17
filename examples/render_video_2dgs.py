"""Renders a side-by-side RGB + surface-normal orbit video of a trained
2DGS scene, for the README.

The 2DGS counterpart to `render_video.py`. Same analytic circular orbit
(fitted to the capture rig's own plane -- see `fit_orbit_frame`, imported
from that module rather than duplicated), but the second pane is the
rendered *normal* map rather than depth, because normals are the thing
2DGS actually buys over 3DGS: a 3D ellipsoid has no well-defined surface
orientation, a flat disk does.

Both panes come out of a single `render_2dgs` call per frame -- the
rasterizer produces image, depth and normal in the same pass, so the
normal pane costs nothing extra beyond the compositing.

Usage: uv run python examples/render_video_2dgs.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from render_video import fit_orbit_frame

from metalsplat import Camera, render_2dgs
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.export2dgs import load_ply

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
PLY_PATH = Path(__file__).parent / "garden_2dgs_best.ply"
OUT_DIR = Path(__file__).parent / "video_frames_2dgs"
DOCS_DIR = Path(__file__).parent.parent / "docs"
VIDEO_PATH = DOCS_DIR / "garden_2dgs_orbit.mp4"
WEBP_PATH = DOCS_DIR / "garden_2dgs_orbit.webp"
STILL_PATH = DOCS_DIR / "garden_2dgs_orbit.png"

NUM_FRAMES = 720  # one full revolution (24s at 30fps)
FPS = 30
RENDER_SCALE = 0.75  # fraction of capture resolution, per pane
# Sub-frames averaged per output frame (shutter integration). render_video.py
# documents why this matters for the 3DGS orbit -- the same temporal aliasing
# applies here, and 720 frames x 4 sub-frames is the setting its measured
# flicker table lands on. Costs 4x the renders.
SHUTTER_SAMPLES = 4
BOB_AMPLITUDE = 0.06  # gentle vertical drift, as a fraction of orbit radius
STILL_FRAME = 0  # which frame to also save as the static README image

# The frames are rendered large for the mp4 and downscaled for the two
# README assets, at the same dimensions the 3DGS orbit already ships at.
#
# Both are noticeably harder to compress than the 3DGS pair, because the
# normal pane is high-entropy almost everywhere the scene isn't flat: grass
# and foliage genuinely have a different surface orientation every few
# pixels, so there is little for an encoder to predict. That is real signal,
# not noise to be filtered away, so the size is bought back with frame rate
# and quantization rather than by smoothing the image. At these settings the
# WebP lands ~5.0MB against the 3DGS one's 4.7MB, and the mp4 ~7.9MB against
# 5.4MB.
WEBP_WIDTH = 760
WEBP_FPS = 10
WEBP_QUALITY = 60
MP4_CRF = 28
STILL_WIDTH = 924

# Alpha below which a pixel counts as empty. Same reasoning as
# render_video.py's DEPTH_COVERAGE_THRESH: this scene is genuinely
# semi-transparent in places, so the cut has to sit near "any coverage at
# all" rather than "mostly opaque" or it punches holes through real surfaces.
NORMAL_COVERAGE_THRESH = 0.1

# No post-training cleanup pass here: metalsplat.cleanup's prune_isolated /
# damp_view_dependence are written against GaussianModel (3 scales) and have
# no Gaussian2DModel equivalent yet. The orbit is rendered from the raw
# trained model.


def build_camera_path(
    cameras: list[Camera], points: torch.Tensor, scale: float, subframes: int
) -> list[Camera]:
    """`NUM_FRAMES * subframes` cameras evenly spaced around the fitted orbit.

    Consecutive groups of `subframes` span exactly one output frame's worth
    of motion, so they can be averaged into it.
    """
    centre, up, right, radius, height = fit_orbit_frame(cameras)
    forward = torch.linalg.cross(up, right)
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
        bob = BOB_AMPLITUDE * radius * math.sin(theta)  # one sine, so it loops
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
    return path


def normal_to_rgb(normal: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    """World-space normal -> the standard [-1,1] -> [0,1] false-colour map.

    `normal` off the rasterizer is an alpha-weighted *sum* (its magnitude is
    ~alpha), not a unit vector, so it is divided by alpha and renormalized
    before mapping -- otherwise semi-transparent surfaces render as washed
    out toward grey rather than showing their actual orientation. Same
    correction the normal-consistency loss makes for the same reason.

    World space, not camera space: a world-space normal is a property of the
    surface alone, so a correctly reconstructed wall keeps one steady colour
    for the whole orbit and any shimmer is real geometric inconsistency
    rather than an artefact of the camera moving. Camera-space normals would
    look smoother here while hiding exactly the defect worth showing.
    """
    covered = alpha > NORMAL_COVERAGE_THRESH
    unit = torch.nn.functional.normalize(
        normal / alpha.clamp_min(1e-6).unsqueeze(-1), dim=-1
    )
    rgb = (unit * 0.5 + 0.5) * covered[..., None]
    return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _ffmpeg(pattern: str, codec_args: list[str], out_path: Path) -> None:
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
            *codec_args,
            str(out_path),
        ],
        check=True,
    )


def encode_mp4(pattern: str, out_path: Path) -> None:
    args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(MP4_CRF)]
    _ffmpeg(pattern, args, out_path)


def encode_webp(frames: list[Path], out_path: Path) -> None:
    """Animated WebP for inline README display.

    GitHub strips <video> tags from rendered markdown, so the README embeds
    the orbit as an image; WebP is the only widely-supported animated format
    that survives that and still looks acceptable at this size.

    Written with Pillow rather than ffmpeg: ffmpeg only encodes WebP when
    it was built against libwebp, which the current Homebrew build is not,
    and failing at the last step of a multi-minute render for a missing
    optional codec is a bad trade. Pillow is already a dependency here.
    """
    step = max(1, round(FPS / WEBP_FPS))
    kept = frames[::step]
    images = []
    for path in kept:
        im = Image.open(path).convert("RGB")
        height = round(im.height * WEBP_WIDTH / im.width)
        images.append(im.resize((WEBP_WIDTH, height), Image.LANCZOS))
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / WEBP_FPS),
        loop=0,
        quality=WEBP_QUALITY,
        method=4,
    )


def render_frames() -> None:
    if not PLY_PATH.exists():
        raise FileNotFoundError(
            f"No trained 2DGS scene at {PLY_PATH}; run train_garden_2dgs.py first."
        )

    print(f"Loading {PLY_PATH.name}...", flush=True)
    model = load_ply(PLY_PATH, device=DEVICE)
    print(f"{model.num_points} surfels, sh_degree={model.sh_degree}", flush=True)

    scene = load_colmap_scene(DATA_ROOT, device=DEVICE)
    shutter = max(1, SHUTTER_SAMPLES)
    path = build_camera_path(scene.cameras, scene.points, RENDER_SCALE, shutter)
    pane = path[0]
    print(
        f"{NUM_FRAMES} frame orbit ({shutter} sub-frames each) fitted to "
        f"{len(scene.cameras)} capture poses, {pane.img_width}x{pane.img_height} per pane",
        flush=True,
    )

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    background = torch.zeros(3, device=DEVICE)
    for i in range(NUM_FRAMES):
        accum = None
        mid = None
        with torch.no_grad():
            for s in range(shutter):
                aux = render_2dgs(
                    model,
                    path[i * shutter + s].to(DEVICE),
                    background=background,
                    return_aux=True,
                )
                accum = aux.image if accum is None else accum + aux.image
                if s == shutter // 2:
                    mid = aux  # normals come from one instant, not a blur
            torch.mps.synchronize()

        rgb = ((accum / shutter).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        normal = normal_to_rgb(mid.normal, 1.0 - mid.final_T)
        Image.fromarray(np.hstack([rgb, normal])).save(OUT_DIR / f"frame_{i:04d}.png")
        if i % 30 == 0:
            print(f"  frame {i}/{NUM_FRAMES}", flush=True)


def encode_all() -> None:
    frames = sorted(OUT_DIR.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(
            f"No frames in {OUT_DIR}; run without --encode-only to render them."
        )

    still = Image.open(frames[STILL_FRAME])
    still_height = round(still.height * STILL_WIDTH / still.width)
    still.resize((STILL_WIDTH, still_height), Image.LANCZOS).save(STILL_PATH)

    print(f"Encoding {len(frames)} frames...", flush=True)
    encode_mp4(str(OUT_DIR / "frame_%04d.png"), VIDEO_PATH)
    encode_webp(frames, WEBP_PATH)
    print(
        f"Wrote {VIDEO_PATH.name}, {WEBP_PATH.name} and {STILL_PATH.name} "
        f"to {DOCS_DIR} ({len(frames) / FPS:.1f}s @ {FPS}fps)",
        flush=True,
    )


def main() -> None:
    # --encode-only re-runs just the (seconds-long) encode step against
    # frames already on disk, rather than the multi-minute render, which is
    # what you want when tuning output size/quality.
    if "--encode-only" not in sys.argv:
        render_frames()
    DOCS_DIR.mkdir(exist_ok=True)
    encode_all()


if __name__ == "__main__":
    main()
