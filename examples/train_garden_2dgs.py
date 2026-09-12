"""Trains a Gaussian2DModel to reconstruct the real "garden" COLMAP scene in
data/garden, using the 2D Gaussian Splatting paper's (Huang et al. 2024)
own hyperparameters: per-group learning rates (feature/opacity/scaling/
rotation held constant, only position decayed), lambda_dssim=0.2,
lambda_normal=0.05 (active after iteration 7000), lambda_dist=0.0 (active
after iteration 3000; 0 is the paper's own default for general/unbounded
scenes -- raise it for bounded, object-centric scenes), SH degree 3 grown
by one band every 1000 steps, and 30,000 total iterations -- all read
directly off the official reference implementation's `arguments/__init__.py`
`OptimizationParams` and `train.py`'s loss/schedule
(https://github.com/hbb1/2d-gaussian-splatting).

One deliberate deviation: the paper's `position_lr_init`/`position_lr_final`
(0.00016/0.0000016) are calibrated for the reference implementation's own
scene-normalization step (translating/rescaling by the training cameras'
extent), which this codebase's COLMAP loader doesn't apply. Using those
absolute values directly on raw COLMAP-scale coordinates would train far
too slowly (or not at all) on a real-world scene, so -- like
`train_garden.py`'s 3DGS pipeline already does, for the same documented
reason -- the *absolute* position learning rate is calibrated to this
scene's own point spacing instead. The *shape* of the schedule (log-linear
decay to 1% of the initial rate) is identical to the paper's, and every
other hyperparameter is used exactly as published.

Adaptive density control (split/clone/prune, see `metalsplat.densify2dgs`)
runs on the paper's own schedule (`densify_from_iter=500` to
`densify_until_iter=15000`, every `densification_interval=100` steps,
`opacity_reset_interval=3000`) with one more deliberate deviation: the
paper's literal `densify_grad_threshold=0.0002` is calibrated for a
*plain* screen-space-gradient-norm signal, whereas this codebase's
densification signal is AbsGS-style (sum of |gradient|, not the signed
sum -- see `metalsplat.optim.SparseAdam`'s docstring and
`train_garden.py`'s identical reasoning for why that's preferred here).
The two signals live on different numeric scales, so reusing the paper's
literal threshold could mean never densifying, or densifying everything,
depending on luck. This reuses `train_garden.py`'s self-calibrating
approach instead: the `densify_grad_percentile` quantile of the *first*
densification round is frozen as an absolute bar for the rest of
training. There is still no loss-driven seeding equivalent to
`metalsplat.seed` for 2DGS, but the paper's own schedule doesn't have an
equivalent of that either, so nothing is missing relative to the paper
specifically.

This repo's 2DGS path still has no filter3d equivalent or
`seed_uncovered_regions` equivalent -- out of scope here, same as before.
(`.ply` export is now implemented, see `metalsplat.export2dgs` -- same
format as the official reference implementation's own `.ply` files,
just 2 `scale_*` properties instead of 3.)

One more deviation from `train_garden.py`: its `calibrate_initial_scale`
targets a 3px initial screen radius, which for this scene lands ~7.5x
below `scene_scale` (the point-cloud spacing `densify_and_prune_2dgs`'s
split-vs-clone decision is calibrated against). Since split only fires
for gaussians *already larger* than `scene_scale`, that gap meant almost
no gaussian qualified for hundreds of steps -- densification was nearly
pure clone (duplicate exactly in place, then slowly drift apart via
gradient descent) rather than split (an immediate, differently-positioned
offset). Confirmed via an instrumented run: 0 splits for the first two
densify rounds, and only a handful for several more. Clone-then-drift is
particularly slow wherever many gaussians overlap and share credit/blame
for the same pixels (the drift signal is diluted across all of them), so
this disproportionately kept complex, overlapping regions stuck as a
blurry pile of near-duplicate disks long after simple/background regions
(where a lone clone's drift signal is concentrated, not shared) had
separated and sharpened. Raising the target to 10px narrows that gap
(init scale ~45% of scene_scale here, instead of ~14%) without
reintroducing the original giant-overlapping-blob failure mode
`train_garden.py`'s comment describes (measured ~23px unrecalibrated on
this scene) -- verified empirically: splits appear within the first few
rounds instead of being starved for ~500 steps, and held-out PSNR at a
fixed step count improves. This does not fully close the gap between
complex/overlapping regions and simple ones -- some of that is 2DGS's
own planarity prior working against non-planar, repeated detail (foliage,
moss) rather than a densification bug -- but it removes the
easily-avoidable part of it.

One more gap, found by timing a real 30k-step run (wall-clock time per
1000 steps kept climbing well past the point densification stops):
`OPACITY_RESET_INTERVAL` keeps firing every 3000 steps all the way to
`NUM_ITERS`, but pruning here only ever ran *inside*
`densify_and_prune_2dgs`, which itself only runs through `DENSIFY_STOP`
(15000). Every reset after that point caps a fresh batch of gaussians'
opacity near zero with nothing left to clean them up afterward -- they
sit there forever, still costing full projection/rasterization compute
every step while contributing ~nothing to the image, compounding with
each subsequent reset (18000, 21000, 24000, 27000). `train_garden.py`
already solves this for 3DGS with a standalone prune schedule that keeps
running past its own densify-stop point; this adds the identical thing
here via `prune_low_opacity_2dgs`.

Usage: uv run python examples/train_garden_2dgs.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from metalsplat import Gaussian2DModel, render_2dgs, save_ply_2dgs
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.densify import reset_opacity
from metalsplat.densify2dgs import densify_and_prune_2dgs, prune_low_opacity_2dgs
from metalsplat.losses import (
    distortion_loss,
    gaussian_splatting_loss,
    normal_consistency_loss,
)
from metalsplat.optim import SparseAdam, migrate_optimizer_state

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
OUT_DIR = Path(__file__).parent

# --- 2DGS paper defaults (arguments/__init__.py's OptimizationParams) ---
NUM_ITERS = 11_000
FEATURE_LR = 0.0025  # paper splits this into f_dc (this rate) / f_rest (this / 20);
# this repo's SH parameter isn't split that way, so one rate covers all bands.
OPACITY_LR = 0.05
SCALING_LR = 0.005
ROTATION_LR = 0.001
LAMBDA_DSSIM = 0.2
LAMBDA_NORMAL = 0.05
LAMBDA_NORMAL_START_ITER = 7000  # paper: `lambda_normal if iteration > 7000 else 0`
LAMBDA_DIST = 0.0  # paper default for general/unbounded scenes; raise for bounded ones
LAMBDA_DIST_START_ITER = 3000  # paper: `lambda_dist if iteration > 3000 else 0`
SH_DEGREE = 3
SH_DEGREE_INTERVAL = 1000  # paper: +1 band every 1000 steps (`oneupSHdegree`)
INIT_OPACITY = 0.1  # paper default (`inverse_sigmoid(0.1 * ones(...))`)
DENSIFY_START = 500  # paper: densify_from_iter
DENSIFY_STOP = 15_000  # paper: densify_until_iter
DENSIFY_INTERVAL = 100  # paper: densification_interval
PRUNE_OPACITY_THRESH = 0.05  # paper: opacity_cull
OPACITY_RESET_INTERVAL = (
    3000  # paper default (reset_opacity()'s own 0.01 cap matches too)
)
# --- end paper defaults ---

# Calibrated on the first densification round, then frozen -- see module
# docstring for why this replaces the paper's literal densify_grad_threshold.
DENSIFY_GRAD_PERCENTILE = 0.9

# Standalone prune schedule, matching train_garden.py's identical one: unlike
# densify_and_prune_2dgs's own inline prune (which only runs during the
# DENSIFY_START..DENSIFY_STOP window), this keeps running for the rest of
# training -- see module docstring for why that gap matters once
# OPACITY_RESET_INTERVAL keeps firing past DENSIFY_STOP.
#
# Deliberately its own, *lower* threshold rather than reusing
# PRUNE_OPACITY_THRESH (0.05): OPACITY_RESET_INTERVAL and PRUNE_INTERVAL can
# land on the same step, and reset_opacity() caps every gaussian at 0.01 --
# a threshold above that would prune the entire model the instant a reset
# and a prune coincide (confirmed by a smoke test: n=0 immediately after
# such a step). train_garden.py's own standalone prune uses the same 0.005
# for the identical reason, safely below its own 0.01 reset cap.
STANDALONE_PRUNE_OPACITY_THRESH = 0.005
PRUNE_START = 100
PRUNE_STOP = NUM_ITERS
PRUNE_INTERVAL = 100

EVAL_EVERY = 1000
EVAL_HOLDOUT_STRIDE = 8
EVAL_IMAGES_SAVED = 3


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().item()
    if mse <= 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = (tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def estimate_scene_scale(points: torch.Tensor, sample_size: int = 5000) -> float:
    """Median nearest-neighbor distance between sparse points -- see
    train_garden.py's identical helper for the full rationale (this
    script's module docstring explains why it stands in for the paper's
    scene-normalization-calibrated position learning rate).
    """
    n = points.shape[0]
    idx = torch.randperm(n, device=points.device)[: min(sample_size, n)]
    sample = points[idx]
    d = torch.cdist(sample, sample)
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.median().item()


def calibrate_initial_scale(
    points: torch.Tensor,
    cameras: list,
    scene_scale: float,
    target_pixel_radius: float = 3.0,
    sample_points: int = 3000,
    sample_cameras: int = 20,
) -> float:
    """Scales `scene_scale` so a gaussian of that size projects to roughly
    `target_pixel_radius` pixels on screen -- see train_garden.py's
    identical helper for the full rationale.
    """
    n = points.shape[0]
    idx = torch.randperm(n, device=points.device)[: min(sample_points, n)]
    sample = points[idx]

    cam_idx = torch.randperm(len(cameras))[: min(sample_cameras, len(cameras))]
    magnifications = []
    for ci in cam_idx.tolist():
        cam = cameras[ci]
        pts_cam = sample.to(cam.R_wc.device) @ cam.R_wc.T + cam.t_wc
        z = pts_cam[:, 2]
        in_front = z > 0.1
        if in_front.any():
            magnifications.append((cam.fx / z[in_front]).median().item())

    median_magnification = torch.tensor(magnifications).median().item()
    multiplier = target_pixel_radius / (scene_scale * median_magnification)
    return scene_scale * multiplier


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Expected COLMAP scene at {DATA_ROOT}")

    print("Loading COLMAP scene...", flush=True)
    scene = load_colmap_scene(DATA_ROOT, device=DEVICE)
    n_images = len(scene.cameras)
    eval_idx = list(range(0, n_images, EVAL_HOLDOUT_STRIDE))
    train_idx = [i for i in range(n_images) if i not in set(eval_idx)]
    print(
        f"{n_images} images: {len(train_idx)} train, {len(eval_idx)} eval", flush=True
    )
    print(f"{scene.points.shape[0]} initial sparse points")

    scene_scale = estimate_scene_scale(scene.points)
    # target_pixel_radius=10 (not train_garden.py's 3): see module docstring
    # for why 3px starves densify_and_prune_2dgs's split path on this scene.
    init_scale = calibrate_initial_scale(
        scene.points, scene.cameras, scene_scale, target_pixel_radius=10.0
    )
    print(
        f"Scene scale (median NN spacing): {scene_scale:.4f}, calibrated initial gaussian scale: {init_scale:.5f}",
        flush=True,
    )

    # Paper's exact ratio: position_lr_final / position_lr_init = 0.01.
    lr_means_init = 0.02 * scene_scale
    lr_means_final = lr_means_init * 0.01
    print(
        f"means lr: {lr_means_init:.5f} -> {lr_means_final:.5f} (log-linear decay, "
        f"paper's 1% ratio; see module docstring for why the absolute value is "
        f"scene-calibrated rather than the paper's literal 0.00016)",
        flush=True,
    )

    n_points = scene.points.shape[0]
    scales = torch.full((n_points, 2), init_scale, device=DEVICE)
    opacities = torch.full((n_points,), INIT_OPACITY, device=DEVICE)
    model = Gaussian2DModel(
        scene.points,
        scales=scales,
        colors=scene.colors,
        opacities=opacities,
        sh_degree=SH_DEGREE,
    ).to(DEVICE)
    model.active_sh_degree = 0
    print(f"SH degree starts at 0, +1 every {SH_DEGREE_INTERVAL} steps", flush=True)

    # Five param groups, matching the paper's training_setup exactly: only
    # the first (means/"xyz") gets its learning rate scheduled per step,
    # the rest stay constant for the whole run.
    def make_optimizer(m: Gaussian2DModel, lr_means: float) -> torch.optim.Optimizer:
        color_param = m.raw_colors if m.sh_degree == 0 else m.raw_sh
        return SparseAdam(
            [
                {"params": [m.means], "lr": lr_means},
                {"params": [color_param], "lr": FEATURE_LR},
                {"params": [m.raw_scales], "lr": SCALING_LR},
                {"params": [m.raw_quats], "lr": ROTATION_LR},
                {"params": [m.raw_opacities], "lr": OPACITY_LR},
            ]
        )

    optimizer = make_optimizer(model, lr_means_init)
    background = torch.zeros(3, device=DEVICE)

    def eval_and_save(step: int) -> None:
        with torch.no_grad():
            psnrs = []
            for k, idx in enumerate(eval_idx):
                aux = render_2dgs(
                    model, scene.cameras[idx], background=background, return_aux=True
                )
                torch.mps.synchronize()
                psnrs.append(psnr(aux.image, scene.images[idx]))
                if k < EVAL_IMAGES_SAVED:
                    save_image(
                        aux.image, OUT_DIR / f"garden_2dgs_eval_{k}_step{step}.png"
                    )
                    # World-space normal in [-1, 1] -> [0, 1] for display,
                    # the standard normal-map visualization convention.
                    save_image(
                        aux.normal * 0.5 + 0.5,
                        OUT_DIR / f"garden_2dgs_normal_{k}_step{step}.png",
                    )
            mean_psnr = sum(psnrs) / len(psnrs)
            print(
                f"  eval PSNR ({len(psnrs)} held-out views): {mean_psnr:.2f} dB",
                flush=True,
            )

        # Held-out PSNR doesn't necessarily peak on the last step (see
        # train_garden.py's identical reasoning), so keep the best
        # checkpoint rather than trusting the final one. An eval that lands
        # exactly on an opacity-reset step is a known, expected dip (see
        # module docstring / commit history) -- it won't overwrite a
        # genuinely better earlier checkpoint since it's never the max.
        nonlocal best
        if mean_psnr > best[0]:
            best = (mean_psnr, step)
            save_ply_2dgs(model, OUT_DIR / "garden_2dgs_best.ply")

    best = (float("-inf"), 0)  # (psnr, step) of the best checkpoint so far

    print("Saving target/initial renders for eval view 0...", flush=True)
    save_image(scene.images[eval_idx[0]], OUT_DIR / "garden_2dgs_target_0.png")
    eval_and_save(0)

    grad_accum = torch.zeros(model.num_points, device=DEVICE)
    grad_count = torch.zeros(model.num_points, device=DEVICE)
    # Calibrated on the first densification round, then held fixed -- see
    # module docstring.
    densify_threshold = None

    previous_sh_degree = model.active_sh_degree
    start = time.time()
    for step in range(1, NUM_ITERS + 1):
        t = step / NUM_ITERS
        lr_means = lr_means_init * (lr_means_final / lr_means_init) ** t
        optimizer.param_groups[0]["lr"] = lr_means

        idx = train_idx[int(torch.randint(len(train_idx), (1,)).item())]
        cam = scene.cameras[idx]
        target = scene.images[idx]

        optimizer.zero_grad()
        aux = render_2dgs(
            model,
            cam,
            background=background,
            return_aux=True,
            abs_grad_accum=grad_accum,
        )
        photo_loss = gaussian_splatting_loss(
            aux.image, target, lambda_dssim=LAMBDA_DSSIM
        )

        # Paper's exact iteration-gated regularizer weights.
        lambda_normal = LAMBDA_NORMAL if step > LAMBDA_NORMAL_START_ITER else 0.0
        lambda_dist = LAMBDA_DIST if step > LAMBDA_DIST_START_ITER else 0.0
        normal_loss = lambda_normal * normal_consistency_loss(
            aux.normal, aux.depth, cam
        )
        dist_loss = lambda_dist * distortion_loss(aux.distortion)

        loss = photo_loss + normal_loss + dist_loss
        loss.backward()  # accumulates into grad_accum in place (AbsGS-style)
        visible = aux.valid > 0.5
        optimizer.step(visible)

        with torch.no_grad():
            grad_count[visible] += 1.0

        if step % 25 == 0 or step == 1:
            torch.mps.synchronize()
            elapsed = time.time() - start
            print(
                f"step {step:5d}  loss {loss.item():.5f}  n {model.num_points}  "
                f"(photo {photo_loss.item():.5f}  normal {normal_loss.item():.5f}  "
                f"dist {dist_loss.item():.5f})  "
                f"({elapsed:.1f}s elapsed, {elapsed / step:.2f}s/step)",
                flush=True,
            )

        if DENSIFY_START <= step <= DENSIFY_STOP and step % DENSIFY_INTERVAL == 0:
            model, stats = densify_and_prune_2dgs(
                model,
                grad_accum,
                grad_count,
                scene_scale=scene_scale,
                grad_percentile=DENSIFY_GRAD_PERCENTILE,
                prune_opacity_thresh=PRUNE_OPACITY_THRESH,
                grad_threshold=densify_threshold,
            )
            if densify_threshold is None:
                densify_threshold = stats.grad_threshold
                print(
                    f"  densify threshold calibrated to {densify_threshold:.3e} "
                    f"(p{100 * DENSIFY_GRAD_PERCENTILE:.0f} of round 1); fixed from here",
                    flush=True,
                )
            optimizer = migrate_optimizer_state(
                optimizer, make_optimizer(model, lr_means), stats.source_index
            )
            grad_accum = torch.zeros(model.num_points, device=DEVICE)
            grad_count = torch.zeros(model.num_points, device=DEVICE)
            print(
                f"  densify @ step {step}: {stats.n_before} -> {stats.n_after} "
                f"(+{stats.n_split} split, +{stats.n_cloned} cloned, -{stats.n_pruned} pruned)",
                flush=True,
            )

        if OPACITY_RESET_INTERVAL and step % OPACITY_RESET_INTERVAL == 0:
            reset_opacity(model)
            print(f"  opacity reset @ step {step}", flush=True)

        # Standalone prune: keeps running after DENSIFY_STOP, unlike
        # densify_and_prune_2dgs's own inline prune -- see module docstring.
        if PRUNE_START <= step <= PRUNE_STOP and step % PRUNE_INTERVAL == 0:
            model, n_pruned, prune_index = prune_low_opacity_2dgs(
                model, prune_opacity_thresh=STANDALONE_PRUNE_OPACITY_THRESH
            )
            if n_pruned > 0:
                optimizer = migrate_optimizer_state(
                    optimizer, make_optimizer(model, lr_means), prune_index
                )
                grad_accum = torch.zeros(model.num_points, device=DEVICE)
                grad_count = torch.zeros(model.num_points, device=DEVICE)
                print(
                    f"  prune @ step {step}: -{n_pruned} (n={model.num_points})",
                    flush=True,
                )

        if SH_DEGREE_INTERVAL and step % SH_DEGREE_INTERVAL == 0:
            active = model.increase_sh_degree()
            if active != previous_sh_degree:
                print(f"  SH degree -> {active} @ step {step}", flush=True)
                previous_sh_degree = active

        if step % EVAL_EVERY == 0:
            eval_and_save(step)

    eval_and_save(NUM_ITERS)
    ply_path = OUT_DIR / "garden_2dgs.ply"
    save_ply_2dgs(model, ply_path)
    print(
        f"Done. {model.num_points} gaussians. Renders saved to {OUT_DIR}, "
        f"final scene saved to {ply_path}.",
        flush=True,
    )
    print(
        f"Best held-out PSNR {best[0]:.2f} dB @ step {best[1]} -> "
        f"{OUT_DIR / 'garden_2dgs_best.ply'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
