"""Trains a Gaussian2DModel to reconstruct the real "garden" COLMAP scene in
data/garden, using the 2D Gaussian Splatting paper's (Huang et al. 2024)
own hyperparameters: per-group learning rates (feature/opacity/scaling/
rotation held constant, only position decayed), lambda_dssim=0.2,
lambda_normal (active after iteration 7000; 0.01 here rather
than the paper's 0.05 -- see the constant), lambda_dist=100 (active
after iteration 3000; the paper's own weight for unbounded scenes, which
garden is -- 1000 for bounded ones -- rather than the reference repo's
shipped 0.0 default) and SH degree 3 grown by one band every 1000 steps --
all read directly off the official reference implementation's
`arguments/__init__.py` `OptimizationParams` and `train.py`'s loss/schedule
(https://github.com/hbb1/2d-gaussian-splatting).

The schedule is shortened rather than copied: this runs `NUM_ITERS`
15,000 steps against the paper's 30,000, and stops densifying at
`DENSIFY_STOP` 9,000 against its `densify_until_iter=15000`. Both are
budget choices for a single-GPU-on-a-laptop run, not claims about the
paper; every *rate* and *weight* below is the published one, and the
loss/regularizer start iterations (7000 for normal, 3000 for distortion)
are unchanged, so the shortened run still spends its last third under the
full regularized objective. Raise both back to the paper's values for a
faithful reproduction.

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
runs on the paper's own schedule (`densify_from_iter=500`, every
`densification_interval=100` steps, `opacity_reset_interval=3000`, and
`densify_until_iter` shortened to `DENSIFY_STOP` as described above) with
one more deliberate deviation: the
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

The initial gaussian size uses `train_garden.py`'s own 3px screen-radius
target. It used to be raised to 10px here, to work around
`densify_and_prune_2dgs` splitting only gaussians larger than an absolute
world-space bar (the sparse cloud's median nearest-neighbor spacing):
a 3px target lands well below that bar, so nothing qualified to split for
hundreds of steps and densification degenerated into pure clone-then-drift,
which is slowest exactly where gaussians overlap and share the drift
signal. That bar is now relative to the model's own size distribution
(`split_scale_quantile`, see `metalsplat.densify2dgs`), so splits fire
from the first round at any initial size and the workaround is no longer
needed -- which matters, because it was expensive: a 10px target stacks
140 semi-transparent disks on the average pixel at step 0 against 22.5 at
3px (measured on this scene at `RESOLUTION_DOWNSCALE=2.0`), and every one
of those is depth-spread the distortion regularizer then has to undo.

Note the two calibrations never agreed in the first place and could not
be made to: the initial size is set in *screen* units and depends on
`RESOLUTION_DOWNSCALE`, while the old bar was a fixed *world* length. At
half resolution the same 10px target produces twice the world-space size
it would at full resolution, so any hand-tuned pairing of the two silently
breaks the moment the training resolution changes.

This does not fully close the gap between complex/overlapping regions and
simple ones -- some of that is 2DGS's own planarity prior working against
non-planar, repeated detail (foliage, moss) rather than a densification
bug -- but it removes the easily-avoidable part of it.

One more gap, found by timing a real 30k-step run (wall-clock time per
1000 steps kept climbing well past the point densification stops):
opacity resets used to keep firing every 3000 steps all the way to
`NUM_ITERS`, but pruning here only ever ran *inside*
`densify_and_prune_2dgs`, which itself stops at `DENSIFY_STOP`. Every
reset past that point caps a fresh batch of gaussians' opacity near zero
with nothing left to clean them up afterward -- they sit there forever,
still costing full projection/rasterization compute every step while
contributing ~nothing to the image, and each subsequent reset adds
another batch. `OPACITY_STOP_RESET` fixes that by retiring the resets
while densification is still running, so every reset is still followed by
rounds that can rebuild what it dissolves -- see that constant for why its
bound has to be exclusive, and what one reset landing exactly on
`DENSIFY_STOP` costs once the normal-consistency term is active.

`prune_low_opacity_2dgs` also runs on its own schedule rather than only
inside `densify_and_prune_2dgs` (the same standalone-prune answer
`train_garden.py` uses for 3DGS), but it stops at `DENSIFY_STOP` here
rather than running to `NUM_ITERS` -- see `PRUNE_STOP`.

Usage: uv run python examples/train_garden_2dgs.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from metalsplat import Gaussian2DModel, render_2dgs, save_ply_2dgs
from metalsplat.camera import camera_extent
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.densify import reset_opacity
from metalsplat.densify2dgs import densify_and_prune_2dgs, prune_low_opacity_2dgs
from metalsplat.losses import (
    distortion_loss,
    gaussian_splatting_loss,
    normal_consistency_loss,
    psnr,
)
from metalsplat.optim import SparseAdam, migrate_optimizer_state, sh_lr_scale

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
OUT_DIR = Path(__file__).parent
# >1.0 resizes every loaded image by 1/RESOLUTION_DOWNSCALE (2.0 = half
# resolution) before training -- fewer pixels means faster steps (every
# render/backward pays for H*W pixels), at the cost of fine detail. See
# metalsplat.data.colmap.load_colmap_scene's docstring; intrinsics are
# rescaled automatically, no other change needed.
RESOLUTION_DOWNSCALE = 2.0

# --- 2DGS paper defaults (arguments/__init__.py's OptimizationParams) ---
NUM_ITERS = 15_000
FEATURE_LR = 0.0025  # paper splits this into f_dc (this rate) / f_rest (this / 20);
# this repo keeps SH in one tensor, so the split is a per-coefficient
# multiplier on it (metalsplat.optim.sh_lr_scale).
OPACITY_LR = 0.05
SCALING_LR = 0.005
ROTATION_LR = 0.001
LAMBDA_DSSIM = 0.2
# The paper's value is 0.05. Measured deviation, for the same reason as
# the position learning rate above: the paper's weight is calibrated
# against its own scene normalization, and the term's *gradient* here is
# not on the scale that weight assumes. On this scene at the 11k-gaussian
# checkpoint, 0.05 makes the regularizer outweigh the photometric loss on
# every geometric parameter -- gradient on means 1.6e-5 against 1.2e-5,
# on quats 1.5e-6 against 1.0e-6, on opacity 1.5x the photometric median.
# A term that dominates the data term steers the geometry rather than
# regularizing it, and the surfaces visibly break up shortly after
# LAMBDA_NORMAL_START_ITER.
#
# Measured over matched 3000-step runs (term on at 1500, both of this
# file's opacity-schedule fixes in place), against a no-regularizer
# control at 22.8% folded / 24.49 dB:
#
#   0.05  8.6% folded, 23.21 dB -- best surface coherence, but the normal
#         map keeps a needle structure and scale-ratio p99 rises to 16.4
#         against the control's 11.5
#   0.01 18.0% folded, 25.93 dB -- smooth normal map, scale ratio p99
#         11.4 (i.e. no needles), and the best PSNR of any arm including
#         the control
#
# Ramping 0.05 in over 500 steps instead of switching it on was also
# measured (9.2% folded, 23.38 dB): indistinguishable from the step
# change, so the breakup is the term's sustained strength, not a shock at
# switch-on. Raise this back toward 0.05 if surface quality for meshing
# matters more than the render, and watch the normal map rather than PSNR
# when you do.
LAMBDA_NORMAL = 0.01
LAMBDA_NORMAL_START_ITER = 7000  # paper: `lambda_normal if iteration > 7000 else 0`
# The reference repo ships 0.0; the paper uses 100 (unbounded, which garden
# is) / 1000 (bounded). Those values transfer directly now that the
# distortion map is the paper's actual regularizer -- the squared pairwise
# second moment of the per-pixel weight distribution over *normalized*
# depth, matching the official CUDA rasterizer (see
# metalsplat.reference.rasterize_2dgs_ref's module docstring, which also
# records the signed first-power variant this used to be and why that one
# made the geometry worse rather than better).
#
# Measured on this scene at the ~500k-gaussian checkpoint: the distortion
# term is ~1e-5 against a ~0.06 photometric loss, so 100 puts it at ~1.6%
# of the total -- a regularizer, not a second objective. The old 0.01 here
# was not a considered weight, it was the value the broken term had to be
# beaten down to before it stopped visibly wrecking the surfaces.
LAMBDA_DIST = 100.0
LAMBDA_DIST_START_ITER = 3000  # paper: `lambda_dist if iteration > 3000 else 0`
SH_DEGREE = 3
SH_DEGREE_INTERVAL = 1000  # paper: +1 band every 1000 steps (`oneupSHdegree`)
INIT_OPACITY = 0.1  # paper default (`inverse_sigmoid(0.1 * ones(...))`)
DENSIFY_START = 500  # paper: densify_from_iter
DENSIFY_STOP = 9_000  # paper: densify_until_iter
DENSIFY_INTERVAL = 100  # paper: densification_interval
# Divide the AbsGS signal by covered pixels rather than by view count, so
# the densification bar means the same thing at every depth (see
# densify2dgs). OFF, because it removes the brake that makes densification
# terminate.
#
# The un-normalized signal is a sum over the pixels a gaussian covers, so
# splitting or cloning one lowers its score and it stops re-qualifying:
# growth decays (measured on this scene, gaussians added per round: 11% ->
# 4.8% -> 4.0%) and the count converges. Dividing that sum by coverage
# removes the dependence on size, so a clone scores exactly what its parent
# did and qualifies again next round. Measured: 138k -> 700k by step 1500
# and still accelerating, ~3M by step 3000. Recalibrating the threshold
# every round instead of freezing it bounds the rate but not the total --
# a percentile always selects a fixed fraction, which still compounds
# (~7.5%/round, steady rather than decaying).
#
# Keep it False unless you pair it with a growth cap (max_points) or an
# absolute threshold tuned for the normalized scale. The two signals differ
# by ~100x, so a threshold calibrated under one is meaningless under the
# other.
PIXEL_NORMALIZED_DENSIFY = False
PRUNE_OPACITY_THRESH = 0.05  # paper: opacity_cull
# Paper: once past the first opacity reset, densify rounds also prune
# splats whose larger scale exceeds 0.1 x the camera extent (its 20px
# screen-size check never fires, see metalsplat.densify2dgs).
PRUNE_MAX_WORLD_FRACTION = 0.1
OPACITY_RESET_INTERVAL = (
    3000  # paper default (reset_opacity()'s own 0.01 cap matches too)
)
# Exclusive, and that is the whole point: the last reset lands at 6000,
# ~3000 steps before DENSIFY_STOP, so every reset is still followed by
# densify rounds that can rebuild what it dissolves. An inclusive bound
# here would fire one final reset at 9000 -- the exact step densification
# retires -- which caps every gaussian at 0.01 with nothing left to grow
# the model back, while PRUNE_INTERVAL keeps culling everything that
# drifts under STANDALONE_PRUNE_OPACITY_THRESH. Measured on a shortened
# run with the normal-consistency term active: that one reset cost 10.4k
# gaussians at the very next prune round and ~2-3k every round after, a
# fifth of the model, monotonically, with no densification left to answer
# (the same schedule without the normal term recovers in ~300 steps, so
# this only bites once LAMBDA_NORMAL is on -- which is why it reads as
# "the normal loss broke the splats").
#
# The paper has no explicit equivalent: there, resets live inside the
# densification block and so stop at densify_until_iter by construction --
# i.e. upstream's bound is exclusive of the no-densification regime too.
OPACITY_STOP_RESET = 9000
# --- end paper defaults ---

# Calibrated on the first densification round, then frozen -- see module
# docstring for why this replaces the paper's literal densify_grad_threshold.
DENSIFY_GRAD_PERCENTILE = 0.9

# Standalone prune schedule, matching train_garden.py's identical one: it
# runs on its own interval rather than only inside densify_and_prune_2dgs.
#
# It stops at DENSIFY_STOP, though, which train_garden.py's does not: a
# prune with no densification behind it is a one-way drain. It existed to
# clean up the gaussians that post-DENSIFY_STOP opacity resets stranded,
# and OPACITY_STOP_RESET's exclusive bound now means no reset ever lands
# there, so the reason is gone -- while the cost is not. With the
# normal-consistency term active, that term demotes the opacity of the
# gaussians whose normals disagree with the depth surface (measured: it
# flips 16% of gaussians from "the photometric loss wants more opacity" to
# "the total wants less"), so a prune running past DENSIFY_STOP deletes
# ~2k of them per round, monotonically, for the rest of the run with
# nothing able to replace them. Measured on a shortened run: stopping it
# at DENSIFY_STOP keeps 21k more gaussians (163k vs 142k) and gains 0.5 dB
# for identical surface coherence. This is also what the paper does --
# there, pruning only ever runs inside the densification block.
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
PRUNE_STOP = DENSIFY_STOP
PRUNE_INTERVAL = 100

EVAL_EVERY = 1000
EVAL_HOLDOUT_STRIDE = 8
EVAL_IMAGES_SAVED = 3


def _as_float(x: torch.Tensor | float) -> float:
    """Log helper: a regularizer term is a plain 0.0 when its weight is 0
    (the term is skipped rather than built), and a tensor otherwise. `.item()`
    rather than `float()` so a still-attached tensor doesn't warn.
    """
    return x.item() if torch.is_tensor(x) else float(x)


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
    scene = load_colmap_scene(DATA_ROOT, device=DEVICE, downscale=RESOLUTION_DOWNSCALE)
    n_images = len(scene.cameras)
    eval_idx = list(range(0, n_images, EVAL_HOLDOUT_STRIDE))
    train_idx = [i for i in range(n_images) if i not in set(eval_idx)]
    cam0 = scene.cameras[0]
    print(
        f"{n_images} images: {len(train_idx)} train, {len(eval_idx)} eval "
        f"({cam0.img_width}x{cam0.img_height}, downscale={RESOLUTION_DOWNSCALE})",
        flush=True,
    )
    print(f"{scene.points.shape[0]} initial sparse points")

    # scene_scale is only the position-learning-rate calibration now --
    # densify_and_prune_2dgs's split-vs-clone bar no longer reads it (see
    # module docstring and metalsplat.densify2dgs).
    scene_scale = estimate_scene_scale(scene.points)
    init_scale = calibrate_initial_scale(
        scene.points,
        [scene.cameras[i] for i in train_idx],  # held-out views stay unseen
        scene_scale,
        target_pixel_radius=3.0,
    )
    print(
        f"Scene scale (median NN spacing): {scene_scale:.4f}, calibrated initial gaussian scale: {init_scale:.5f}",
        flush=True,
    )

    extent = camera_extent([scene.cameras[i] for i in train_idx])
    max_world_size = PRUNE_MAX_WORLD_FRACTION * extent
    print(
        f"camera extent {extent:.3f}: pruning scales > {max_world_size:.3f}",
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
        if m.sh_degree == 0:
            color_group = {"params": [m.raw_colors], "lr": FEATURE_LR}
        else:
            color_group = {
                "params": [m.raw_sh],
                "lr": FEATURE_LR,
                "lr_scale": sh_lr_scale(m.raw_sh),
            }
        return SparseAdam(
            [
                {"params": [m.means], "lr": lr_means},
                color_group,
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
                psnrs.append(psnr(aux.image, scene.images[idx]).item())
                if k < EVAL_IMAGES_SAVED:
                    save_image(
                        aux.image, OUT_DIR / f"garden_2dgs_eval_{k}_step{step}.png"
                    )
                    # aux.normal is an alpha-weighted *sum* (magnitude ~alpha),
                    # not a unit vector -- divide by alpha and renormalize
                    # before mapping to [0, 1], same correction
                    # normal_consistency_loss and render_video_2dgs.py's
                    # normal_to_rgb make, or semi-transparent regions (edges,
                    # foliage) wash out toward grey instead of showing their
                    # actual orientation.
                    alpha = (1.0 - aux.final_T).clamp_min(1e-6)
                    unit_normal = torch.nn.functional.normalize(
                        aux.normal / alpha.unsqueeze(-1), dim=-1
                    )
                    covered = (1.0 - aux.final_T) > 0.1
                    save_image(
                        (unit_normal * 0.5 + 0.5) * covered[..., None],
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
    # Covered-pixel counts, the per-pixel normalizer for grad_accum -- see
    # PIXEL_NORMALIZED_DENSIFY and densify2dgs's docstring.
    pixel_count = torch.zeros(model.num_points, device=DEVICE)
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
            pixel_count_accum=pixel_count if PIXEL_NORMALIZED_DENSIFY else None,
        )
        photo_loss = gaussian_splatting_loss(
            aux.image, target, lambda_dssim=LAMBDA_DSSIM
        )

        # Paper's exact iteration-gated regularizer weights. A zero-weighted
        # term is skipped outright rather than multiplied by 0.0: building it
        # anyway still runs its forward and drags its whole subgraph through
        # backward for a contribution that is identically zero. Both terms
        # are gated on by the paper's schedule (distortion from step 3000,
        # normal from 7000), so this only pays off early -- but that early
        # stretch is where the step count is highest, and building both
        # unconditionally measured ~1.9ms of a ~40ms step.
        lambda_normal = LAMBDA_NORMAL if step > LAMBDA_NORMAL_START_ITER else 0.0
        lambda_dist = LAMBDA_DIST if step > LAMBDA_DIST_START_ITER else 0.0

        loss = photo_loss
        normal_loss: torch.Tensor | float = 0.0
        dist_loss: torch.Tensor | float = 0.0
        if lambda_normal > 0.0:
            normal_loss = lambda_normal * normal_consistency_loss(
                aux.normal, aux.depth, 1.0 - aux.final_T, cam
            )
            loss = loss + normal_loss
        if lambda_dist > 0.0:
            dist_loss = lambda_dist * distortion_loss(aux.distortion)
            loss = loss + dist_loss
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
                f"(photo {photo_loss.item():.5f}  normal {_as_float(normal_loss):.5f}  "
                f"dist {_as_float(dist_loss):.5f})  "
                f"({elapsed:.1f}s elapsed, {elapsed / step:.2f}s/step)",
                flush=True,
            )

        if DENSIFY_START <= step <= DENSIFY_STOP and step % DENSIFY_INTERVAL == 0:
            model, stats = densify_and_prune_2dgs(
                model,
                grad_accum,
                grad_count,
                pixel_count=pixel_count if PIXEL_NORMALIZED_DENSIFY else None,
                grad_percentile=DENSIFY_GRAD_PERCENTILE,
                prune_opacity_thresh=PRUNE_OPACITY_THRESH,
                grad_threshold=densify_threshold,
                max_world_size=(
                    max_world_size
                    if OPACITY_RESET_INTERVAL and step > OPACITY_RESET_INTERVAL
                    else None
                ),
            )
            # `stats.grad_threshold` is None when the round did nothing and
            # so never computed a bar. Freezing that would leave the
            # threshold at a placeholder for the rest of training, and
            # since every gradient clears a placeholder of 0, *every*
            # visible gaussian would split or clone every round. Skip the
            # freeze and recalibrate on the next round that does work.
            if densify_threshold is None and stats.grad_threshold is not None:
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
            pixel_count = torch.zeros(model.num_points, device=DEVICE)
            print(
                f"  densify @ step {step}: {stats.n_before} -> {stats.n_after} "
                f"(+{stats.n_split} split, +{stats.n_cloned} cloned, -{stats.n_pruned} pruned)",
                flush=True,
            )

        if (
            OPACITY_RESET_INTERVAL
            and step % OPACITY_RESET_INTERVAL == 0
            and step < OPACITY_STOP_RESET
        ):
            reset_opacity(model, optimizer=optimizer)
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
                pixel_count = torch.zeros(model.num_points, device=DEVICE)
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
