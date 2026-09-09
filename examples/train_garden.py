"""Trains a GaussianModel to reconstruct the real "garden" COLMAP scene in
data/garden -- random training view each step, the standard 3DGS L1+D-SSIM
loss, SparseAdam (see metalsplat.optim) with a decayed means learning rate so
gaussians outside the sampled view this step aren't spuriously decayed,
periodic adaptive density control (split/clone/prune), and periodic
loss/PSNR logging with saved renders for a couple of held-out eval views.
Optional spherical harmonics (SH_DEGREE=2) for view-dependent color.

Usage: uv run python examples/train_garden.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from metalsplat import GaussianModel, render, save_ply
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.densify import densify_and_prune, prune_low_opacity, reset_opacity
from metalsplat.filter3d import carry_filter_3d, compute_3d_filter
from metalsplat.losses import gaussian_splatting_loss
from metalsplat.optim import SparseAdam, migrate_optimizer_state
from metalsplat.seed import seed_uncovered_regions

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
OUT_DIR = Path(__file__).parent
NUM_ITERS = 30_000
EVAL_EVERY = 1000
LR_OTHER_INIT = 0.01
LR_OTHER_FINAL = LR_OTHER_INIT * 0.1  # color/scale/rotation are decayed, like means
LR_OPACITY = 0.05
SH_DEGREE = 3  # 0 = plain RGB; 1..3 = spherical harmonics (3 = the 3DGS default)
SH_DEGREE_INTERVAL = 0  # 0 disables (fit every band from the start)
# Mip-Splatting's 3D smoothing filter: band-limits each gaussian to the
# finest detail any training camera resolved it at, so reconstruction
# cannot invent sub-sampling-rate structure that shows up as speckle when
# the camera moves closer than any training view. 0 disables.
FILTER_3D_SCALE = 0.2  # the Mip-Splatting paper's value
# A full recompute is O(gaussians x cameras) -- 184ms at 438k gaussians and
# 185 views, and it grows linearly, so running it after every densify,
# seed and prune (once per ~77 steps) came to dominate a step as the model
# grew. Between refreshes the filter is carried across by inheritance;
# this is how often it is rebuilt from scratch to pick up drifting means.
FILTER_3D_REFRESH = 1000
LAMBDA_DSSIM = 0.2  # 3DGS default: loss = (1-lambda)*L1 + lambda*D-SSIM
EVAL_HOLDOUT_STRIDE = (
    8  # every 8th image is held out for eval, matching common NeRF/gsplat convention
)
EVAL_IMAGES_SAVED = 3  # how many held-out renders to write to disk (all are scored)
INIT_OPACITY = 0.1

# Adaptive density control schedule
DENSIFY_START = 1000
DENSIFY_STOP = 15_000
DENSIFY_INTERVAL = 250
# Densification bar. The percentile is only used to *calibrate*: the first
# densification round takes this quantile of the observed gradients and
# freezes it as an absolute threshold for the rest of training.
#
# A percentile applied every round never self-limits -- it promotes a fixed
# fraction however well-fit the model already is, so the count grows
# geometrically (measured: ~9.5% per round) until it hits DENSIFY_MAX_POINTS
# and then sits there with ~10% of the model brand new every 250 steps. With
# the old 400k cap that saturated early and the model settled; with a 5M cap
# and a 14000-step window it would have compounded to ~22M and never
# settled. An absolute bar decays naturally as gaussians fit their region,
# which is what the reference implementation relies on.
DENSIFY_GRAD_PERCENTILE = 0.9  # calibration quantile for the first round only
DENSIFY_GRAD_THRESHOLD = None  # None = calibrate from the first round
DENSIFY_MAX_POINTS = 5_000_000

# Loss-driven seeding schedule (fills sky/distant-background gaps that
# split/clone alone can't reach, since those start with ~no gaussians)
SEED_START = 100
SEED_STOP = 4000
SEED_INTERVAL = 250
SEED_RESIDUAL_THRESH = 0.15
SEED_COVERAGE_THRESH = 0.8
SEED_MAX_PER_CALL = 300

OPACITY_RESET_INTERVAL = 1500  # 0/None disables
OPACITY_RESET_STOP = 3000
OPACITY_RESET_VALUE = 0.01
PRUNE_START = 100
PRUNE_STOP = NUM_ITERS
PRUNE_INTERVAL = 100
PRUNE_OPACITY_THRESH = 0.005


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
    """Median nearest-neighbor distance between sparse points -- a natural
    "local spacing" unit for this scene's (COLMAP-arbitrary) coordinate
    scale, used both for the initial gaussian scale and to calibrate the
    means learning rate to that scale rather than a fixed absolute value.
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
    `target_pixel_radius` pixels on screen. Needed because the raw
    world-space nearest-neighbor spacing says nothing about how zoomed-in
    the cameras are -- for this scene it was ~0.12 world units but the
    camera sits close enough to a locally dense cluster that it projected
    to a ~23px radius, so 138k mostly-overlapping oversized gaussians
    produced an undifferentiated blur with no usable per-gaussian
    gradient signal.
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
    print(f"{scene.points.shape[0]} sparse points", flush=True)

    scene_scale = estimate_scene_scale(scene.points)
    init_scale = calibrate_initial_scale(scene.points, scene.cameras, scene_scale)
    print(
        f"Scene scale (median NN spacing): {scene_scale:.4f}, calibrated initial gaussian scale: {init_scale:.5f}",
        flush=True,
    )

    n_points = scene.points.shape[0]
    scales = torch.full((n_points, 3), init_scale, device=DEVICE)
    opacities = torch.full((n_points,), INIT_OPACITY, device=DEVICE)
    model = GaussianModel(
        scene.points,
        scales=scales,
        colors=scene.colors,
        opacities=opacities,
        sh_degree=SH_DEGREE,
    ).to(DEVICE)
    if SH_DEGREE_INTERVAL:
        model.active_sh_degree = 0
        print(f"SH degree starts at 0, +1 every {SH_DEGREE_INTERVAL} steps", flush=True)

    # Means live in the scene's (COLMAP-arbitrary) coordinate units, so their
    # learning rate is calibrated to the scene's own spacing unit rather
    # than a fixed absolute value -- a step of ~2% of the local gaussian
    # spacing per iteration. Decayed exponentially to ~1% of that over
    # training (matching standard 3DGS practice): without decay, positions
    # keep taking large steps late in training when they should be settling,
    # which shows up as color-cast drift and falling eval PSNR after the
    # model has already found good structure (observed empirically on this
    # scene -- PSNR peaked around step 500 then degraded with a constant LR).
    lr_means_init = 0.02 * scene_scale
    lr_means_final = lr_means_init * 0.01
    print(
        f"means lr: {lr_means_init:.5f} -> {lr_means_final:.5f} (exponential decay)",
        flush=True,
    )
    print(
        f"other lr: {LR_OTHER_INIT:.5f} -> {LR_OTHER_FINAL:.5f} (exponential decay)",
        flush=True,
    )
    print(f"opacity lr: {LR_OPACITY:.5f} (constant, see LR_OPACITY)", flush=True)

    # Param group order matters: the training loop updates groups 0 and 1's
    # LR each step and deliberately leaves group 2 (opacity) alone.
    def make_optimizer(
        m: GaussianModel, lr_means: float, lr_other: float
    ) -> torch.optim.Optimizer:
        color_param = m.raw_colors if m.sh_degree == 0 else m.raw_sh
        return SparseAdam(
            [
                {"params": [m.means], "lr": lr_means},
                {"params": [m.raw_scales, m.raw_quats, color_param], "lr": lr_other},
                {"params": [m.raw_opacities], "lr": LR_OPACITY},
            ]
        )

    optimizer = make_optimizer(model, lr_means_init, LR_OTHER_INIT)
    background = torch.zeros(3, device=DEVICE)

    def eval_and_save(step: int) -> None:
        # Score on *every* held-out view: a 3-view average swings by a
        # couple of dB run to run, which is enough to mistake noise for a
        # real regression (and vice versa) when tuning. Only the first few
        # are written out as images.
        with torch.no_grad():
            psnrs = []
            for k, idx in enumerate(eval_idx):
                pred = render(
                    model,
                    scene.cameras[idx],
                    background=background,
                    filter_3d=filter_3d,
                )
                torch.mps.synchronize()
                psnrs.append(psnr(pred, scene.images[idx]))
                if k < EVAL_IMAGES_SAVED:
                    save_image(pred, OUT_DIR / f"garden_eval_{k}_step{step}.png")
            mean_psnr = sum(psnrs) / len(psnrs)
            print(
                f"  eval PSNR ({len(psnrs)} held-out views): {mean_psnr:.2f} dB",
                flush=True,
            )

        # Held-out PSNR peaks before the last step on this scene (22.19 at
        # 3500 vs 21.24 at 5000), so keep the best checkpoint rather than
        # trusting the final one. Only trustworthy because every held-out
        # view is scored -- a 3-view average swings by more than this.
        nonlocal best
        if mean_psnr > best[0]:
            best = (mean_psnr, step)
            save_ply(model, OUT_DIR / "garden_best.ply")

    best = (float("-inf"), 0)  # (psnr, step) of the best checkpoint so far

    # The 3D filter is per-gaussian, so it has to be recomputed every time
    # the gaussian set changes -- a stale one is indexed by the old ordering
    # and would silently band-limit the wrong gaussians.
    def recompute_filter_3d():
        if not FILTER_3D_SCALE:
            return None
        return compute_3d_filter(
            model.means.detach(), scene.cameras, sampling_scale=FILTER_3D_SCALE
        )

    # Calibrated on the first densification round, then held fixed.
    densify_threshold = DENSIFY_GRAD_THRESHOLD

    filter_3d = recompute_filter_3d()
    if filter_3d is not None:
        seen = (filter_3d > 0).float().mean().item()
        print(
            f"3D filter: scale {FILTER_3D_SCALE}, {100 * seen:.1f}% of gaussians observed",
            flush=True,
        )

    print("Saving target/initial renders for eval view 0...", flush=True)
    save_image(scene.images[eval_idx[0]], OUT_DIR / "garden_target_0.png")
    eval_and_save(0)

    grad_accum = torch.zeros(model.num_points, device=DEVICE)
    grad_count = torch.zeros(model.num_points, device=DEVICE)
    previous_sh_degree = model.active_sh_degree

    start = time.time()
    for step in range(1, NUM_ITERS + 1):
        t = step / NUM_ITERS
        lr_means = lr_means_init * (lr_means_final / lr_means_init) ** t
        lr_other = LR_OTHER_INIT * (LR_OTHER_FINAL / LR_OTHER_INIT) ** t
        optimizer.param_groups[0]["lr"] = lr_means
        optimizer.param_groups[1]["lr"] = lr_other

        idx = train_idx[int(torch.randint(len(train_idx), (1,)).item())]
        cam = scene.cameras[idx]
        target = scene.images[idx]

        optimizer.zero_grad()
        aux = render(
            model,
            cam,
            background=background,
            return_aux=True,
            abs_grad_accum=grad_accum,
            filter_3d=filter_3d,
        )
        pred, valid, final_T = aux.image, aux.valid, aux.final_T
        loss = gaussian_splatting_loss(pred, target, lambda_dssim=LAMBDA_DSSIM)
        loss.backward()  # accumulates into grad_accum in place (AbsGS-style, see rendering.render)
        visible = valid > 0.5
        optimizer.step(visible)

        with torch.no_grad():
            grad_count[visible] += 1.0

        if step % 25 == 0 or step == 1:
            torch.mps.synchronize()
            elapsed = time.time() - start
            print(
                f"step {step:5d}  loss {loss.item():.5f}  n {model.num_points}  "
                f"({elapsed:.1f}s elapsed, {elapsed / step:.2f}s/step)",
                flush=True,
            )

        if DENSIFY_START <= step <= DENSIFY_STOP and step % DENSIFY_INTERVAL == 0:
            model, stats = densify_and_prune(
                model,
                grad_accum,
                grad_count,
                scene_scale=scene_scale,
                grad_percentile=DENSIFY_GRAD_PERCENTILE,
                max_points=DENSIFY_MAX_POINTS,
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
                optimizer, make_optimizer(model, lr_means, lr_other), stats.source_index
            )
            grad_accum = torch.zeros(model.num_points, device=DEVICE)
            grad_count = torch.zeros(model.num_points, device=DEVICE)
            filter_3d = carry_filter_3d(filter_3d, stats.parent_index)
            print(
                f"  densify @ step {step}: {stats.n_before} -> {stats.n_after} "
                f"(+{stats.n_split} split, +{stats.n_cloned} cloned, -{stats.n_pruned} pruned)",
                flush=True,
            )

        if SEED_START <= step <= SEED_STOP and step % SEED_INTERVAL == 0:
            model, seed_stats = seed_uncovered_regions(
                model,
                cam,
                pred.detach(),
                target,
                final_T,
                init_scale=init_scale,
                residual_thresh=SEED_RESIDUAL_THRESH,
                coverage_thresh=SEED_COVERAGE_THRESH,
                max_seeds_per_call=SEED_MAX_PER_CALL,
                near=0.2,
                max_points=DENSIFY_MAX_POINTS,
            )
            if seed_stats.n_seeded > 0:
                optimizer = migrate_optimizer_state(
                    optimizer,
                    make_optimizer(model, lr_means, lr_other),
                    seed_stats.source_index,
                )
                grad_accum = torch.zeros(model.num_points, device=DEVICE)
                grad_count = torch.zeros(model.num_points, device=DEVICE)
                filter_3d = carry_filter_3d(filter_3d, seed_stats.parent_index)
            print(
                f"  seed @ step {step}: {seed_stats.n_before} -> {seed_stats.n_after} "
                f"(+{seed_stats.n_seeded} seeded)",
                flush=True,
            )

        if (
            OPACITY_RESET_INTERVAL
            and step <= OPACITY_RESET_STOP
            and step % OPACITY_RESET_INTERVAL == 0
        ):
            reset_opacity(model, value=OPACITY_RESET_VALUE)
            print(
                f"  opacity reset @ step {step} (cap {OPACITY_RESET_VALUE})", flush=True
            )

        if PRUNE_START <= step <= PRUNE_STOP and step % PRUNE_INTERVAL == 0:
            model, n_pruned, prune_index = prune_low_opacity(
                model, prune_opacity_thresh=PRUNE_OPACITY_THRESH
            )
            if n_pruned > 0:
                optimizer = migrate_optimizer_state(
                    optimizer, make_optimizer(model, lr_means, lr_other), prune_index
                )
                grad_accum = torch.zeros(model.num_points, device=DEVICE)
                grad_count = torch.zeros(model.num_points, device=DEVICE)
                # Pruning only removes; survivors keep their radius unchanged.
                filter_3d = carry_filter_3d(filter_3d, prune_index)
                print(
                    f"  prune @ step {step}: -{n_pruned} (n={model.num_points})",
                    flush=True,
                )

        if FILTER_3D_SCALE and FILTER_3D_REFRESH and step % FILTER_3D_REFRESH == 0:
            # Means drift between refreshes, so inheritance is only an
            # approximation; rebuild from the cameras periodically.
            filter_3d = recompute_filter_3d()

        if SH_DEGREE_INTERVAL and step % SH_DEGREE_INTERVAL == 0:
            active = model.increase_sh_degree()
            if active != previous_sh_degree:
                print(f"  SH degree -> {active} @ step {step}", flush=True)
                previous_sh_degree = active

        if step % EVAL_EVERY == 0:
            eval_and_save(step)

    eval_and_save(NUM_ITERS)
    ply_path = OUT_DIR / "garden.ply"
    save_ply(model, ply_path)
    print(
        f"Done. Renders saved to {OUT_DIR}, final scene saved to {ply_path}", flush=True
    )
    print(
        f"Best held-out PSNR {best[0]:.2f} dB @ step {best[1]} -> {OUT_DIR / 'garden_best.ply'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
