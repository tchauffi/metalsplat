"""Adaptive density control (split / clone / prune) for `Gaussian2DModel`,
mirroring `metalsplat.densify`'s 3DGS version. See that module's docstring
for the overall rationale (why fixed gaussian counts fail).

Deliberately a parallel module rather than a generalized/shared
implementation: the two model classes have no common base class anywhere
in this codebase (see `sh_color.py`'s docstring for why), and the only
genuinely 2D-specific piece -- confining a split child's random offset to
the parent's tangent plane instead of sampling a full 3D offset -- is
small enough that duplicating the surrounding percentile/threshold/clone
logic is cheaper and safer than introducing a shared abstraction across
that boundary.

`DensifyStats`, `NEW_GAUSSIAN` and `reset_opacity` are reused directly
from `metalsplat.densify`/`metalsplat.optim` -- they operate purely
through `.opacities`/`.raw_opacities`/optimizer param groups, with no
model-reconstruction step, so nothing about them is 3DGS-specific.
"""

from __future__ import annotations

import torch

from metalsplat.densify import DensifyStats, select_candidates, split_clone_prune
from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.utils.quaternion import quat_to_rotmat


def densify_and_prune_2dgs(
    model: Gaussian2DModel,
    grad_accum: torch.Tensor,  # (N,) accumulated means2d-grad norms since last call
    grad_count: torch.Tensor,  # (N,) number of steps each gaussian was visible
    grad_percentile: float = 0.8,
    grad_threshold: float | None = None,
    prune_opacity_thresh: float = 0.005,
    split_scale_factor: float = 1.6,
    split_scale_quantile: float = 0.5,
    max_points: int | None = None,
    pixel_count: torch.Tensor | None = None,  # (N,) covered pixels, see below
    max_world_size: float | None = None,
) -> tuple[Gaussian2DModel, DensifyStats]:
    """Splits, clones and prunes a `Gaussian2DModel`, returning the new
    model and stats. See `metalsplat.densify.densify_and_prune` for the
    full parameter semantics -- the differences are how a split child's
    random offset is sampled (confined to the parent's tangent plane,
    since a 2D splat has no third, depth-axis scale to offset along), the
    relative `split_scale_quantile` bar described below (3DGS's version
    still takes an absolute `scene_scale`), and the optional `pixel_count`
    normalization further down.

    `split_scale_quantile` sets the split-vs-clone bar: a selected
    gaussian is *split* (replaced by two smaller, tangentially-offset
    children) if its larger scale axis exceeds this quantile of the whole
    population's, and *cloned* (duplicated in place at the same size)
    otherwise.

    This is deliberately relative to the model's own current size
    distribution rather than an absolute world-space length, because the
    absolute version has no way to stay calibrated. An initial gaussian
    size is naturally chosen in *screen* units (a target pixel radius --
    see `train_garden_2dgs.calibrate_initial_scale`), while an absolute
    bar like the sparse cloud's median nearest-neighbor spacing lives in
    *world* units, and nothing keeps the two in step: change the training
    resolution, the scene, or the initialization and they drift apart. On
    the garden scene at half resolution they landed 3.6x apart, which put
    every gaussian below the bar -- so *nothing* qualified to split for
    hundreds of steps and densification degenerated into pure
    clone-then-drift (duplicate in place, then separate only as slowly as
    gradient descent moves them apart). That is at its worst exactly where
    it hurts most: where many gaussians overlap, the drift signal is
    shared among all of them, so complex regions stay a blurry pile long
    after simple ones have sharpened. Working around it by inflating the
    initial size until it approached the bar cost 6.2x the initial
    overdraw (140 vs 22.5 gaussians composited per pixel, measured).

    A relative bar removes the failure mode rather than compensating for
    it: "larger than typical" is always answerable, at any initialization
    and any resolution. It also keeps adapting -- late in training, when
    high-error gaussians tend to be small detail rather than large blobs,
    most candidates fall below the bar and clone.

    Note this does not change how *many* gaussians a round adds, only
    which kind: split drops the parent and adds two children, clone keeps
    the parent and adds one, so both are net +1 per candidate however the
    partition falls. Unlike `pixel_count` below, it therefore cannot
    affect whether densification terminates.

    `pixel_count`, if given, is the per-gaussian covered-pixel count from
    `render_2dgs(..., pixel_count_accum=...)`, and replaces `grad_count`
    as the divisor for `grad_accum`. `grad_accum` is a sum over every
    (pixel, gaussian) pair, so dividing by the number of *views* leaves a
    signal proportional to each gaussian's screen **area**: on the garden
    scene the mean signal falls ~35x from the nearest depth quintile to
    the farthest, which with a fixed absolute `grad_threshold` means
    distant gaussians are essentially never selected however badly they
    reconstruct their region -- the far field silently stops densifying.
    Dividing by covered pixels instead makes it a per-pixel mean, which is
    comparable across depths.

    **This removes the brake that makes densification terminate, so it is
    not safe on its own.** The un-normalized sum falls when a gaussian is
    split or cloned, because each child covers fewer pixels than the
    parent did; that decay is what eventually puts every gaussian below a
    fixed threshold and stops growth. Divide it by coverage and a clone
    scores exactly what its parent scored, so it qualifies again on the
    next round, and the next. Measured on the garden scene against an
    otherwise identical run: gaussians added per round decays 11% -> 4.8%
    -> 4.0% un-normalized (count converges), and *accelerates* normalized
    (138k -> 700k by step 1500, ~3M by 3000). Recalibrating the threshold
    each round bounds the rate but not the total, since a percentile always
    selects a fixed fraction -- a steady ~7.5% per round still compounds.
    Pair it with `max_points`, or with an absolute threshold tuned for the
    normalized scale (the two differ by ~100x), or leave it off.

    This is pixel-aware in the sense of Pixel-GS (Zhang et al. 2024), but
    not that paper's rule: it weights *views* by coverage to accelerate
    large gaussians, whereas this normalizes *by* coverage to stop small
    ones being starved. The failure modes are opposite, so the direction
    is too; a gaussian covering one pixel in every view is unaffected by
    the paper's weighting and rescued by this.

    `max_points` is a hard cap on split/clone growth, as in
    `metalsplat.densify.densify_and_prune`; pruning runs at the cap too.

    `grad_count` is still what defines visibility, so a gaussian that was
    in frustum but composited into no pixel is skipped rather than
    dividing by zero.

    Pruning follows the reference's order, as in
    `metalsplat.densify.densify_and_prune`: split and clone first, then
    prune the result -- rows with opacity at most `prune_opacity_thresh`,
    or (if given) a larger scale axis above `max_world_size` (the reference
    uses 0.1 x the camera extent). A large high-gradient splat is therefore
    split rather than dropped, and a clone of a low-opacity splat is pruned
    with it. The reference's 20px screen-size check is not reproduced: its
    densification_postfix zeroes max_radii2D first, so it never fires.
    """
    device = model.means.device
    n = model.num_points

    visible = grad_count > 0
    avg_grad = torch.zeros(n, device=device)
    if pixel_count is None:
        avg_grad[visible] = grad_accum[visible] / grad_count[visible]
    else:
        # Only gaussians that actually covered a pixel have a meaningful
        # per-pixel mean; the rest keep 0 and fall below any threshold.
        # `pixel_count` is wired up independently of `grad_count` (it needs
        # `pixel_count_accum=` on `render_2dgs`), so a caller who passes one
        # and forgets the other narrows every gaussian away here, and
        # select_candidates then selects nothing rather than calibrating on
        # an empty set.
        covered = visible & (pixel_count > 0)
        avg_grad[covered] = grad_accum[covered] / pixel_count[covered]
        visible = covered
    candidates, threshold = select_candidates(
        avg_grad, visible, grad_percentile, grad_threshold, max_points
    )

    scales = model.scales.detach()  # (N, 2)

    # Relative split-vs-clone bar -- see the docstring for why this is a
    # quantile of the population's own sizes rather than an absolute
    # world-space length. Taken over every gaussian, not just the selected
    # candidates, so it answers "is this larger than typical for the
    # model?" rather than "is it larger than the other high-error ones?".
    extent = scales.max(dim=-1).values
    if bool(candidates.any()):
        split_bar = torch.quantile(extent, split_scale_quantile)
        is_large = candidates & (extent > split_bar)
    else:
        is_large = candidates
    split_idx = is_large.nonzero(as_tuple=True)[0]
    clone_idx = (candidates & ~is_large).nonzero(as_tuple=True)[0]

    # Two children per split parent, offset within the parent's tangent
    # plane only: a 2D splat has no third axis to offset along.
    rotmat = quat_to_rotmat(model.quats.detach()[split_idx])  # (K, 3, 3)
    tangent = rotmat[..., :, :2]  # (K, 3, 2): t_u/t_v columns only, no normal axis
    samples = torch.randn(2, split_idx.numel(), 2, device=device) * scales[split_idx]
    split_offsets = torch.einsum("kij,skj->ski", tangent, samples)  # (2, K, 3)

    return split_clone_prune(
        model,
        split_idx,
        clone_idx,
        split_offsets,
        split_scale_factor=split_scale_factor,
        prune_opacity_thresh=prune_opacity_thresh,
        max_world_size=max_world_size,
        threshold=threshold,
    )


def prune_low_opacity_2dgs(
    model: Gaussian2DModel, prune_opacity_thresh: float = 0.005
) -> tuple[Gaussian2DModel, int, torch.Tensor]:
    """See `metalsplat.densify.prune_low_opacity` -- identical logic,
    `Gaussian2DModel` reconstruction.
    """
    device = model.means.device
    keep_mask = model.opacities.detach() > prune_opacity_thresh
    n_pruned = int((~keep_mask).sum().item())
    if n_pruned == 0:
        return model, 0, torch.arange(model.num_points, device=device)
    return model.select(keep_mask), n_pruned, keep_mask.nonzero(as_tuple=True)[0]
