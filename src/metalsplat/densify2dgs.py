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

from metalsplat.densify import DensifyStats
from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.optim import NEW_GAUSSIAN
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

    `grad_count` is still what defines visibility, so a gaussian that was
    in frustum but composited into no pixel is skipped rather than
    dividing by zero.
    """
    device = model.means.device
    n = model.num_points

    visible = grad_count > 0
    n_before = n
    if not bool(visible.any()) or (max_points is not None and n >= max_points):
        unchanged = torch.arange(n, device=device)
        return model, DensifyStats(
            n_before,
            0,
            0,
            0,
            n_before,
            float(grad_threshold or 0.0),
            unchanged,
            unchanged,
        )

    avg_grad = torch.zeros(n, device=device)
    if pixel_count is None:
        avg_grad[visible] = grad_accum[visible] / grad_count[visible]
    else:
        # Only gaussians that actually covered a pixel have a meaningful
        # per-pixel mean; the rest keep 0 and fall below any threshold.
        covered = visible & (pixel_count > 0)
        avg_grad[covered] = grad_accum[covered] / pixel_count[covered]
        visible = covered
    if grad_threshold is None:
        threshold = float(torch.quantile(avg_grad[visible], grad_percentile))
    else:
        threshold = float(grad_threshold)
    candidates = visible & (avg_grad >= threshold)

    means = model.means.detach()
    scales = model.scales.detach()  # (N, 2)
    quats = model.quats.detach()
    opacities = model.opacities.detach()
    color_like = (model.colors if model.sh_degree == 0 else model.raw_sh).detach()

    # Relative split-vs-clone bar -- see the docstring for why this is a
    # quantile of the population's own sizes rather than an absolute
    # world-space length. Taken over every gaussian, not just the selected
    # candidates, so it answers "is this larger than typical for the
    # model?" rather than "is it larger than the other high-error ones?".
    extent = scales.max(dim=-1).values
    split_bar = torch.quantile(extent, split_scale_quantile)
    is_large = candidates & (extent > split_bar)
    is_small = candidates & ~is_large

    split_idx = is_large.nonzero(as_tuple=True)[0]
    clone_idx = is_small.nonzero(as_tuple=True)[0]

    color_shape = tuple(color_like.shape[1:])  # (3,) or (9, 3)

    def _expand2(x):  # (K, *color_shape) -> (2*K, *color_shape), two identical copies
        return x.unsqueeze(0).expand(2, *([-1] * x.dim())).reshape(-1, *color_shape)

    def _empty(*shape):
        return means.new_zeros(0, *shape)

    if split_idx.numel() > 0:
        rotmat = quat_to_rotmat(quats[split_idx])  # (K, 3, 3)
        tangent = rotmat[..., :, :2]  # (K, 3, 2): t_u/t_v columns only, no normal axis
        std = scales[split_idx]  # (K, 2)
        samples = torch.randn(2, split_idx.numel(), 2, device=device) * std
        offsets = torch.einsum("kij,skj->ski", tangent, samples)  # (2, K, 3)
        split_means = (means[split_idx].unsqueeze(0) + offsets).reshape(-1, 3)
        split_scales = (
            (scales[split_idx] / split_scale_factor)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .reshape(-1, 2)
        )
        split_quats = quats[split_idx].unsqueeze(0).expand(2, -1, -1).reshape(-1, 4)
        split_opacities = opacities[split_idx].unsqueeze(0).expand(2, -1).reshape(-1)
        split_color_like = _expand2(color_like[split_idx])
    else:
        split_means, split_scales, split_quats = _empty(3), _empty(2), _empty(4)
        split_opacities, split_color_like = _empty(), _empty(*color_shape)

    clone_means = means[clone_idx]
    clone_scales = scales[clone_idx]
    clone_quats = quats[clone_idx]
    clone_opacities = opacities[clone_idx]
    clone_color_like = color_like[clone_idx]

    keep_mask = ~is_large & (opacities > prune_opacity_thresh)

    final_means = torch.cat([means[keep_mask], split_means, clone_means], dim=0)
    final_scales = torch.cat([scales[keep_mask], split_scales, clone_scales], dim=0)
    final_quats = torch.cat([quats[keep_mask], split_quats, clone_quats], dim=0)
    final_opacities = torch.cat(
        [opacities[keep_mask], split_opacities, clone_opacities], dim=0
    )
    final_color_like = torch.cat(
        [color_like[keep_mask], split_color_like, clone_color_like], dim=0
    )

    if model.sh_degree == 0:
        new_model = Gaussian2DModel(
            final_means,
            scales=final_scales,
            quats=final_quats,
            opacities=final_opacities,
            colors=final_color_like,
        ).to(device)
    else:
        new_model = Gaussian2DModel(
            final_means,
            scales=final_scales,
            quats=final_quats,
            opacities=final_opacities,
            sh_degree=model.sh_degree,
            sh_coeffs=final_color_like,
            active_sh_degree=model.active_sh_degree,
        ).to(device)

    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    fresh = torch.full(
        (2 * split_idx.numel() + clone_idx.numel(),),
        NEW_GAUSSIAN,
        dtype=torch.int64,
        device=device,
    )
    source_index = torch.cat([keep_idx, fresh])
    parent_index = torch.cat([keep_idx, split_idx, split_idx, clone_idx])

    n_after = final_means.shape[0]
    n_pruned = int((~keep_mask & ~is_large).sum().item())
    stats = DensifyStats(
        n_before=n_before,
        n_split=int(split_idx.numel()),
        n_cloned=int(clone_idx.numel()),
        n_pruned=n_pruned,
        n_after=n_after,
        grad_threshold=threshold,
        source_index=source_index,
        parent_index=parent_index,
    )
    return new_model, stats


def prune_low_opacity_2dgs(
    model: Gaussian2DModel, prune_opacity_thresh: float = 0.005
) -> tuple[Gaussian2DModel, int, torch.Tensor]:
    """See `metalsplat.densify.prune_low_opacity` -- identical logic,
    `Gaussian2DModel` reconstruction.
    """
    device = model.means.device
    n_before = model.num_points
    keep_mask = model.opacities.detach() > prune_opacity_thresh
    n_pruned = int((~keep_mask).sum().item())
    if n_pruned == 0:
        return model, 0, torch.arange(n_before, device=device)

    means = model.means.detach()[keep_mask]
    scales = model.scales.detach()[keep_mask]
    quats = model.quats.detach()[keep_mask]
    opacities = model.opacities.detach()[keep_mask]

    if model.sh_degree == 0:
        new_model = Gaussian2DModel(
            means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=model.colors.detach()[keep_mask],
        ).to(device)
    else:
        new_model = Gaussian2DModel(
            means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_degree=model.sh_degree,
            sh_coeffs=model.raw_sh.detach()[keep_mask],
            active_sh_degree=model.active_sh_degree,
        ).to(device)

    return new_model, n_pruned, keep_mask.nonzero(as_tuple=True)[0]
