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
    scene_scale: float,
    grad_percentile: float = 0.8,
    grad_threshold: float | None = None,
    prune_opacity_thresh: float = 0.005,
    split_scale_factor: float = 1.6,
    max_points: int | None = None,
) -> tuple[Gaussian2DModel, DensifyStats]:
    """Splits, clones and prunes a `Gaussian2DModel`, returning the new
    model and stats. See `metalsplat.densify.densify_and_prune` for the
    full parameter semantics (identical here) -- the only difference is
    how a split child's random offset is sampled (confined to the
    parent's tangent plane, since a 2D splat has no third, depth-axis
    scale to offset along).
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
    avg_grad[visible] = grad_accum[visible] / grad_count[visible]
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

    is_large = candidates & (scales.max(dim=-1).values > scene_scale)
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
