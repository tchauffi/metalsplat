"""Adaptive density control (split / clone / prune), the standard 3D
Gaussian Splatting recipe for keeping gaussian density matched to scene
content during training. Without this, a fixed gaussian count (e.g. one
per COLMAP sparse point) can't add detail where reconstruction is poor or
remove gaussians that have become useless -- overloaded gaussians instead
compensate by growing/recoloring in unstable ways, which shows up as
training-time color drift and falling held-out PSNR.

This is training-loop logic, not part of the core differentiable pipeline
(no Metal kernels here -- it operates between training steps, not inside
the forward/backward render).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metalsplat.gaussians import GaussianModel, logit
from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.optim import NEW_GAUSSIAN
from metalsplat.utils.quaternion import quat_to_rotmat


@dataclass
class DensifyStats:
    n_before: int
    n_split: int  # gaussians replaced by 2 new ones each
    n_cloned: int  # gaussians duplicated in place
    n_pruned: int  # gaussians removed (low opacity)
    n_after: int
    # The absolute screen-space gradient bar used this round. Pass it back in
    # as `grad_threshold` to freeze it; see densify_and_prune.
    grad_threshold: float = 0.0
    # (n_after,) int64: where each surviving gaussian came from in the old
    # model, or NEW_GAUSSIAN (-1) if it was just created. Feed this to
    # metalsplat.optim.migrate_optimizer_state -- rebuilding the optimizer
    # without it discards Adam's moments for every gaussian.
    source_index: torch.Tensor | None = None
    # (n_after,) int64: like source_index, but split children and clones point
    # at the gaussian they came *from* rather than -1. Adam state deliberately
    # does not follow that (new gaussians start cold, as in the reference), but
    # per-gaussian quantities derived from position -- the 3D filter radius --
    # can be carried from the parent instead of recomputed from scratch.
    parent_index: torch.Tensor | None = None


def densify_and_prune(
    model: GaussianModel,
    grad_accum: torch.Tensor,  # (N,) accumulated means2d-grad norms since last call
    grad_count: torch.Tensor,  # (N,) number of steps each gaussian was visible
    scene_scale: float,
    grad_percentile: float = 0.8,
    grad_threshold: float | None = None,
    prune_opacity_thresh: float = 0.005,
    split_scale_factor: float = 1.6,
    max_points: int | None = None,
) -> tuple[GaussianModel, DensifyStats]:
    """Splits, clones and prunes, returning the new model and stats.

    `grad_threshold`, if given, is an *absolute* bar on the average
    screen-space gradient: a gaussian is a split/clone candidate only if it
    exceeds it. This is what the reference implementation does, and it
    matters because it self-limits -- as gaussians fit their region their
    gradients fall below the bar and densification stops on its own.

    With `grad_threshold=None` the bar is the `grad_percentile` quantile of
    this round's gradients instead, which does *not* self-limit: it
    promotes a fixed fraction every round no matter how well-fit the model
    is, so the count grows geometrically until it hits `max_points` and
    then sits there with a large fraction of the model perpetually new.
    That is fine when the cap is the real constraint and disastrous when it
    is not, so the returned `stats.grad_threshold` lets a caller calibrate
    on the first round and freeze it thereafter.
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
    scales = model.scales.detach()
    quats = model.quats.detach()
    opacities = model.opacities.detach()
    # Whatever per-gaussian "color-like" tensor the model uses -- (N, 3) for
    # plain RGB, (N, 9, 3) raw SH coefficients for sh_degree=2 (using the
    # raw coefficients here, not colors_from_view(...), so a densify round
    # doesn't need a camera and preserves learned view-dependent detail
    # instead of collapsing it back to a flat DC-only color).
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
        std = scales[split_idx]  # (K, 3)
        samples = torch.randn(2, split_idx.numel(), 3, device=device) * std
        offsets = torch.einsum("kij,skj->ski", rotmat, samples)  # (2, K, 3)
        split_means = (means[split_idx].unsqueeze(0) + offsets).reshape(-1, 3)
        split_scales = (
            (scales[split_idx] / split_scale_factor)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .reshape(-1, 3)
        )
        split_quats = quats[split_idx].unsqueeze(0).expand(2, -1, -1).reshape(-1, 4)
        split_opacities = opacities[split_idx].unsqueeze(0).expand(2, -1).reshape(-1)
        split_color_like = _expand2(color_like[split_idx])
    else:
        split_means, split_scales, split_quats = _empty(3), _empty(3), _empty(4)
        split_opacities, split_color_like = _empty(), _empty(*color_shape)

    clone_means = means[clone_idx]
    clone_scales = scales[clone_idx]
    clone_quats = quats[clone_idx]
    clone_opacities = opacities[clone_idx]
    clone_color_like = color_like[clone_idx]

    # Split removes the original (replaced by 2 new); clone keeps the
    # original as well as adding a duplicate; prune removes low-opacity
    # gaussians regardless of candidacy.
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
        new_model = GaussianModel(
            final_means,
            scales=final_scales,
            quats=final_quats,
            opacities=final_opacities,
            colors=final_color_like,
        ).to(device)
    else:
        new_model = GaussianModel(
            final_means,
            scales=final_scales,
            quats=final_quats,
            opacities=final_opacities,
            sh_degree=model.sh_degree,
            sh_coeffs=final_color_like,
            active_sh_degree=model.active_sh_degree,
        ).to(device)

    # Split children and clones start with zeroed Adam state, and a split
    # parent's state dies with it -- matching the reference implementation,
    # which appends zeros for every gaussian it creates.
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    fresh = torch.full(
        (2 * split_idx.numel() + clone_idx.numel(),),
        NEW_GAUSSIAN,
        dtype=torch.int64,
        device=device,
    )
    source_index = torch.cat([keep_idx, fresh])
    # Split children and clones sit essentially where their parent did.
    parent_index = torch.cat([keep_idx, split_idx, split_idx, clone_idx])

    n_after = final_means.shape[0]
    n_pruned = int(
        (~keep_mask & ~is_large).sum().item()
    )  # low-opacity prunes among non-split gaussians
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


@torch.no_grad()
def reset_opacity(model: GaussianModel | Gaussian2DModel, value: float = 0.01) -> None:
    """Caps every gaussian's opacity at `value`, in place. Standard 3DGS
    trick: periodically forces all gaussians back to near-transparent, so
    ones that only got high opacity by occluding/compensating for a
    neighbor (rather than genuinely representing something) have to
    re-earn it through training or fall below the prune threshold and get
    removed by the next prune_low_opacity() call. Modifies
    `model.raw_opacities.data` in place (same Parameter object, same
    Adam momentum buffers) rather than rebuilding the model, since no
    gaussian is added or removed.

    Works for both `GaussianModel` and `Gaussian2DModel` unchanged -- it
    only touches `.opacities`/`.raw_opacities`, never means/scales/quats,
    so nothing here is 3D-vs-2D-specific. `metalsplat.densify2dgs` reuses
    this directly rather than duplicating it.
    """
    new_opacities = torch.clamp(model.opacities, max=value)
    model.raw_opacities.data = logit(new_opacities)


def prune_low_opacity(
    model: GaussianModel, prune_opacity_thresh: float = 0.005
) -> tuple[GaussianModel, int, torch.Tensor]:
    """Removes gaussians with opacity below the threshold. Standalone (no
    split/clone candidate computation) so it can run on its own, more
    frequent schedule, and -- unlike densify_and_prune -- keep running
    after adaptive density control's split/clone window has ended (e.g.
    to clean up after reset_opacity(), which can otherwise leave a lot of
    now-useless near-transparent gaussians sitting around for the rest of
    training).

    Returns (model, n_pruned, source_index); pass source_index to
    metalsplat.optim.migrate_optimizer_state when rebuilding the optimizer,
    or Adam's moments for every surviving gaussian are lost too.
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
        new_model = GaussianModel(
            means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=model.colors.detach()[keep_mask],
        ).to(device)
    else:
        new_model = GaussianModel(
            means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_degree=model.sh_degree,
            sh_coeffs=model.raw_sh.detach()[keep_mask],
            active_sh_degree=model.active_sh_degree,
        ).to(device)

    return new_model, n_pruned, keep_mask.nonzero(as_tuple=True)[0]
