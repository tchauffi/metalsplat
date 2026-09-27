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

import math
from dataclasses import dataclass

import torch

from metalsplat.gaussians import GaussianModel, logit
from metalsplat.optim import NEW_GAUSSIAN
from metalsplat.splat_model import SplatModel
from metalsplat.utils.quaternion import quat_to_rotmat


@dataclass
class DensifyStats:
    n_before: int
    n_split: int  # gaussians replaced by 2 new ones each
    n_cloned: int  # gaussians duplicated in place
    n_pruned: int  # rows removed after split/clone (low opacity or too large)
    n_after: int
    # The absolute screen-space gradient bar used this round, or None if the
    # round did nothing (no visible gaussians, or already at `max_points`)
    # and so never computed one. Pass it back in as `grad_threshold` to
    # freeze it; see densify_and_prune. The None matters: a caller that
    # freezes the *first* round's bar has to be able to tell a real
    # calibration from a no-op, or it would freeze a placeholder and select
    # every visible gaussian forever after.
    grad_threshold: float | None = None
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


def select_candidates(
    avg_grad: torch.Tensor,  # (N,) average densification signal
    visible: torch.Tensor,  # (N,) bool, gaussians with a meaningful avg_grad
    grad_percentile: float,
    grad_threshold: float | None,
    max_points: int | None,
) -> tuple[torch.Tensor, float | None]:
    """Split/clone candidates for one densify round, and the bar used.

    Returns `(candidates, threshold)`. When nothing is visible, or the model
    is already at `max_points`, no gaussian is selected and no bar is
    computed, so `threshold` is just `grad_threshold` passed back (None if
    the caller has not frozen one yet -- see DensifyStats.grad_threshold).

    Every candidate adds exactly one gaussian net (a split replaces one with
    two, a clone adds one), so under `max_points` the candidates are cut to
    the `max_points - N` highest-gradient ones: the cap holds exactly rather
    than being overshot by a whole round's worth of candidates.
    """
    n = avg_grad.shape[0]
    none = torch.zeros(n, dtype=torch.bool, device=avg_grad.device)
    budget = None if max_points is None else max_points - n
    if not bool(visible.any()) or (budget is not None and budget <= 0):
        return none, grad_threshold

    if grad_threshold is None:
        threshold = float(torch.quantile(avg_grad[visible], grad_percentile))
    else:
        threshold = float(grad_threshold)
    candidates = visible & (avg_grad >= threshold)

    if budget is not None:
        idx = candidates.nonzero(as_tuple=True)[0]
        if idx.numel() > budget:
            candidates = none.clone()
            candidates[idx[avg_grad[idx].topk(budget).indices]] = True
    return candidates, threshold


def densify_and_prune(
    model: GaussianModel,
    grad_accum: torch.Tensor,  # (N,) accumulated means2d-grad norms since last call
    grad_count: torch.Tensor,  # (N,) number of steps each gaussian was visible
    scene_scale: float,  # split/clone boundary: largest scale above it -> split
    grad_percentile: float = 0.8,
    grad_threshold: float | None = None,
    prune_opacity_thresh: float = 0.005,
    split_scale_factor: float = 1.6,
    max_points: int | None = None,
    max_radii2d: torch.Tensor | None = None,
    max_screen_size: float | None = None,
    max_world_size: float | None = None,
) -> tuple[GaussianModel, DensifyStats]:
    """Splits, clones and prunes, returning the new model and stats.

    A candidate whose largest scale exceeds `scene_scale` is split into two
    smaller children; any other candidate is cloned. The reference sets this
    boundary to `percent_dense` (0.01) x the camera extent.

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

    `max_points` is a hard cap on split/clone growth: a round keeps only as
    many of the highest-gradient candidates as fit under it, and selects
    none at the cap. Pruning still runs either way.

    Pruning follows the reference's order: split and clone first, then
    prune the *result*, so a large gaussian with a high gradient is split
    rather than dropped, and a clone or split child is pruned by the same
    rules as everything else. A row is pruned if its opacity is at most
    `prune_opacity_thresh`, or its largest world-space scale exceeds
    `max_world_size` (the reference uses 0.1 x the camera extent), or --
    for surviving originals only -- its largest projected radius since the
    last round (`max_radii2d`, pixels) exceeds `max_screen_size`. Each size
    check is skipped when its threshold is None.

    The screen-size check exists for completeness but is best left off:
    the reference passes 20px, but its densification_postfix zeroes
    max_radii2D before the check runs, so there it never fires. Actually
    enforcing it deletes legitimately large foreground gaussians near the
    cameras and leaves holes.
    """
    device = model.means.device
    n = model.num_points

    visible = grad_count > 0
    avg_grad = torch.zeros(n, device=device)
    avg_grad[visible] = grad_accum[visible] / grad_count[visible]
    candidates, threshold = select_candidates(
        avg_grad, visible, grad_percentile, grad_threshold, max_points
    )

    scales = model.scales.detach()
    is_large = candidates & (scales.max(dim=-1).values > scene_scale)
    split_idx = is_large.nonzero(as_tuple=True)[0]
    clone_idx = (candidates & ~is_large).nonzero(as_tuple=True)[0]

    # Two children per split parent, offset by a sample from the parent's
    # own 3D gaussian, as in the reference.
    rotmat = quat_to_rotmat(model.quats.detach()[split_idx])  # (K, 3, 3)
    samples = torch.randn(2, split_idx.numel(), 3, device=device) * scales[split_idx]
    split_offsets = torch.einsum("kij,skj->ski", rotmat, samples)  # (2, K, 3)

    screen_big = None
    if max_screen_size is not None and max_radii2d is not None:

        def screen_big(unsplit_idx):
            return max_radii2d[unsplit_idx] > max_screen_size

    return split_clone_prune(
        model,
        split_idx,
        clone_idx,
        split_offsets,
        split_scale_factor=split_scale_factor,
        prune_opacity_thresh=prune_opacity_thresh,
        max_world_size=max_world_size,
        threshold=threshold,
        screen_big=screen_big,
    )


def split_clone_prune(
    model: SplatModel,
    split_idx: torch.Tensor,  # (K,) gaussians to replace by two children each
    clone_idx: torch.Tensor,  # (C,) gaussians to duplicate in place
    split_offsets: torch.Tensor,  # (2, K, 3) world offset of each child from its parent
    split_scale_factor: float,
    prune_opacity_thresh: float,
    max_world_size: float | None,
    threshold: float | None,
    screen_big=None,  # (unsplit_idx) -> (len,) bool, prune surviving originals
) -> tuple[SplatModel, DensifyStats]:
    """The split/clone/prune step both densify functions share, once they
    have chosen which gaussians to split or clone and where the children go.

    Every row is copied from the raw parameters (`SplatModel.raw_rows`), so
    the gaussians that are merely carried over come out bit-identical, and a
    split child is its parent with the offset added to its mean and
    `log(split_scale_factor)` taken off its raw scales. Rebuilding through
    the activations instead, as this used to, capped every raw opacity at
    +-9.2 (the logit clamp) and renormalized every quaternion under Adam
    moments accumulated for its unnormalized value -- on every prune round,
    for every gaussian.
    """
    device = model.means.device
    n = model.num_points
    raw = model.raw_rows()

    unsplit = torch.ones(n, dtype=torch.bool, device=device)
    unsplit[split_idx] = False
    unsplit_idx = unsplit.nonzero(as_tuple=True)[0]

    children = {k: v[split_idx.repeat(2)] for k, v in raw.items()}
    children["means"] = children["means"] + split_offsets.reshape(-1, 3)
    children["raw_scales"] = children["raw_scales"] - math.log(split_scale_factor)
    # Split removes the original (replaced by 2 new); clone keeps the
    # original as well as adding a duplicate.
    rows = {
        k: torch.cat([v[unsplit_idx], children[k], v[clone_idx]])
        for k, v in raw.items()
    }
    n_new = 2 * split_idx.numel() + clone_idx.numel()
    # Split children and clones start with zeroed Adam state, and a split
    # parent's state dies with it -- matching the reference implementation,
    # which appends zeros for every gaussian it creates.
    rows_source = torch.cat(
        [
            unsplit_idx,
            torch.full((n_new,), NEW_GAUSSIAN, dtype=torch.int64, device=device),
        ]
    )
    # Split children and clones sit essentially where their parent did.
    rows_parent = torch.cat([unsplit_idx, split_idx, split_idx, clone_idx])

    # Prune the densified set, as the reference does (see densify_and_prune).
    prune = torch.sigmoid(rows["raw_opacities"]) <= prune_opacity_thresh
    if max_world_size is not None:
        prune |= rows["raw_scales"].exp().max(dim=-1).values > max_world_size
    if screen_big is not None:
        # New rows have no screen history yet, as in the reference.
        prune[: unsplit_idx.numel()] |= screen_big(unsplit_idx)
    keep = ~prune
    if n_new == 0 and not bool(prune.any()):
        unchanged = torch.arange(n, device=device)
        return model, DensifyStats(n, 0, 0, 0, n, threshold, unchanged, unchanged)

    new_model = model.with_rows({k: v[keep] for k, v in rows.items()})
    stats = DensifyStats(
        n_before=n,
        n_split=int(split_idx.numel()),
        n_cloned=int(clone_idx.numel()),
        n_pruned=int(prune.sum().item()),  # rows removed after densification
        n_after=new_model.num_points,
        grad_threshold=threshold,
        source_index=rows_source[keep],
        parent_index=rows_parent[keep],
    )
    return new_model, stats


@torch.no_grad()
def reset_opacity(
    model: SplatModel,
    value: float = 0.01,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Caps every gaussian's opacity at `value`, in place. Standard 3DGS
    trick: periodically forces all gaussians back to near-transparent, so
    ones that only got high opacity by occluding/compensating for a
    neighbor (rather than genuinely representing something) have to
    re-earn it through training or fall below the prune threshold and get
    removed by the next prune_low_opacity() call. Modifies
    `model.raw_opacities.data` in place (same Parameter object) rather than
    rebuilding the model, since no gaussian is added or removed.

    Pass the training `optimizer` so the opacity's Adam moments are zeroed
    too, keeping the step count -- exactly what the reference's
    `replace_tensor_to_optimizer` does. It matters a great deal: the
    gradient on raw opacity scales with the sigmoid's slope, which is ~10x
    smaller at 0.01 than at typical pre-reset opacities, so a second moment
    left over from before the reset shrinks every post-reset step to a
    fraction of the learning rate. Opacities then cannot climb back before
    the next densify round culls them -- measured on the 2DGS garden run,
    one reset with stale moments pruned 77% of the model (249k -> 58k) at
    the following round. Zeroed moments give the reference's large first
    steps instead.

    Works for both `GaussianModel` and `Gaussian2DModel` unchanged -- it
    only touches `.opacities`/`.raw_opacities`, never means/scales/quats,
    so nothing here is 3D-vs-2D-specific.
    """
    new_opacities = torch.clamp(model.opacities, max=value)
    model.raw_opacities.data = logit(new_opacities)
    if optimizer is not None:
        state = optimizer.state.get(model.raw_opacities)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()


def prune_low_opacity(
    model: SplatModel, prune_opacity_thresh: float = 0.005
) -> tuple[SplatModel, int, torch.Tensor]:
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
    keep_mask = model.opacities.detach() > prune_opacity_thresh
    n_pruned = int((~keep_mask).sum().item())
    if n_pruned == 0:
        return model, 0, torch.arange(model.num_points, device=device)
    return model.select(keep_mask), n_pruned, keep_mask.nonzero(as_tuple=True)[0]
