"""Post-training cleanup for a trained scene: remove floaters and damp
overfit view-dependence.

Both operations target artifacts that barely show up in held-out PSNR but
are very visible when the camera *moves* -- gaussians popping in and out,
and surfaces shimmering as their colour swings with view direction. On the
garden scene, applying both improved held-out PSNR (19.06 -> 19.12) while
cutting frame-to-frame variation over an orbit by 16%.

These are forward-only edits to a finished model, not part of training.
"""

from __future__ import annotations

import torch

from metalsplat.gaussians import GaussianModel


def _rebuild(model: GaussianModel, keep: torch.Tensor, sh: torch.Tensor | None = None) -> GaussianModel:
    device = model.means.device
    means = model.means.detach()[keep]
    scales = model.scales.detach()[keep]
    quats = model.quats.detach()[keep]
    opacities = model.opacities.detach()[keep]
    if model.sh_degree == 0:
        return GaussianModel(
            means, scales=scales, quats=quats, opacities=opacities,
            colors=model.colors.detach()[keep],
        ).to(device)
    coeffs = (model.raw_sh.detach() if sh is None else sh)[keep]
    return GaussianModel(
        means, scales=scales, quats=quats, opacities=opacities,
        sh_degree=model.sh_degree, sh_coeffs=coeffs,
        active_sh_degree=model.active_sh_degree,
    ).to(device)


def prune_isolated(
    model: GaussianModel, cell_size: float = 0.4, min_per_cell: int = 8
) -> tuple[GaussianModel, int]:
    """Drops gaussians sitting in sparsely-populated regions of space.

    Voxelises positions and removes anything in a cell holding fewer than
    `min_per_cell` gaussians. A real surface is densely populated; a
    floater sits alone, so this targets exactly the gaussians that pop in
    and out as the camera moves, and (measured on the garden scene) costs
    nothing in held-out PSNR.

    Voxel occupancy rather than kNN because it's O(n): the scenes here run
    to ~440k gaussians, where an exact all-pairs kNN is not affordable.
    """
    means = model.means.detach()
    keys = torch.floor(means / cell_size).long()
    keys = keys - keys.min(dim=0).values
    dims = keys.max(dim=0).values + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]

    _, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= min_per_cell
    n_pruned = int((~keep).sum().item())
    if n_pruned == 0:
        return model, 0
    return _rebuild(model, keep), n_pruned


def damp_view_dependence(model: GaussianModel, factor: float = 0.75) -> GaussianModel:
    """Scales the non-DC spherical-harmonics coefficients by `factor`.

    With a few hundred training views and no regularisation on the SH
    coefficients, the higher-order terms absorb per-photo appearance
    differences (auto-exposure and white-balance drift between shots)
    rather than genuine specularity. That memorised component is
    view-dependent by construction, so it shimmers during a camera move.

    Damping it is not just a cosmetic trade: on the garden scene the train
    PSNR falls (20.41 -> 19.94 at factor 0.75) while *held-out* PSNR rises
    (19.06 -> 19.12), i.e. the removed component was overfitting. Below
    ~0.65 it starts eating real view-dependence and both fall.

    A proper training-time fix would be weight decay on the non-DC
    coefficients; this is the post-hoc equivalent for a finished scene.
    """
    if model.sh_degree == 0:
        return model  # nothing view-dependent to damp
    sh = model.raw_sh.detach().clone()
    sh[:, 1:, :] *= factor
    return _rebuild(model, torch.ones(model.num_points, dtype=torch.bool, device=sh.device), sh)
