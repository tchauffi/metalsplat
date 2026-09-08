"""Mip-Splatting's 3D smoothing filter (Yu et al. 2024).

The 2D half of Mip-Splatting already lives in the projection kernel: the
eps2d dilation low-passes the *screen-space* gaussian, and the
compensation factor removes the energy that dilation adds. That handles
sampling on the image plane, but not the scene itself.

The problem it leaves is that a gaussian can be reconstructed with detail
finer than any training camera ever resolved. Nothing during training
penalises that -- every training view is too far away to see the
difference -- but move the camera closer than any training view and those
sub-sampling-rate gaussians appear as high-frequency speckle, because they
encode structure that was never actually observed.

The fix is to band-limit each gaussian to the finest sampling rate any
training camera saw it at. For a pinhole camera, that rate is f/z, so a
gaussian observed no closer than z_min has a smallest resolvable world
extent of about z_min/f. Dilating its 3D covariance by

    Sigma' = Sigma + r^2 I,   r = sampling_scale * z_min / f

removes everything below that scale.

This needs no kernel change. A covariance is R diag(s^2) R^T, and the
identity is rotation-invariant, so

    Sigma + r^2 I = R (diag(s^2) + r^2 I) R^T = R diag(s^2 + r^2) R^T

-- the dilated gaussian has the *same* rotation and per-axis scales
sqrt(s_i^2 + r^2). Dilation inflates the integral of the density the same
way the 2D filter does, so opacity is scaled by
sqrt(det(Sigma)/det(Sigma')) to compensate; see `apply_3d_filter`.
"""

from __future__ import annotations

import torch

from metalsplat.camera import Camera

DEFAULT_SAMPLING_SCALE = 0.2  # the paper's value


@torch.no_grad()
def compute_3d_filter(
    means: torch.Tensor,  # (N, 3) world-space gaussian centres
    cameras: list[Camera],
    sampling_scale: float = DEFAULT_SAMPLING_SCALE,
    near: float = 0.2,
    frustum_margin: float = 1.15,
    camera_chunk: int = 16,
) -> torch.Tensor:
    """Per-gaussian filter radius `r` in world units, (N,).

    For each gaussian, finds the smallest z/f over the training cameras
    that actually see it -- the finest world-space detail any of them could
    resolve there -- and scales it by `sampling_scale`.

    `frustum_margin` accepts gaussians slightly outside the image bounds,
    so one sitting just off the edge of every view is still band-limited by
    the views that nearly see it rather than being left unfiltered.

    Gaussians no camera sees get `r = 0`: there is no observation to derive
    a sampling rate from, so they are left alone rather than filtered by a
    number that would be invented.

    Cameras are processed `camera_chunk` at a time as batched matmuls. A
    camera-at-a-time loop allocates a dozen (N,) temporaries per camera,
    which at a few hundred thousand gaussians and a couple of hundred views
    costs hundreds of milliseconds -- enough to dominate a training step
    when the gaussian count grows. The chunk bounds peak memory at
    N x camera_chunk floats per intermediate.
    """
    n = means.shape[0]
    device = means.device
    if n == 0 or not cameras:
        return torch.zeros(n, device=device)

    best = torch.full((n,), float("inf"), device=device)

    for i in range(0, len(cameras), camera_chunk):
        chunk = [c.to(device) for c in cameras[i : i + camera_chunk]]
        # (C,3,3) rotations and (C,3) translations, stacked once per chunk.
        rot = torch.stack([c.R_wc for c in chunk])
        trans = torch.stack([c.t_wc for c in chunk])
        fx = torch.tensor([float(c.fx) for c in chunk], device=device)
        fy = torch.tensor([float(c.fy) for c in chunk], device=device)
        half_w = torch.tensor([0.5 * c.img_width for c in chunk], device=device)
        half_h = torch.tensor([0.5 * c.img_height for c in chunk], device=device)

        # Only the three camera-space components are needed, each (N, C).
        x = means @ rot[:, 0, :].T + trans[:, 0]
        y = means @ rot[:, 1, :].T + trans[:, 1]
        z = means @ rot[:, 2, :].T + trans[:, 2]

        z_safe = z.clamp_min(near)
        # Offsets from the principal point, so the bounds test needs no cx/cy.
        on_screen = ((fx * x / z_safe).abs() < frustum_margin * half_w) & (
            (fy * y / z_safe).abs() < frustum_margin * half_h
        )
        visible = (z > near) & on_screen

        # z / f is the world extent one pixel covers at that depth.
        focal = torch.maximum(fx, fy)
        extent = torch.where(visible, z_safe / focal, torch.full_like(z_safe, float("inf")))
        best = torch.minimum(best, extent.min(dim=1).values)

    unseen = torch.isinf(best)
    return torch.where(unseen, torch.zeros_like(best), sampling_scale * best)


def apply_3d_filter(
    scales: torch.Tensor,  # (N, 3) positive
    opacities: torch.Tensor,  # (N,)
    filter_3d: torch.Tensor,  # (N,) filter radius from compute_3d_filter
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dilates `scales` by the filter and compensates `opacities` for it.

    Returns (dilated_scales, compensated_opacities). The compensation is
    sqrt(det(Sigma) / det(Sigma')), which for a diagonalised covariance is
    just the ratio of the products of the per-axis scales -- without it,
    band-limiting a gaussian would also make it denser.
    """
    r2 = (filter_3d * filter_3d).unsqueeze(-1)
    dilated = (scales * scales + r2).sqrt()
    compensation = scales.prod(dim=-1) / dilated.prod(dim=-1).clamp_min(1e-12)
    return dilated, opacities * compensation


@torch.no_grad()
def carry_filter_3d(
    filter_3d: torch.Tensor | None,
    parent_index: torch.Tensor,
) -> torch.Tensor | None:
    """Reindexes a filter onto a model whose gaussian set just changed.

    A full `compute_3d_filter` is O(gaussians x cameras) and costs hundreds
    of milliseconds at a few hundred thousand gaussians -- far too much to
    pay after every densify, seed and prune, which together fire roughly
    once per 77 steps. But none of those operations invalidate the whole
    filter: a prune only removes gaussians, and split children and clones
    sit essentially where their parent did, so they inherit its radius.

    Gaussians with no parent (freshly seeded, `parent_index` -1) get 0,
    i.e. unfiltered, until the next full recompute picks them up.

    Positions do drift as training moves the means, so this is an
    approximation between periodic full refreshes, not a replacement for
    them.
    """
    if filter_3d is None:
        return None
    out = filter_3d.new_zeros(parent_index.shape[0])
    known = parent_index >= 0
    if bool(known.any()):
        out[known] = filter_3d[parent_index[known]]
    return out
