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
    """
    n = means.shape[0]
    device = means.device
    best = torch.full((n,), float("inf"), device=device)

    for cam in cameras:
        cam = cam.to(device)
        p = means @ cam.R_wc.T + cam.t_wc
        z = p[:, 2]
        in_front = z > near
        z_safe = z.clamp_min(near)

        u = cam.fx * p[:, 0] / z_safe + cam.cx
        v = cam.fy * p[:, 1] / z_safe + cam.cy
        half_w = 0.5 * cam.img_width
        half_h = 0.5 * cam.img_height
        on_screen = ((u - cam.cx).abs() < frustum_margin * half_w) & (
            (v - cam.cy).abs() < frustum_margin * half_h
        )

        # z / f is the world extent one pixel covers at that depth.
        focal = max(float(cam.fx), float(cam.fy))
        extent = z_safe / focal
        best = torch.where(in_front & on_screen, torch.minimum(best, extent), best)

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
