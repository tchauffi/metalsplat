"""The standard 3D Gaussian Splatting training loss: L = (1-lambda)*L1 +
lambda*D-SSIM, D-SSIM = (1 - SSIM)/2 (Kerbl et al. 2023 use lambda=0.2).

SSIM is a local-window statistic computable via conv2d, which already runs
fine on MPS -- no custom Metal kernel needed here (same reasoning as
metalsplat.ops.tiling using plain torch ops rather than a kernel).

Also 2DGS's two regularizers (`distortion_loss`, `normal_consistency_loss`)
-- plain differentiable torch ops on `render_2dgs`'s aux outputs, no custom
kernel needed for these either (the expensive part, the distortion map's
own O(N) accumulation, already happened inside `rasterize_2dgs.metal`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from metalsplat.camera import Camera

DEFAULT_LAMBDA_DSSIM = 0.2


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device, dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] @ g1d[None, :]
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """pred, target: (H, W, C) in [0, 1]. Returns scalar mean SSIM over the image."""
    pred_c = pred.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    target_c = target.permute(2, 0, 1).unsqueeze(0)
    channels = pred_c.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_pred = F.conv2d(pred_c, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target_c, window, padding=pad, groups=channels)
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = (
        F.conv2d(pred_c * pred_c, window, padding=pad, groups=channels) - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target_c * target_c, window, padding=pad, groups=channels)
        - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred_c * target_c, window, padding=pad, groups=channels)
        - mu_pred_target
    )

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


def gaussian_splatting_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_dssim: float = DEFAULT_LAMBDA_DSSIM
) -> torch.Tensor:
    """pred, target: (H, W, C) in [0, 1]. The standard 3DGS training loss."""
    l1 = (pred - target).abs().mean()
    d_ssim = (1.0 - ssim(pred, target)) / 2.0
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * d_ssim


def distortion_loss(distortion_map: torch.Tensor) -> torch.Tensor:
    """`distortion_map`: (H, W), `render_2dgs(..., return_aux=True).distortion`
    (Mip-NeRF-360/2DGS's "concentrate the weight along the ray"
    regularizer -- see `rasterize_2dgs_ref`'s module docstring for its
    exact definition). The expensive part already happened in the
    rasterizer; this is mostly the scalar reduction.

    Clamped at 0 per pixel first: the map is defined on the compositing
    (mean-depth-sorted) order rather than each pixel's actual ray-splat
    intersection order, so a pixel whose contributing gaussians happen to
    be out of order in `z_hit` can come out slightly negative -- an
    approximation artifact, not a real "negative distortion". Averaging
    those in unclamped would let the optimizer *reward* creating more such
    artifacts instead of only ever penalizing spread-out depth.
    """
    return distortion_map.clamp_min(0.0).mean()


def normal_consistency_loss(
    rendered_normal: torch.Tensor,  # (H, W, 3), world-space, from render_2dgs
    rendered_depth: torch.Tensor,  # (H, W), camera-space z, from render_2dgs
    camera: Camera,
) -> torch.Tensor:
    """2DGS's normal-consistency regularizer: compares the alpha-composited
    surfel normal against a "pseudo-normal" derived from the local shape of
    the rendered depth map (image-space finite differences of unprojected
    3D points) -- teaches depth and normals to agree with each other,
    which is what makes the reconstructed surface usable for meshing.

    The pseudo-normal is `.detach()`ed -- treated as a fixed target for
    this loss term (the common, more stable choice in reimplementations)
    rather than differentiated through; gradient still reaches
    `rendered_depth` via the comparison, just not via the pseudo-normal's
    own construction.

    Only defined on interior pixels (finite differences need both
    neighbours), so this compares `rendered_normal[1:-1, 1:-1]` against
    the pseudo-normal.
    """
    h, w = rendered_depth.shape
    device, dtype = rendered_depth.device, rendered_depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + 0.5,
        torch.arange(w, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    x_cam = (xs - camera.cx) / camera.fx * rendered_depth
    y_cam = (ys - camera.cy) / camera.fy * rendered_depth
    points_cam = torch.stack([x_cam, y_cam, rendered_depth], dim=-1)  # (H, W, 3)
    # world = (cam - t_wc) @ R_wc, inverting cam = world @ R_wc.T + t_wc.
    points_world = (points_cam - camera.t_wc) @ camera.R_wc

    dx = points_world[1:-1, 2:, :] - points_world[1:-1, :-2, :]
    dy = points_world[2:, 1:-1, :] - points_world[:-2, 1:-1, :]
    pseudo_normal = F.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1).detach()

    normal_interior = rendered_normal[1:-1, 1:-1, :]
    # Align sign to the (already camera-facing) rendered normal rather than
    # hardcoding a convention -- the finite-difference cross product's
    # orientation depends on pixel-grid handedness, not scene geometry.
    dot = (pseudo_normal * normal_interior).sum(-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    pseudo_normal = pseudo_normal * sign

    return (1.0 - (normal_interior * pseudo_normal).sum(-1)).mean()
