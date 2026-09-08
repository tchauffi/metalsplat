"""Loss-driven gaussian seeding: finds image regions with high training
residual but ~no existing gaussian coverage, and seeds brand-new gaussians
there by unprojecting the pixel with a plausible borrowed depth.

Why this exists: adaptive density control (metalsplat.densify) only
splits/clones *existing* gaussians, so it can't add coverage to a region
that starts with ~zero gaussians in it -- e.g. sky or a distant background
visible in only some camera views, where COLMAP's sparse point cloud has
few or no points (SfM can't triangulate features on sky at all, and gets
very few on low-parallax distant surfaces). Those regions stay
under-reconstructed (near-black against a zero background) no matter how
long training runs, and look inconsistent from frame to frame depending on
how much of that region each camera sees. This module actively places new
gaussians into exactly those gaps.

Training-loop logic, not part of the core differentiable pipeline (no
Metal kernels -- it operates between training steps, not inside the
forward/backward render, same as metalsplat.densify).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metalsplat.camera import Camera
from metalsplat.gaussians import GaussianModel
from metalsplat.reference.sh_ref import SH_C0


@dataclass
class SeedStats:
    n_before: int
    n_seeded: int
    n_after: int


def seed_uncovered_regions(
    model: GaussianModel,
    camera: Camera,
    pred: torch.Tensor,  # (H, W, 3) rendered image this step
    target: torch.Tensor,  # (H, W, 3) ground truth
    final_T: torch.Tensor,  # (H, W) per-pixel transmittance from render(..., return_aux=True)
    init_scale: float,
    residual_thresh: float = 0.15,
    coverage_thresh: float = 0.8,  # final_T above this counts as "uncovered"
    max_seeds_per_call: int = 300,
    fallback_depth_percentile: float = 0.8,
    near: float = 0.2,
    max_points: int | None = None,
) -> tuple[GaussianModel, SeedStats]:
    device = model.means.device
    n_before = model.num_points
    if max_points is not None and n_before >= max_points:
        return model, SeedStats(n_before, 0, n_before)

    residual = (pred - target).abs().mean(dim=-1)  # (H, W)
    uncovered = final_T > coverage_thresh
    needs_seed = uncovered & (residual > residual_thresh)

    ys, xs = needs_seed.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return model, SeedStats(n_before, 0, n_before)
    if ys.numel() > max_seeds_per_call:
        perm = torch.randperm(ys.numel(), device=device)[:max_seeds_per_call]
        ys, xs = ys[perm], xs[perm]
    if max_points is not None:
        ys, xs = ys[: max_points - n_before], xs[: max_points - n_before]
    k = ys.numel()
    if k == 0:
        return model, SeedStats(n_before, 0, n_before)

    # A purely uncovered pixel (e.g. true sky) has no real depth to
    # estimate from, so fall back to a representative "far" depth: a high
    # percentile of this camera's *currently visible* gaussian depths.
    with torch.no_grad():
        means_cam = model.means.detach() @ camera.R_wc.T + camera.t_wc
        depths_cam = means_cam[:, 2]
        in_front = depths_cam > near
        fallback_depth = (
            torch.quantile(depths_cam[in_front], fallback_depth_percentile).item()
            if bool(in_front.any())
            else 10.0 * init_scale
        )

    pixel_x = xs.float() + 0.5
    pixel_y = ys.float() + 0.5
    x_cam = (pixel_x - camera.cx) * fallback_depth / camera.fx
    y_cam = (pixel_y - camera.cy) * fallback_depth / camera.fy
    z_cam = torch.full_like(x_cam, fallback_depth)
    cam_pts = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (K, 3)

    # Inverse of `cam = R_wc @ world + t_wc`: world = R_wc^T @ (cam - t_wc)
    # = R_wc^T @ cam + camera.position (row-vector form: cam_pts @ R_wc).
    new_means = cam_pts @ camera.R_wc + camera.position

    new_scales = torch.full((k, 3), init_scale, device=device)
    new_quats = torch.zeros(k, 4, device=device)
    new_quats[:, 0] = 1.0
    new_opacities = torch.full((k,), 0.1, device=device)
    new_colors = target[ys, xs]  # bootstrap color directly from ground truth

    means = model.means.detach()
    scales = model.scales.detach()
    quats = model.quats.detach()
    opacities = model.opacities.detach()
    color_like = (model.colors if model.sh_degree == 0 else model.raw_sh).detach()

    final_means = torch.cat([means, new_means], dim=0)
    final_scales = torch.cat([scales, new_scales], dim=0)
    final_quats = torch.cat([quats, new_quats], dim=0)
    final_opacities = torch.cat([opacities, new_opacities], dim=0)

    if model.sh_degree == 0:
        final_color_like = torch.cat([color_like, new_colors], dim=0)
        new_model = GaussianModel(
            final_means, scales=final_scales, quats=final_quats,
            opacities=final_opacities, colors=final_color_like,
        ).to(device)
    else:
        # Match the model's own coefficient count, which depends on its degree.
        new_sh = torch.zeros(k, color_like.shape[1], 3, device=device)
        new_sh[:, 0, :] = (new_colors - 0.5) / SH_C0
        final_color_like = torch.cat([color_like, new_sh], dim=0)
        new_model = GaussianModel(
            final_means, scales=final_scales, quats=final_quats,
            opacities=final_opacities, sh_degree=model.sh_degree, sh_coeffs=final_color_like,
            active_sh_degree=model.active_sh_degree,
        ).to(device)

    n_after = final_means.shape[0]
    return new_model, SeedStats(n_before=n_before, n_seeded=k, n_after=n_after)
