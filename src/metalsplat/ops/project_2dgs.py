"""torch.autograd.Function wrapping the Metal 2DGS projection kernels
(metalsplat.kernels.project_2dgs). Forward/backward math matches
metalsplat.reference.project_2dgs_ref exactly -- that module is validated
against this one in tests/test_project_2dgs.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.ops._validate import check_float32, check_shape
from metalsplat.reference.rasterize_2dgs_ref import DEFAULT_FILTER_SIZE
from metalsplat.reference.tiling_ref import MAX_SIGMA_EXTENT, sigma_extent


class Project2DGSGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor,  # (N, 2), positive (s_u, s_v)
        quats: torch.Tensor,  # (N, 4), unit, (w, x, y, z)
        cutoffs: torch.Tensor,  # (N,) binning alpha cutoff in sigmas, not differentiated
        R_wc: torch.Tensor,  # (3, 3)
        t_wc: torch.Tensor,  # (3,)
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_width: int,
        img_height: int,
        near: float,
        filter_size: float,
    ):
        n = means.shape[0]
        device = means.device
        means_c = means.contiguous()
        scales_c = scales.contiguous()
        quats_c = quats.contiguous()
        rwc_flat = R_wc.to(device=device, dtype=torch.float32).contiguous().reshape(-1)
        twc = t_wc.to(device=device, dtype=torch.float32).contiguous()

        means2d = torch.empty(n, 2, device=device, dtype=torch.float32)
        depths = torch.empty(n, device=device, dtype=torch.float32)
        rects = torch.empty(n, 4, device=device, dtype=torch.float32)
        valid = torch.empty(n, device=device, dtype=torch.float32)
        transform = torch.empty(n, 9, device=device, dtype=torch.float32)
        normal = torch.empty(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("project_2dgs")
            lib.project_2dgs_forward(
                means_c,
                scales_c,
                quats_c,
                cutoffs.contiguous(),
                rwc_flat,
                twc,
                float(fx),
                float(fy),
                float(cx),
                float(cy),
                float(img_width),
                float(img_height),
                float(near),
                float(filter_size),
                means2d,
                depths,
                rects,
                valid,
                transform,
                normal,
                threads=n,
            )

        ctx.save_for_backward(means_c, scales_c, quats_c, rwc_flat, twc, valid)
        ctx.fx, ctx.fy, ctx.cx, ctx.cy = fx, fy, cx, cy
        ctx.near = near
        ctx.n = n
        # Structural: tile-binning rectangles and a 0/1 flag.
        ctx.mark_non_differentiable(rects, valid)
        return means2d, depths, rects, valid, transform, normal

    @staticmethod
    def backward(
        ctx,
        grad_means2d,
        grad_depths,
        grad_rects,
        grad_valid,
        grad_transform,
        grad_normal,
    ):
        means, scales, quats, rwc_flat, twc, valid = ctx.saved_tensors
        n = ctx.n
        device = means.device

        d_means = torch.zeros(n, 3, device=device, dtype=torch.float32)
        d_scales = torch.zeros(n, 2, device=device, dtype=torch.float32)
        d_quats = torch.zeros(n, 4, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("project_2dgs")
            lib.project_2dgs_backward(
                means,
                scales,
                quats,
                rwc_flat,
                twc,
                float(ctx.fx),
                float(ctx.fy),
                float(ctx.cx),
                float(ctx.cy),
                float(ctx.near),
                valid,
                grad_means2d.contiguous(),
                grad_depths.contiguous(),
                grad_transform.contiguous(),
                grad_normal.contiguous(),
                d_means,
                d_scales,
                d_quats,
                threads=n,
            )

        return (d_means, d_scales, d_quats) + (None,) * 11


def project_gaussians_2dgs(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    R_wc: torch.Tensor,
    t_wc: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_width: int,
    img_height: int,
    near: float = 0.2,
    filter_size: float = DEFAULT_FILTER_SIZE,
    opacities: torch.Tensor | None = None,
):
    """Project 2D gaussian splats to 2D screen space using the Metal kernels.

    Returns `(means2d, depths, rects, valid, transform, normal)` -- see
    metalsplat.reference.project_2dgs_ref.Projection2DGSResult for field
    semantics. `opacities`, if given, tightens each binning rectangle to
    where that splat's alpha actually clears the rasterizer's cutoff;
    without it every rectangle is sized for full opacity. `filter_size`
    must be the one the rasterizer uses, since its screen-space filter
    widens every footprint.
    """
    device = means.device
    n = means.shape[0]
    check_float32("means", means, (n, 3), device)
    check_float32("scales", scales, (n, 2), device)
    check_float32("quats", quats, (n, 4), device)
    check_shape("R_wc", R_wc, (3, 3))
    check_shape("t_wc", t_wc, (3,))
    if opacities is None:
        cutoffs = torch.full((n,), MAX_SIGMA_EXTENT, device=device)
    else:
        check_shape("opacities", opacities, (n,))
        cutoffs = sigma_extent(opacities.detach().to(device), n, device)
    return Project2DGSGaussians.apply(
        means,
        scales,
        quats,
        cutoffs,
        R_wc,
        t_wc,
        fx,
        fy,
        cx,
        cy,
        img_width,
        img_height,
        near,
        filter_size,
    )
