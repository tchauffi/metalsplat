"""torch.autograd.Function wrapping the Metal 2DGS projection kernels
(metalsplat.kernels.project_2dgs). Forward/backward math matches
metalsplat.reference.project_2dgs_ref exactly -- that module is validated
against this one in tests/test_project_2dgs.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel


class Project2DGSGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor,  # (N, 2), positive (s_u, s_v)
        quats: torch.Tensor,  # (N, 4), unit, (w, x, y, z)
        R_wc: torch.Tensor,  # (3, 3)
        t_wc: torch.Tensor,  # (3,)
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_width: int,
        img_height: int,
        near: float = 0.2,
        eps2d: float = 0.3,
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
        conics = torch.empty(n, 3, device=device, dtype=torch.float32)
        radii = torch.empty(n, device=device, dtype=torch.float32)
        valid = torch.empty(n, device=device, dtype=torch.float32)
        compensations = torch.empty(n, device=device, dtype=torch.float32)
        transform = torch.empty(n, 9, device=device, dtype=torch.float32)
        normal = torch.empty(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("project_2dgs")
            lib.project_2dgs_forward(
                means_c,
                scales_c,
                quats_c,
                rwc_flat,
                twc,
                float(fx),
                float(fy),
                float(cx),
                float(cy),
                float(img_width),
                float(img_height),
                float(near),
                float(eps2d),
                means2d,
                depths,
                conics,
                radii,
                valid,
                compensations,
                transform,
                normal,
                threads=n,
            )

        ctx.save_for_backward(means_c, scales_c, quats_c, rwc_flat, twc, valid)
        ctx.fx, ctx.fy, ctx.cx, ctx.cy = fx, fy, cx, cy
        ctx.img_width, ctx.img_height = img_width, img_height
        ctx.near, ctx.eps2d = near, eps2d
        ctx.n = n
        return means2d, depths, conics, radii, valid, compensations, transform, normal

    @staticmethod
    def backward(
        ctx,
        grad_means2d,
        grad_depths,
        grad_conics,
        grad_radii,
        grad_valid,
        grad_compensations,
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
                float(ctx.img_width),
                float(ctx.img_height),
                float(ctx.near),
                float(ctx.eps2d),
                valid,
                grad_means2d.contiguous(),
                grad_conics.contiguous(),
                grad_compensations.contiguous(),
                grad_transform.contiguous(),
                grad_normal.contiguous(),
                d_means,
                d_scales,
                d_quats,
                threads=n,
            )

        return (
            d_means,
            d_scales,
            d_quats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    eps2d: float = 0.3,
):
    """Project 2D gaussian splats to 2D screen space using the Metal kernels.

    Returns `(means2d, depths, conics, radii, valid, compensation,
    transform, normal)` -- see
    metalsplat.reference.project_2dgs_ref.Projection2DGSResult for field
    semantics.
    """
    return Project2DGSGaussians.apply(
        means,
        scales,
        quats,
        R_wc,
        t_wc,
        fx,
        fy,
        cx,
        cy,
        img_width,
        img_height,
        near,
        eps2d,
    )
