"""torch.autograd.Function wrapping the Metal projection kernels
(metalsplat.kernels.project). Forward/backward math matches
metalsplat.reference.project_ref exactly -- that module is validated
against this one in tests/test_project.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel


class ProjectGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor,  # (N, 3), positive
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

        if n > 0:
            lib = load_kernel("project")
            lib.project_forward(
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
                threads=n,
            )

        ctx.save_for_backward(means_c, scales_c, quats_c, rwc_flat, twc, valid)
        ctx.fx, ctx.fy, ctx.cx, ctx.cy = fx, fy, cx, cy
        ctx.near, ctx.eps2d = near, eps2d
        ctx.n = n
        return means2d, depths, conics, radii, valid

    @staticmethod
    def backward(ctx, grad_means2d, grad_depths, grad_conics, grad_radii, grad_valid):
        means, scales, quats, rwc_flat, twc, valid = ctx.saved_tensors
        n = ctx.n
        device = means.device

        d_means = torch.zeros(n, 3, device=device, dtype=torch.float32)
        d_scales = torch.zeros(n, 3, device=device, dtype=torch.float32)
        d_quats = torch.zeros(n, 4, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("project")
            lib.project_backward(
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
                float(ctx.eps2d),
                valid,
                grad_means2d.contiguous(),
                grad_conics.contiguous(),
                d_means,
                d_scales,
                d_quats,
                threads=n,
            )

        return d_means, d_scales, d_quats, None, None, None, None, None, None, None, None, None, None


def project_gaussians(
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
    """Project 3D gaussians to 2D screen space using the Metal kernels.

    Returns (means2d, depths, conics, radii, valid) -- see
    metalsplat.reference.project_ref.ProjectionResult for field semantics.
    """
    return ProjectGaussians.apply(
        means, scales, quats, R_wc, t_wc, fx, fy, cx, cy, img_width, img_height, near, eps2d
    )
