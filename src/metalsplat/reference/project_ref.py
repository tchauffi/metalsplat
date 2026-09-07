"""Pure-PyTorch reference projection of 3D gaussians to 2D screen space.

This is the executable spec for the ``project`` pipeline stage: it is plain
differentiable torch ops, so ``torch.autograd`` gives a correct backward for
free. The Metal kernels in ``metalsplat.kernels.project`` reimplement this
same math for speed; this module is what their forward *and* backward
outputs get numerically checked against, and it doubles as a CPU-compatible
fallback.

Conventions (matching the original 3D Gaussian Splatting paper / gsplat):
- World-space covariance: ``Sigma = R(quat) @ diag(scale)^2 @ R(quat)^T``.
- Camera-space transform: ``x_cam = x_world @ R_wc^T + t_wc`` (``R_wc``/``t_wc``
  is the world-to-camera rotation/translation, i.e. standard extrinsics).
- 2D covariance via the EWA/affine approximation: ``Sigma2d = J @ Sigma_cam @ J^T``
  where ``J`` is the Jacobian of the pinhole projection at the gaussian's
  camera-space mean, and ``Sigma_cam = R_wc @ Sigma @ R_wc^T``.
- ``conic`` stores the inverse of the (eps-regularized) 2D covariance as its
  three independent entries ``(a, b, c)`` for the symmetric matrix
  ``[[a, b], [b, c]]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metalsplat.utils.quaternion import quat_to_rotmat


@dataclass
class ProjectionResult:
    means2d: torch.Tensor  # (N, 2) pixel-space centers
    depths: torch.Tensor  # (N,) camera-space z
    conics: torch.Tensor  # (N, 3) inverse-2D-covariance entries (a, b, c)
    radii: torch.Tensor  # (N,) integer pixel radius (3-sigma extent), 0 if culled
    valid: torch.Tensor  # (N,) bool, False for culled gaussians


def project_gaussians(
    means: torch.Tensor,  # (N, 3) world space
    scales: torch.Tensor,  # (N, 3) positive
    quats: torch.Tensor,  # (N, 4) unit quaternions (w, x, y, z)
    R_wc: torch.Tensor,  # (3, 3) world-to-camera rotation
    t_wc: torch.Tensor,  # (3,) world-to-camera translation
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_width: int,
    img_height: int,
    near: float = 0.2,
    eps2d: float = 0.3,
) -> ProjectionResult:
    means_cam = means @ R_wc.T + t_wc  # (N, 3)
    x, y, z = means_cam.unbind(-1)
    z_safe = z.clamp_min(near)

    rotmat = quat_to_rotmat(quats)  # (N, 3, 3)
    scale_mat = torch.diag_embed(scales)  # (N, 3, 3)
    m = rotmat @ scale_mat  # (N, 3, 3), so Sigma = m @ m^T
    sigma_world = m @ m.transpose(-1, -2)  # (N, 3, 3)
    sigma_cam = R_wc @ sigma_world @ R_wc.T  # (N, 3, 3), broadcasts R_wc over N

    zeros = torch.zeros_like(x)
    j_row0 = torch.stack([fx / z_safe, zeros, -fx * x / (z_safe * z_safe)], dim=-1)
    j_row1 = torch.stack([zeros, fy / z_safe, -fy * y / (z_safe * z_safe)], dim=-1)
    jac = torch.stack([j_row0, j_row1], dim=-2)  # (N, 2, 3)

    sigma2d = jac @ sigma_cam @ jac.transpose(-1, -2)  # (N, 2, 2)
    a = sigma2d[:, 0, 0] + eps2d
    b = sigma2d[:, 0, 1]
    c = sigma2d[:, 1, 1] + eps2d

    det = a * c - b * b
    det_safe = det.clamp_min(1e-12)
    conic_a = c / det_safe
    conic_b = -b / det_safe
    conic_c = a / det_safe
    conics = torch.stack([conic_a, conic_b, conic_c], dim=-1)

    mid = 0.5 * (a + c)
    disc = (mid * mid - det).clamp_min(0.0)
    lambda_max = mid + disc.sqrt()
    radii = torch.ceil(3.0 * lambda_max.clamp_min(0.0).sqrt())

    means2d_x = fx * x / z_safe + cx
    means2d_y = fy * y / z_safe + cy
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)

    in_front = z > near
    positive_det = det > 0
    nonzero_radius = radii > 0
    in_bounds = (
        (means2d_x + radii >= 0)
        & (means2d_x - radii < img_width)
        & (means2d_y + radii >= 0)
        & (means2d_y - radii < img_height)
    )
    valid = in_front & positive_det & nonzero_radius & in_bounds

    radii = torch.where(valid, radii, torch.zeros_like(radii))

    return ProjectionResult(
        means2d=means2d,
        depths=z,
        conics=conics,
        radii=radii,
        valid=valid,
    )
