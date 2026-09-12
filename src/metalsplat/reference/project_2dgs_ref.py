"""Pure-PyTorch reference projection of 2D gaussian splats ("surfels") to
2D screen space. Executable spec / CPU fallback for
``metalsplat.kernels.project_2dgs``, same role as ``project_ref`` for 3DGS.

This does two distinct jobs:

1. **Tile-culling bound.** Reuses the *exact* 3DGS EWA/conic approximation
   (``reference.project_ref.project_gaussians``) by treating the splat as a
   degenerate 3D ellipsoid whose missing depth-axis scale is a fixed small
   ``EPS_3RD_AXIS``. This produces ``means2d``/``depths``/``conics``/
   ``radii``/``valid``/``compensation`` in the exact layout 3DGS uses, so
   ``ops.tiling.bin_and_sort_gaussians``/``kernels/tiling.metal`` are reused
   completely unmodified for tile binning. Because that linear
   approximation can *underestimate* a flat disk's true screen footprint
   near edge-on views (where the exact ray-splat silhouette extends further
   than the first-order Taylor expansion predicts), ``RADIUS_SAFETY_MARGIN``
   inflates the resulting radius before it's used for culling.
2. **Exact per-gaussian data for the ray-splat intersection** the
   rasterizer performs per pixel (``rasterize_2dgs``): the 9 independent
   entries of ``M = W @ H`` and the camera-facing world-space normal.

   ``H`` is the local tangent-plane-to-world embedding: a 4x4 matrix whose
   columns are ``(s_u*t_u, s_v*t_v, 0, mean)`` (as homogeneous 4-vectors,
   last component 0 for the two direction columns and 1 for the point
   column) -- a local point ``(u, v, *, 1)`` maps to
   ``mean + u*s_u*t_u + v*s_v*t_v`` regardless of the ``*`` slot, since
   ``H``'s 3rd column is exactly zero.

   ``W`` is the camera's homogeneous projection matching this codebase's
   pinhole convention (``x_pixel = fx*x/z + cx``), built with 2DGS's
   specific *un-normalized-by-z* screen convention: a world/camera-space
   point ``(x, y, z, 1)`` maps to screen-homogeneous
   ``(fx*x + cx*z, fy*y + cy*z, z, z)`` -- rows 2 and 3 are identical,
   deliberately, because this is what makes the per-pixel ray-plane
   pullback trick (``rasterize_2dgs``) work without ever inverting a
   per-pixel matrix.

   Because ``H``'s 3rd column is exactly zero and ``W``'s rows 2/3 are
   exactly identical, ``M = W @ H`` has an all-zero 3rd column and
   duplicate 3rd/4th rows: only 3 rows x 3 columns = 9 entries are
   independent. ``transform`` below stores exactly that 3x3 (row-major:
   rows are the screen x-numerator/y-numerator/z rows; columns correspond
   to ``H``'s ``t_u``, ``t_v``, and ``mean`` columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metalsplat.reference.project_ref import project_gaussians as _project_gaussians_3d
from metalsplat.utils.quaternion import quat_to_rotmat

EPS_3RD_AXIS = 1e-6  # world units; stand-in for the missing depth-axis scale
RADIUS_SAFETY_MARGIN = 2.5  # tile-culling radius multiplier, see module docstring


@dataclass
class Projection2DGSResult:
    means2d: torch.Tensor  # (N, 2) pixel-space centers
    depths: torch.Tensor  # (N,) camera-space z of the gaussian mean
    conics: torch.Tensor  # (N, 3) tile-culling-only inverse-2D-covariance (a, b, c)
    radii: torch.Tensor  # (N,) integer pixel radius (tile-culling bound), 0 if culled
    valid: torch.Tensor  # (N,) bool, False for culled gaussians
    compensation: torch.Tensor  # (N,) anti-aliasing opacity scale in [0, 1]
    transform: torch.Tensor  # (N, 3, 3) M's 9 independent entries, row-major
    normal: torch.Tensor  # (N, 3) world-space normal, flipped to face the camera


def project_gaussians_2dgs(
    means: torch.Tensor,  # (N, 3) world space
    scales: torch.Tensor,  # (N, 2) positive, (s_u, s_v)
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
) -> Projection2DGSResult:
    n = means.shape[0]
    device, dtype = means.device, means.dtype

    scales3 = torch.cat(
        [scales, torch.full((n, 1), EPS_3RD_AXIS, device=device, dtype=dtype)], dim=-1
    )
    culling = _project_gaussians_3d(
        means,
        scales3,
        quats,
        R_wc,
        t_wc,
        fx,
        fy,
        cx,
        cy,
        img_width,
        img_height,
        near=near,
        eps2d=eps2d,
    )
    radii = torch.where(
        culling.valid,
        torch.ceil(culling.radii * RADIUS_SAFETY_MARGIN),
        culling.radii,
    )
    # ops.tiling/kernels/tiling.metal derive the actual per-tile bounding
    # box from `conics` (via the 2D covariance's diagonal), not from
    # `radii` -- radii only gates whether a gaussian is culled at all. So
    # the safety margin has to inflate the covariance the conic represents
    # too, or a tilted disk's silhouette can still fall outside every tile
    # it got binned into even with a margined `radii`. Scaling Sigma2d by
    # margin^2 is scaling its inverse (the conic) by 1/margin^2.
    conics = culling.conics / (RADIUS_SAFETY_MARGIN * RADIUS_SAFETY_MARGIN)

    rotmat = quat_to_rotmat(quats)  # (N, 3, 3)
    t_u = rotmat[..., :, 0] * scales[:, 0:1]
    t_v = rotmat[..., :, 1] * scales[:, 1:2]
    normal = rotmat[..., :, 2]

    # 2DGS orients the normal toward the camera: a splat is a two-sided
    # infinitesimally-thin disk, so "which side" is otherwise undefined,
    # and both shading and the normal-consistency loss need a consistent
    # convention.
    cam_pos = -(R_wc.T @ t_wc)
    view_dir = cam_pos.unsqueeze(0) - means  # (N, 3); unnormalized, only sign matters
    flip = (normal * view_dir).sum(-1, keepdim=True) < 0
    normal = torch.where(flip, -normal, normal)

    mean_cam = means @ R_wc.T + t_wc  # (N, 3)
    tu_cam = t_u @ R_wc.T  # (N, 3), direction: no translation
    tv_cam = t_v @ R_wc.T  # (N, 3), direction: no translation

    row0 = torch.stack(
        [
            fx * tu_cam[:, 0] + cx * tu_cam[:, 2],
            fx * tv_cam[:, 0] + cx * tv_cam[:, 2],
            fx * mean_cam[:, 0] + cx * mean_cam[:, 2],
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            fy * tu_cam[:, 1] + cy * tu_cam[:, 2],
            fy * tv_cam[:, 1] + cy * tv_cam[:, 2],
            fy * mean_cam[:, 1] + cy * mean_cam[:, 2],
        ],
        dim=-1,
    )
    row2 = torch.stack([tu_cam[:, 2], tv_cam[:, 2], mean_cam[:, 2]], dim=-1)
    transform = torch.stack([row0, row1, row2], dim=-2)  # (N, 3, 3)

    return Projection2DGSResult(
        means2d=culling.means2d,
        depths=culling.depths,
        conics=conics,
        radii=radii,
        valid=culling.valid,
        compensation=culling.compensation,
        transform=transform,
        normal=normal,
    )
