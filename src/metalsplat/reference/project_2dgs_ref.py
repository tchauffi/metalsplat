"""Pure-PyTorch reference projection of 2D gaussian splats ("surfels") to
2D screen space. Executable spec (runnable on CPU) for
``metalsplat.kernels.project_2dgs``, same role as ``project_ref`` for 3DGS.

This does two distinct jobs:

1. **Exact screen footprint** for tile binning and culling (`rects`,
   `valid`), from the same transform the rasterizer intersects -- see
   ``surfel_rects``. The rasterizer composites a pixel wherever
   ``min(rho_uv, rho_screen)`` is under the alpha cutoff, so the
   footprint is the perspective image of the local ellipse
   ``u^2 + v^2 <= c^2`` joined with the screen-space filter's disk, and
   both have closed-form bounding boxes. This replaced a reused 3DGS EWA
   (local-affine) bound inflated by a 2.5x safety margin: the affine
   approximation underestimates a tilted or near-camera disk's true
   perspective footprint, and the margin that papered over that also made
   every face-on splat's box ~2.5x too wide in each axis.
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

from metalsplat.reference.rasterize_2dgs_ref import DEFAULT_FILTER_SIZE
from metalsplat.reference.tiling_ref import EMPTY_RECT, MAX_SIGMA_EXTENT, sigma_extent
from metalsplat.utils.quaternion import quat_to_rotmat


@dataclass
class Projection2DGSResult:
    means2d: torch.Tensor  # (N, 2) pixel-space centers
    depths: torch.Tensor  # (N,) camera-space z of the gaussian mean
    rects: (
        torch.Tensor
    )  # (N, 4) exact screen footprint, xmin/ymin/xmax/ymax; empty if culled
    valid: torch.Tensor  # (N,) bool, False for culled gaussians
    transform: torch.Tensor  # (N, 3, 3) M's 9 independent entries, row-major
    normal: torch.Tensor  # (N, 3) world-space normal, flipped to face the camera


def surfel_rects(
    transform: torch.Tensor,  # (N, 3, 3), rows as in Projection2DGSResult.transform
    means2d: torch.Tensor,  # (N, 2)
    cutoff: torch.Tensor,  # (N,) alpha cutoff radius in splat sigmas
    filter_size: float,
    img_width: int,
    img_height: int,
) -> torch.Tensor:
    """(N, 4) exact screen rectangle of everywhere each splat composites.

    The rasterizer's alpha falls below its cutoff where
    ``min(rho_uv, rho_screen) > c^2``, so the footprint is the union of:

    - the perspective image of the local ellipse ``u^2 + v^2 <= c^2``.
      With ``T`` the transform (rows: x-numerator, y-numerator, w), a local
      point ``p = (u, v, 1)`` lands at screen ``(T0.p / T2.p, T1.p /
      T2.p)``. The ellipse is the conic ``diag(1, 1, -c^2)``; its image's
      *dual* conic is ``C* = T diag(c^2, c^2, -1) T^T``, and an axis-aligned
      tangent ``x = x0`` satisfies ``C*_00 - 2 x0 C*_02 + x0^2 C*_22 = 0``,
      giving center ``C*_02 / C*_22`` and half-extent
      ``sqrt(center^2 - C*_00 / C*_22)``. This is exact, where the old EWA
      bound was a first-order approximation. The image is a bounded
      ellipse iff ``C*_22 < 0`` -- i.e. the local ellipse stays entirely
      in front of the camera plane; otherwise its image is unbounded and
      the whole frame is the only safe bound. Computed in coordinates
      relative to the projected center (``T0 - mx*T2``, ``T1 - my*T2``) so
      that a small splat far from the principal point does not lose its
      extent to float32 cancellation.
    - the screen-space filter's disk, radius ``c * filter_size`` around
      ``means2d``.
    """
    mx, my = means2d[:, 0:1], means2d[:, 1:2]
    r2 = transform[:, 2, :]
    r0 = transform[:, 0, :] - mx * r2
    r1 = transform[:, 1, :] - my * r2
    c2 = cutoff * cutoff
    d = torch.stack([c2, c2, -torch.ones_like(c2)], dim=-1)  # (N, 3)

    c22 = (d * r2 * r2).sum(-1)
    bounded = c22 < 0
    c22_safe = torch.where(bounded, c22, -torch.ones_like(c22))
    ox = (d * r0 * r2).sum(-1) / c22_safe
    oy = (d * r1 * r2).sum(-1) / c22_safe
    hx = (ox * ox - (d * r0 * r0).sum(-1) / c22_safe).clamp_min(0.0).sqrt()
    hy = (oy * oy - (d * r1 * r1).sum(-1) / c22_safe).clamp_min(0.0).sqrt()
    mx, my = mx[:, 0], my[:, 0]
    ellipse = torch.stack([mx + ox - hx, my + oy - hy, mx + ox + hx, my + oy + hy], -1)
    frame = torch.tensor(
        [0.0, 0.0, float(img_width), float(img_height)],
        device=transform.device,
        dtype=transform.dtype,
    ).expand_as(ellipse)
    rect = torch.where(bounded[:, None], ellipse, frame)

    s = cutoff * filter_size
    return torch.stack(
        [
            torch.minimum(rect[:, 0], mx - s),
            torch.minimum(rect[:, 1], my - s),
            torch.maximum(rect[:, 2], mx + s),
            torch.maximum(rect[:, 3], my + s),
        ],
        dim=-1,
    )


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
    filter_size: float = DEFAULT_FILTER_SIZE,
    opacities: torch.Tensor | None = None,  # (N,), tightens `rects`; see below
) -> Projection2DGSResult:
    """`valid` is judged at the largest cutoff any opacity reaches
    (MAX_SIGMA_EXTENT), so visibility does not depend on opacity, as in
    3DGS. `rects` uses each splat's own opacity-aware cutoff when
    `opacities` is given (see tiling_ref.sigma_extent), and full opacity
    otherwise; a culled splat gets EMPTY_RECT.
    """
    n = means.shape[0]
    device, dtype = means.device, means.dtype

    mean_cam = means @ R_wc.T + t_wc  # (N, 3)
    depths = mean_cam[:, 2]
    z_safe = depths.clamp_min(near)
    means2d = torch.stack(
        [fx * mean_cam[:, 0] / z_safe + cx, fy * mean_cam[:, 1] / z_safe + cy],
        dim=-1,
    )

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

    with torch.no_grad():
        full = torch.full((n,), MAX_SIGMA_EXTENT, device=device, dtype=dtype)
        vis = surfel_rects(transform, means2d, full, filter_size, img_width, img_height)
        in_bounds = (
            (vis[:, 2] >= 0)
            & (vis[:, 0] < img_width)
            & (vis[:, 3] >= 0)
            & (vis[:, 1] < img_height)
        )
        valid = (depths > near) & in_bounds

        cutoff = (
            full
            if opacities is None
            else sigma_extent(opacities.detach(), n, device).to(dtype)
        )
        rects = surfel_rects(
            transform, means2d, cutoff, filter_size, img_width, img_height
        )
        empty = torch.tensor(EMPTY_RECT, device=device, dtype=dtype)
        rects = torch.where((valid & (cutoff > 0))[:, None], rects, empty)

    return Projection2DGSResult(
        means2d=means2d,
        depths=depths,
        rects=rects,
        valid=valid,
        transform=transform,
        normal=normal,
    )
