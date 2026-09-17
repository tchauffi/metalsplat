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

from collections import OrderedDict
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
    regularizer, `sum_{i<j} w_i*w_j*(m_i - m_j)^2` per pixel -- see
    `rasterize_2dgs_ref`'s module docstring for the full definition). The
    expensive part already happened in the rasterizer; this is just the
    scalar reduction.

    No clamp: every term of that sum is a weighted square, so the map is
    non-negative up to float32 cancellation in the rasterizer's
    prefix-moment expansion (measured a few units in the last place, ~1e-5
    of a typical peak -- nothing an optimizer can exploit). It used to be
    clamped at 0, back when the rasterizer accumulated the *signed*
    first-power variant, which went genuinely negative and whose minimizer
    was depth-scrambling rather than depth-concentrating; no clamp could
    fix that, since zero stayed reachable by scrambling. See the same
    docstring.
    """
    return distortion_map.mean()


ALPHA_EPS = 1e-6  # floor for the depth/alpha division below

# Bounded because the entries are full-frame GPU tensors ((H, W, 3) float)
# held for the process lifetime: a multi-resolution or multi-rig eval loop
# visits a new key per geometry and would otherwise accumulate one such
# tensor for each, unfreeable. The intended case needs a single entry (one
# intrinsic matrix for a whole training run), so a handful of slots keeps
# every realistic working set resident while making the worst case finite.
_RAY_CACHE_MAXSIZE = 8
_RAY_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _camera_rays(h, w, camera: Camera, device, dtype) -> torch.Tensor:
    """(H, W, 3) camera-space ray directions (z == 1) through pixel centers.

    Depends only on the image size and intrinsics, so it is built once per
    distinct camera geometry rather than per call -- a capture typically
    shares one intrinsic matrix across every view, making this a single
    cached tensor for a whole training run. Rebuilding it each step costs
    four extra kernel launches on MPS for a result that never changes.

    Least-recently-used beyond `_RAY_CACHE_MAXSIZE` entries; see the note
    there for why it is bounded at all.
    """
    key = (h, w, camera.fx, camera.fy, camera.cx, camera.cy, device, dtype)
    rays = _RAY_CACHE.get(key)
    if rays is not None:
        _RAY_CACHE.move_to_end(key)
    else:
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) + 0.5,
            torch.arange(w, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        rays = torch.stack(
            [
                (xs - camera.cx) / camera.fx,
                (ys - camera.cy) / camera.fy,
                torch.ones_like(xs),
            ],
            dim=-1,
        )
        _RAY_CACHE[key] = rays
        if len(_RAY_CACHE) > _RAY_CACHE_MAXSIZE:
            _RAY_CACHE.popitem(last=False)
    return rays


def normal_consistency_loss(
    rendered_normal: torch.Tensor,  # (H, W, 3), world-space, from render_2dgs
    rendered_depth: torch.Tensor,  # (H, W), camera-space z, from render_2dgs
    alpha: torch.Tensor,  # (H, W) accumulated opacity, i.e. 1 - render_2dgs's final_T
    camera: Camera,
) -> torch.Tensor:
    """2DGS's normal-consistency regularizer: compares the alpha-composited
    surfel normal against a "pseudo-normal" derived from the local shape of
    the rendered depth map (image-space finite differences of unprojected
    3D points) -- teaches depth and normals to agree with each other,
    which is what makes the reconstructed surface usable for meshing.

    Both of the rasterizer's outputs this consumes are *alpha-weighted
    sums*, not averages, so `alpha` is needed to interpret either one --
    this mirrors the official implementation's two uses of `render_alpha`
    (`gaussian_renderer/__init__.py`), and dropping either is silently
    wrong rather than merely approximate:

    - **The depth is un-normalized.** `rendered_depth` is `sum_k w_k*z_k`,
      which is the expected depth scaled by `alpha`. It has to be divided
      by `alpha` before unprojecting, or the unprojected points carry a
      spatially-varying scale factor and the finite differences below
      measure that factor's gradient rather than the surface's. Harmless
      where alpha saturates, badly wrong everywhere else: measured ~70
      degrees of pseudo-normal error at alpha 0.5-0.99, against ~6 degrees
      at alpha ~1.
    - **The rendered normal is un-normalized** too (its magnitude is
      ~alpha), so the unit pseudo-normal is scaled by `alpha` to match.
      Without that the target is systematically too long wherever the
      surface is semi-transparent, and -- the part that actually bites --
      the gradient stays full-strength in empty regions instead of
      vanishing with alpha, so the term keeps pushing normals (and through
      them opacities) up in parts of the frame that should stay empty.

    The pseudo-normal's *direction* is not detached: gradient flows into
    both `rendered_normal` and `rendered_depth`, matching the official
    implementation's `depth_to_normal` path -- letting both sides move
    toward mutual consistency, not just the rendered normal toward a fixed
    target. Its alpha *scale* is detached, also matching.

    Note `alpha` itself carries no gradient here: it comes from
    `render_2dgs`'s `final_T`, which this codebase's rasterizer treats as a
    forward-only structural output. The official rasterizer does
    differentiate its alpha channel, so the depth-normalization path
    differs from it by that (conservative) omission.

    Only defined on interior pixels (finite differences need both
    neighbours), so this compares `rendered_normal[1:-1, 1:-1]` against
    the pseudo-normal.
    """
    h, w = rendered_depth.shape
    # Expected depth. The `where` is not just cosmetic: a plain
    # `rendered_depth / alpha.clamp_min(eps)` would send gradients through
    # the floor at uncovered pixels, scaling them by 1/eps -- measured
    # 1e8-magnitude depth gradients leaking back into the rasterizer from
    # pixels where nothing was rendered at all. `where` blocks the
    # unselected branch, so those pixels contribute a clean zero, which is
    # also what the official implementation's nan_to_num(0/0) does (its
    # backward zeroes the gradient of every non-finite entry).
    covered = alpha > ALPHA_EPS
    expected_depth = torch.where(
        covered, rendered_depth / alpha.clamp_min(ALPHA_EPS), torch.zeros_like(alpha)
    )

    # Unprojection stays in *camera* space and the rendered normal is
    # rotated into it, rather than the other way round. A rotation commutes
    # with both the cross product (R is proper, so cross(Ra, Rb) =
    # R cross(a, b)) and the normalize, so the comparison is identical --
    # but this way the (H, W, 3) point cloud never has to be rotated or
    # translated, which on MPS is most of this function's cost. The
    # translation drops out entirely: finite differences cancel it.
    rays = _camera_rays(h, w, camera, rendered_depth.device, rendered_depth.dtype)
    points_cam = rays * expected_depth[..., None]  # (H, W, 3)

    # Row difference first, column difference second. `cross(d_row, d_col)`
    # is the order whose result comes out *facing the camera*, which is the
    # convention project_2dgs_ref already puts the rendered normal in (and
    # the order the official implementation's `depth_to_normal` uses, where
    # its `dx` is likewise the row difference). Swapping the two mirrors the
    # surface, and the sign of `dot` below is exactly what tells a surface
    # from its mirror image -- so this order is load-bearing, not cosmetic.
    d_row = points_cam[2:, 1:-1, :] - points_cam[:-2, 1:-1, :]
    d_col = points_cam[1:-1, 2:, :] - points_cam[1:-1, :-2, :]
    pseudo_normal = F.normalize(torch.linalg.cross(d_row, d_col, dim=-1), dim=-1)

    normal_interior = rendered_normal[1:-1, 1:-1, :]
    normal_cam = normal_interior @ camera.R_wc.T  # world -> camera

    # Signed, as upstream -- deliberately not `abs()`. The sign carries the
    # entire regularizer: `dot` is +1 where the depth surface faces the
    # camera the way the rendered normal says it does, and -1 where the
    # depth map folds back on itself, which is what a spike or a needle
    # poking out of a surface looks like in the depth map. Only the signed
    # form charges for that fold (up to 2 per pixel). `|dot|` scores a
    # fully folded pixel exactly as well as a correct one, and -- the part
    # that actually destroys geometry -- its gradient pushes `dot` toward
    # whichever end it already sits nearest, so a surface that has begun to
    # fold is driven to fold *harder*. That is a regularizer that grows
    # needles instead of flattening them, and it is what tore this scene
    # apart the step LAMBDA_NORMAL switched on: measured 11k steps into a
    # garden run, 22% of well-covered pixels had already flipped past
    # dot < 0, at no cost under `abs()`.
    #
    # `abs()` was standing in for a fixed *global* handedness (the pixel
    # grid's, not the scene's); the cross product's argument order above is
    # where that belongs, and it costs nothing per pixel.
    dot = (pseudo_normal * normal_cam).sum(-1)
    # `alpha` matches the rendered normal's own alpha weighting; detached,
    # as upstream. It is already gradient-free here (final_T is forward-only)
    # but the detach keeps that intent explicit.
    return (1.0 - alpha[1:-1, 1:-1].detach() * dot).mean()
