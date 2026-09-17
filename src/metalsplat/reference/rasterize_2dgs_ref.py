"""Pure-PyTorch brute-force reference rasterizer for 2D gaussian splats
("surfels"): resolves the *exact* ray-splat intersection per pixel per
gaussian (see ``project_2dgs_ref``'s module docstring for the ``M``/``H``/
``W`` derivation) rather than 3DGS's local-affine conic approximation.

Not tile-based and not remotely fast -- every pixel walks every gaussian in
depth order -- this is deliberately the simplest possible correct
implementation, used only as the executable spec / gradient oracle for the
Metal ``rasterize_2dgs`` kernel on small test scenes, and as a
CPU-compatible fallback. It is plain differentiable torch ops, so
``torch.autograd`` gives its backward for free -- including for the
distortion loss, whose closed-form backward (see `rasterize_2dgs.metal`)
would otherwise be the hardest thing in this feature to hand-derive and
trust; that closed form was validated against `torch.autograd` on this
module's plain recursion before being written into the kernel.

Distortion is Mip-NeRF-360's pairwise second moment of the per-pixel
weight distribution, `sum_{i<j} w_i*w_j*(m_i - m_j)^2`, accumulated in a
single front-to-back pass by the standard prefix-sum expansion
`sum_k w_k * (m_k^2*A_{k-1} - 2*m_k*M1_{k-1} + M2_{k-1})`, where `A`/`M1`/
`M2` are running prefix sums of `w` / `w*m` / `w*m^2` (this is exactly
2DGS's appendix trick, and exactly what `diff-surfel-rasterization`'s
`forward.cu` computes).

The *squared* difference is what makes this a regularizer at all, and it
is worth being explicit about why, because the cheaper-looking
first-power variant `2 * sum_k w_k * (m_k*A_{k-1} - D_{k-1})` =
`sum_{i<j} w_i*w_j*(m_j - m_i)` is a trap this module used to fall into.
That variant is *signed* and therefore depends on the compositing
sequence rather than on the weight distribution: gaussians are composited
in order of their *mean* depth, but each pixel's actual ray-splat
intersection depth `z_hit` need not be monotonic in that order, so the
sum can be driven arbitrarily negative by tilting splats until their
`z_hit` ordering contradicts their mean-depth ordering. Minimizing it
therefore *rewards* edge-on, depth-scrambling splats -- the exact
opposite of the intended "concentrate the weight along the ray" effect --
and no per-pixel clamp fixes that, since a clamp at zero still leaves
depth-scrambling as a free way to reach zero. The squared form has
neither problem: every term is non-negative, the sum is symmetric in
`i`/`j` so the compositing order drops out entirely, and its only
minimizer is a weight distribution concentrated at a single depth.

`m` is *not* the raw metric intersection depth: it is the normalized
inverse-depth `m = far/(far - near) * (1 - near/z_hit)`, which maps
`z_hit` in `[near, far]` onto `[0, 1]`. This matches the official 2DGS
CUDA rasterizer (`diff-surfel-rasterization`'s `forward.cu`, which
computes exactly this `m` from its `near_n = 0.2` / `far_n = 100.0`
constants) and it is what makes the paper's own `lambda_dist` values
(100 for unbounded scenes, 1000 for bounded ones) meaningful. Accumulating
raw metric `z_hit` instead would make the regularizer's magnitude scale
with the square of the scene's units -- at a typical 5m depth, ~4 orders
of magnitude larger -- so a `lambda_dist` copied from the paper would
swamp the photometric loss entirely. `near` is this function's own `near`
parameter (so the two stay consistent if it is changed); `far` is
`DISTORTION_FAR`, matching the official implementation's hardcoded
constant.
"""

from __future__ import annotations

import torch

# Far plane used only to normalize depth for the distortion regularizer.
# Matches diff-surfel-rasterization's `__device__ const float far_n = 100.0`.
DISTORTION_FAR = 100.0

# Standard deviation, in pixels, of the screen-space low-pass fallback
# below. Matches the official rasterizer verbatim (auxiliary.h in
# diff-surfel-rasterization):
#
#     __device__ const float FilterSize = 0.707106; // sqrt(2) / 2
#     __device__ const float FilterInvSquare = 2.0f;
#
# i.e. `rho_screen = 2 * d^2`, a variance of 0.5 px^2. Sub-pixel by
# design: this is an anti-aliasing floor that keeps a splat from falling
# between sample points, not a blur.
#
# The width matters more than it looks, because `rho = min(rho_uv,
# rho_screen)` makes this a *lower bound* on every splat's screen
# footprint -- no splat can render smaller than this filter, however small
# its disk is. Widening it takes away the optimizer's ability to sharpen
# detail by shrinking a splat, and densification answers with overlapping
# larger splats instead; 2.0 here (8x this variance) visibly does that.
#
# Deliberately separate from the projection's `eps2d`: that one dilates
# the 2D *covariance* (px^2) for the EWA/tile-culling bound, this one is a
# filter width (px) for an isotropic screen-space gaussian. They were the
# same number here once -- eps2d = 0.3 read as a variance is sigma
# ~0.548px against this 0.707px, so the old behaviour was slightly tight
# rather than wildly off, and neither knob could move without the other.
DEFAULT_FILTER_SIZE = 0.707106  # sqrt(2) / 2


def rasterize_gaussians_2dgs(
    means2d: torch.Tensor,  # (N, 2) -- depth-sort ordering & the screen-space low-pass term
    depths: torch.Tensor,  # (N,) camera-space z of each gaussian's mean -- depth-sort ordering
    transform: torch.Tensor,  # (N, 3, 3) -- see project_2dgs_ref.Projection2DGSResult.transform
    normal: torch.Tensor,  # (N, 3) world-space, camera-facing
    opacities: torch.Tensor,  # (N,)
    colors: torch.Tensor,  # (N, 3)
    valid: torch.Tensor,  # (N,) bool-ish
    img_width: int,
    img_height: int,
    near: float = 0.2,
    filter_size: float = DEFAULT_FILTER_SIZE,
    background: torch.Tensor | None = None,  # (3,)
):
    """Returns a dict with `image`, `depth`, `normal`, `distortion`, `final_T`.

    `depth`/`normal`/`distortion` are alpha-weighted per-pixel maps, fully
    differentiable (unlike 3DGS's forward-only `rasterize_ref.depth`) --
    2DGS's normal-consistency and distortion losses need real gradients
    through depth. `depth` uses each pixel's actual ray-splat intersection
    depth, not the gaussian's mean depth. `distortion` is the Mip-NeRF
    360-style "concentrate the weight along the ray" regularizer -- see
    this module's docstring for its exact (compositing-order) definition.
    """
    device, dtype = means2d.device, means2d.dtype
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)

    valid_mask = valid > 0.5 if valid.dtype != torch.bool else valid
    order = torch.argsort(
        torch.where(valid_mask, depths, torch.full_like(depths, float("inf")))
    )

    ys, xs = torch.meshgrid(
        torch.arange(img_height, device=device, dtype=dtype) + 0.5,
        torch.arange(img_width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )  # (H, W)
    pixels = torch.stack([xs, ys], dim=-1)  # (H, W, 2)

    image = torch.zeros(img_height, img_width, 3, device=device, dtype=dtype)
    depth_map = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    normal_map = torch.zeros(img_height, img_width, 3, device=device, dtype=dtype)
    distortion_map = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    trans = torch.ones(img_height, img_width, device=device, dtype=dtype)
    # Sticky per-pixel "this pixel is finished" flag, standing in for the
    # kernel's `break`. Needed because this loop is vectorized over pixels
    # and cannot break per pixel: without it, a pixel that stopped could be
    # revived by a later gaussian whose alpha is small enough to satisfy the
    # transmittance test again, which the kernel -- having left the loop --
    # would never do.
    stopped = torch.zeros(img_height, img_width, device=device, dtype=torch.bool)

    # Running prefix sums for the distortion expansion (see module
    # docstring): the zeroth/first/second moments of the weight
    # distribution over normalized depth `m` accumulated so far.
    dist_A = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    dist_M1 = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    dist_M2 = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    dist_scale = DISTORTION_FAR / (DISTORTION_FAR - near)

    for i in order:
        if not bool(valid_mask[i]):
            continue
        row0, row1, row2 = transform[i, 0], transform[i, 1], transform[i, 2]

        # Pull back the pixel's ray-defining planes (X - x*Z = 0, Y - y*Z =
        # 0, in the reduced (X, Y, Z=W) screen space -- see
        # project_2dgs_ref) into the gaussian's local (u, v, 1) frame, then
        # solve the resulting 2x2 linear system via the standard
        # homogeneous-line-intersection cross product.
        h_u = row0 - xs.unsqueeze(-1) * row2  # (H, W, 3)
        h_v = row1 - ys.unsqueeze(-1) * row2  # (H, W, 3)
        cross = torch.linalg.cross(h_u, h_v, dim=-1)  # (H, W, 3)
        w_local = cross[..., 2]
        degenerate = w_local.abs() < 1e-9  # ray (near-)parallel to the splat's plane
        w_safe = torch.where(degenerate, torch.ones_like(w_local), w_local)
        u = cross[..., 0] / w_safe
        v = cross[..., 1] / w_safe
        rho_uv = torch.where(
            degenerate, torch.full_like(w_local, float("inf")), u * u + v * v
        )

        # Screen-space low-pass fallback (2DGS's own anti-aliasing
        # compromise for near-edge-on views): take the larger of the two
        # Gaussian responses, i.e. the smaller exponent.
        d2d = pixels - means2d[i]
        rho_screen = (d2d[..., 0] ** 2 + d2d[..., 1] ** 2) / (filter_size**2)
        uv_active = rho_uv <= rho_screen
        rho = torch.where(uv_active, rho_uv, rho_screen)

        # z at the intersection is the plane row2 (== M's shared Z=W row,
        # which is exactly camera-space z, see project_2dgs_ref) evaluated
        # at the same (u, v, 1) -- but only trusted when the ray-splat term
        # actually won the min() above. Near the boundary where the
        # screen-space fallback takes over, `wloc` can be small-but-not-
        # quite-degenerate, and u/v (scaling as 1/wloc) blow up to
        # arbitrary, numerically unstable values that happen not to affect
        # alpha (rho_screen dominates) but would otherwise poison the
        # depth/distortion outputs and amplify ordinary GPU/CPU float32
        # differences into large, spurious depth disagreements. Falling
        # back to the gaussian's mean depth there (same as the fully
        # degenerate case) matches "this pixel is being shaded by a
        # generic small screen-space blob" rather than an extrapolated,
        # meaningless intersection point.
        z_hit = torch.where(
            degenerate | ~uv_active,
            depths[i].expand_as(w_local),
            row2[0] * u + row2[1] * v + row2[2],
        )

        alpha = (opacities[i] * torch.exp(-0.5 * rho)).clamp(max=0.99)
        # A large, obliquely-tilted disk can have its mean in front of
        # `near` while part of its footprint's intersection lands behind
        # the camera for some pixels -- the per-gaussian `in_front` check
        # in project_2dgs_ref only guards the mean.
        alpha = torch.where(z_hit <= near, torch.zeros_like(alpha), alpha)

        # Matches kernels/rasterize_2dgs.metal exactly, including which
        # gaussian the transmittance cutoff drops. The kernel computes
        # `test_T = T * (1 - alpha)` and breaks *before* compositing when
        # that falls below 1e-4, so the gaussian that would exhaust the
        # pixel contributes nothing. Gating on the pre-update `trans`
        # instead would composite it (at T = 9e-3 and alpha = 0.99, a
        # weight of ~8.9e-3 the kernel never adds) and leave the oracle
        # ~1% off the code it certifies on that pixel.
        #
        # A too-faint gaussian is skipped but does *not* finish the pixel,
        # again as in the kernel (`continue`, not `break`).
        visible = (alpha >= 1.0 / 255.0) & ~stopped
        exhausts = visible & (trans * (1.0 - alpha) < 1e-4)
        alpha_eff = torch.where(visible & ~exhausts, alpha, torch.zeros_like(alpha))
        weight = trans * alpha_eff

        # Normalized depth for the distortion regularizer only -- `depth_map`
        # stays in metric units. Clamped at `near` purely to keep `near/z`
        # finite: alpha (hence weight) is already exactly 0 wherever
        # `z_hit <= near`, so the clamp never touches a contributing term.
        m = dist_scale * (1.0 - near / z_hit.clamp_min(near))

        image = image + weight[..., None] * colors[i]
        depth_map = depth_map + weight * z_hit
        normal_map = normal_map + weight[..., None] * normal[i]
        distortion_map = distortion_map + weight * (
            m * m * dist_A - 2.0 * m * dist_M1 + dist_M2
        )
        dist_A = dist_A + weight
        dist_M1 = dist_M1 + weight * m
        dist_M2 = dist_M2 + weight * m * m
        trans = trans * (1 - alpha_eff)
        stopped = stopped | exhausts

    image = image + trans[..., None] * background

    return {
        "image": image,
        "depth": depth_map,
        "normal": normal_map,
        "distortion": distortion_map,
        "final_T": trans,
    }
