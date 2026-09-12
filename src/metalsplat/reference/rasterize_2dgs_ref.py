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

Distortion is defined on the *compositing sequence* (gaussians sorted by
mean depth, the same order used for alpha blending), as
`2 * sum_k w_k * (z_k * A_{k-1} - D_{k-1})` where `A`/`D` are running
prefix sums of weight / weight*z_hit -- algebraically equal to
`sum_{i<j} w_i*w_j*(z_j - z_i)` over that *sequence* order. This is *not*
the same as `sum_i sum_j w_i*w_j*|z_i - z_j|` re-sorted by actual
intersection depth: 2DGS's exact per-pixel ray-splat intersection depth
can differ from a gaussian's mean depth enough that the compositing
sequence isn't itself monotonic in `z_hit` at a given pixel, in which case
this (deliberately, for O(N) tractability -- re-sorting per pixel would
need an O(N log N) sort with its own backward) can be small or even
slightly negative rather than the ideal always-non-negative penalty. This
matches what an efficient GPU implementation can actually compute, and is
the same order-dependent definition the Metal kernel's running-sum forward
uses -- the two must never be allowed to diverge (e.g. one re-sorting by
`z_hit`, the other not), or kernel-vs-reference tests stop being a
meaningful check of anything.
"""

from __future__ import annotations

import torch


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
    eps2d: float = 0.3,
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

    # Running prefix sums for the distortion telescoping recursion (see
    # module docstring): A = sum of weights so far, D = sum of
    # weight*z_hit so far, in compositing-sequence order.
    dist_A = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    dist_D = torch.zeros(img_height, img_width, device=device, dtype=dtype)

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
        rho_screen = (d2d[..., 0] ** 2 + d2d[..., 1] ** 2) / eps2d
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

        active = trans >= 1e-4
        alpha_eff = torch.where(
            active & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha)
        )
        weight = trans * alpha_eff

        image = image + weight[..., None] * colors[i]
        depth_map = depth_map + weight * z_hit
        normal_map = normal_map + weight[..., None] * normal[i]
        distortion_map = distortion_map + 2.0 * weight * (z_hit * dist_A - dist_D)
        dist_A = dist_A + weight
        dist_D = dist_D + weight * z_hit
        trans = trans * (1 - alpha_eff)

    image = image + trans[..., None] * background

    return {
        "image": image,
        "depth": depth_map,
        "normal": normal_map,
        "distortion": distortion_map,
        "final_T": trans,
    }
