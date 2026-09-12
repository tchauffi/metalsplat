"""torch.autograd.Function wrapping the Metal tile-based 2DGS rasterizer
(metalsplat.kernels.rasterize_2dgs). Forward/backward math matches
metalsplat.reference.rasterize_2dgs_ref exactly -- validated against it in
tests/test_rasterize_2dgs.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE, bin_and_sort_gaussians


class _Rasterize2DGSImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,  # (N, 2)
        transform: torch.Tensor,  # (N, 9) row-major 3x3, see project_2dgs_ref
        normal: torch.Tensor,  # (N, 3)
        opacities: torch.Tensor,  # (N,)
        colors: torch.Tensor,  # (N, 3)
        depths: torch.Tensor,  # (N,) gaussian mean depth, forward-only degenerate fallback
        sorted_ids: torch.Tensor,  # (M,) int32
        tile_bins: torch.Tensor,  # (num_tiles, 2) int32
        tiles_x: int,
        img_width: int,
        img_height: int,
        tile_size: int,
        near: float,
        eps2d: float,
        background: torch.Tensor,  # (3,) float32
        abs_grad_accum: torch.Tensor
        | None,  # (N,), mutated in place by backward -- see below
    ):
        device = means2d.device
        n = means2d.shape[0]
        tiles_y = (tile_bins.shape[0] + tiles_x - 1) // tiles_x if tiles_x > 0 else 0

        means2d_c = means2d.contiguous()
        transform_c = transform.contiguous()
        normal_c = normal.contiguous()
        opacities_c = opacities.contiguous()
        colors_c = colors.contiguous()
        depths_c = depths.detach().contiguous()
        sorted_ids_i32 = sorted_ids.to(torch.int32).contiguous()
        if sorted_ids_i32.numel() == 0:
            sorted_ids_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        tile_bins_i32 = tile_bins.to(torch.int32).contiguous()

        out_image = torch.zeros(
            img_height, img_width, 3, device=device, dtype=torch.float32
        )
        out_depth = torch.zeros(
            img_height, img_width, device=device, dtype=torch.float32
        )
        out_normal = torch.zeros(
            img_height, img_width, 3, device=device, dtype=torch.float32
        )
        out_distortion = torch.zeros(
            img_height, img_width, device=device, dtype=torch.float32
        )
        # Sum of weight*m (normalized depth) per pixel. Not a user-facing
        # output: backward seeds the distortion recursion's running prefix
        # sum from it, and it is a *different* accumulation from out_depth
        # (which sums weight*z in metric units).
        out_dist_depth = torch.zeros(
            img_height, img_width, device=device, dtype=torch.float32
        )
        out_final_T = torch.ones(
            img_height, img_width, device=device, dtype=torch.float32
        )
        out_last_contributor = torch.full(
            (img_height, img_width), -1, device=device, dtype=torch.int32
        )

        width_padded = tiles_x * tile_size
        height_padded = tiles_y * tile_size

        if n > 0 and width_padded > 0 and height_padded > 0:
            lib = load_kernel("rasterize_2dgs")
            lib.rasterize_2dgs_forward(
                means2d_c,
                transform_c,
                normal_c,
                opacities_c,
                colors_c,
                depths_c,
                sorted_ids_i32,
                tile_bins_i32,
                int(tiles_x),
                int(img_width),
                int(img_height),
                int(tile_size),
                float(near),
                float(eps2d),
                background,
                out_image,
                out_depth,
                out_normal,
                out_distortion,
                out_dist_depth,
                out_final_T,
                out_last_contributor,
                threads=(width_padded, height_padded),
                group_size=(tile_size, tile_size),
            )

        ctx.save_for_backward(
            means2d_c,
            transform_c,
            normal_c,
            opacities_c,
            colors_c,
            depths_c,
            sorted_ids_i32,
            tile_bins_i32,
            out_dist_depth,
            out_final_T,
            out_last_contributor,
            background,
        )
        ctx.tiles_x = tiles_x
        ctx.tiles_y = tiles_y
        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.tile_size = tile_size
        ctx.near = near
        ctx.eps2d = eps2d
        ctx.n = n
        ctx.abs_grad_accum = abs_grad_accum
        return out_image, out_depth, out_normal, out_distortion, out_final_T

    @staticmethod
    def backward(
        ctx,
        grad_out_image,
        grad_out_depth,
        grad_out_normal,
        grad_out_distortion,
        grad_final_T,
    ):
        # grad_final_T is ignored: forward-only structural output (coverage
        # detection), same convention as 3DGS's rasterize.py.
        (
            means2d,
            transform,
            normal,
            opacities,
            colors,
            depths,
            sorted_ids,
            tile_bins,
            dist_depth,
            final_T,
            last_contributor,
            background,
        ) = ctx.saved_tensors
        device = means2d.device
        n = ctx.n

        d_means2d = torch.zeros(n, 2, device=device, dtype=torch.float32)
        d_transform = torch.zeros(n, 9, device=device, dtype=torch.float32)
        d_normal = torch.zeros(n, 3, device=device, dtype=torch.float32)
        d_opacities = torch.zeros(n, device=device, dtype=torch.float32)
        d_colors = torch.zeros(n, 3, device=device, dtype=torch.float32)
        # AbsGS-style densification signal (see kernels/rasterize_2dgs.metal):
        # if the caller passed a persistent accumulator, the kernel adds
        # into it in place -- same convention as ops/rasterize.py.
        abs_grad_accum = ctx.abs_grad_accum
        if abs_grad_accum is None:
            abs_grad_accum = torch.zeros(n, device=device, dtype=torch.float32)

        width_padded = ctx.tiles_x * ctx.tile_size
        height_padded = ctx.tiles_y * ctx.tile_size

        if n > 0 and width_padded > 0 and height_padded > 0:
            lib = load_kernel("rasterize_2dgs")
            lib.rasterize_2dgs_backward(
                means2d,
                transform,
                normal,
                opacities,
                colors,
                depths,
                sorted_ids,
                tile_bins,
                int(ctx.tiles_x),
                int(ctx.img_width),
                int(ctx.img_height),
                int(ctx.tile_size),
                float(ctx.near),
                float(ctx.eps2d),
                background,
                final_T,
                dist_depth,
                last_contributor,
                grad_out_image.contiguous(),
                grad_out_depth.contiguous(),
                grad_out_normal.contiguous(),
                grad_out_distortion.contiguous(),
                d_means2d,
                d_transform,
                d_normal,
                d_opacities,
                d_colors,
                abs_grad_accum,
                threads=(width_padded, height_padded),
                group_size=(ctx.tile_size, ctx.tile_size),
            )

        # One gradient per forward() input: means2d, transform, normal,
        # opacities, colors get real gradients; depths, sorted_ids,
        # tile_bins, tiles_x, img_width, img_height, tile_size, near,
        # eps2d, background, abs_grad_accum don't.
        return (
            d_means2d,
            d_transform,
            d_normal,
            d_opacities,
            d_colors,
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
            None,
        )


def rasterize_gaussians_2dgs(
    means2d: torch.Tensor,
    transform: torch.Tensor,
    normal: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    radii: torch.Tensor,
    valid: torch.Tensor,
    conics: torch.Tensor,
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    near: float = 0.2,
    eps2d: float = 0.3,
    background: torch.Tensor | None = None,
    abs_grad_accum: torch.Tensor | None = None,
):
    """Tile-based differentiable ray-splat rasterization of 2D gaussians.

    `means2d`, `transform`, `normal`, `opacities`, `colors` are the
    differentiable per-gaussian tensors (from `project_gaussians_2dgs`);
    `depths`, `radii`, `valid`, `conics` are used only for (non-
    differentiable) tile binning -- `conics`/`radii` are the tile-culling
    approximation project_gaussians_2dgs derives via the reused 3DGS EWA
    path, not used for shading.

    Returns `(image, depth, normal, distortion, final_T)`. Unlike 3DGS's
    `rasterize_gaussians`, `depth` and `normal` here carry real gradients
    (the ray-splat intersection depth is differentiable), and `distortion`
    is the per-pixel Mip-NeRF-360/2DGS regularizer map (sum over
    contributing gaussian pairs of `w_i*w_j*|m_i-m_j|`, on *normalized*
    depth `m = far/(far-near)*(1-near/z)` rather than metric z, matching
    the official implementation -- see
    `metalsplat.reference.rasterize_2dgs_ref`'s module docstring) -- feed
    `distortion.mean()`-style reductions to `metalsplat.losses.
    distortion_loss`. `final_T` is the per-pixel final transmittance,
    forward-only, same convention as 3DGS.

    `abs_grad_accum`, if given, is an (N,) tensor that backward()
    atomically adds each gaussian's screen-space AbsGS-style
    densification signal into -- see kernels/rasterize_2dgs.metal's
    backward derivation note for how it's recovered from the ray-splat
    path (most pixels don't take the screen-space-fallback path that a
    literal `d_means2d` would only capture). Prefer this over
    `means2d.grad.norm()` for `metalsplat.densify2dgs`, matching 3DGS's
    `rendering.render`/`metalsplat.densify` convention.
    """
    binning = bin_and_sort_gaussians(
        means2d.detach(),
        depths.detach(),
        conics.detach(),
        radii.detach(),
        valid,
        img_width,
        img_height,
        tile_size,
    )
    device = means2d.device
    if background is None:
        background = torch.zeros(3, device=device, dtype=torch.float32)

    return _Rasterize2DGSImpl.apply(
        means2d,
        transform,
        normal,
        opacities,
        colors,
        depths,
        binning.sorted_gaussian_ids,
        binning.tile_bins,
        binning.tiles_x,
        img_width,
        img_height,
        tile_size,
        near,
        eps2d,
        background,
        abs_grad_accum,
    )
