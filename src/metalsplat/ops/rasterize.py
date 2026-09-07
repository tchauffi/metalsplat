"""torch.autograd.Function wrapping the Metal tile-based rasterizer
(metalsplat.kernels.rasterize). Forward/backward math matches
metalsplat.reference.rasterize_ref exactly -- validated against it in
tests/test_rasterize.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE, bin_and_sort_gaussians


class _RasterizeGaussiansImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,  # (N, 2)
        conics: torch.Tensor,  # (N, 3)
        opacities: torch.Tensor,  # (N,)
        colors: torch.Tensor,  # (N, 3)
        sorted_ids: torch.Tensor,  # (M,) int32
        tile_bins: torch.Tensor,  # (num_tiles, 2) int32
        tiles_x: int,
        img_width: int,
        img_height: int,
        tile_size: int,
        background: torch.Tensor,  # (3,) float32
    ):
        device = means2d.device
        n = means2d.shape[0]
        tiles_y = (tile_bins.shape[0] + tiles_x - 1) // tiles_x if tiles_x > 0 else 0

        means2d_c = means2d.contiguous()
        conics_c = conics.contiguous()
        opacities_c = opacities.contiguous()
        colors_c = colors.contiguous()
        sorted_ids_i32 = sorted_ids.to(torch.int32).contiguous()
        if sorted_ids_i32.numel() == 0:
            sorted_ids_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        tile_bins_i32 = tile_bins.to(torch.int32).contiguous()

        out_image = torch.zeros(img_height, img_width, 3, device=device, dtype=torch.float32)
        out_final_T = torch.ones(img_height, img_width, device=device, dtype=torch.float32)
        out_last_contributor = torch.full(
            (img_height, img_width), -1, device=device, dtype=torch.int32
        )

        width_padded = tiles_x * tile_size
        height_padded = tiles_y * tile_size

        if n > 0 and width_padded > 0 and height_padded > 0:
            lib = load_kernel("rasterize")
            lib.rasterize_forward(
                means2d_c,
                conics_c,
                opacities_c,
                colors_c,
                sorted_ids_i32,
                tile_bins_i32,
                int(tiles_x),
                int(img_width),
                int(img_height),
                int(tile_size),
                background,
                out_image,
                out_final_T,
                out_last_contributor,
                threads=(width_padded, height_padded),
                group_size=(tile_size, tile_size),
            )

        ctx.save_for_backward(
            means2d_c, conics_c, opacities_c, colors_c, sorted_ids_i32, tile_bins_i32,
            out_final_T, out_last_contributor, background,
        )
        ctx.tiles_x = tiles_x
        ctx.tiles_y = tiles_y
        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.tile_size = tile_size
        ctx.n = n
        return out_image, out_final_T

    @staticmethod
    def backward(ctx, grad_out_image, grad_final_T):
        # grad_final_T is ignored: final_T is a structural/auxiliary output
        # (used by metalsplat.seed for coverage detection), never part of
        # the differentiable training loss.
        (
            means2d, conics, opacities, colors, sorted_ids, tile_bins,
            final_T, last_contributor, background,
        ) = ctx.saved_tensors
        device = means2d.device
        n = ctx.n

        d_means2d = torch.zeros(n, 2, device=device, dtype=torch.float32)
        d_conics = torch.zeros(n, 3, device=device, dtype=torch.float32)
        d_opacities = torch.zeros(n, device=device, dtype=torch.float32)
        d_colors = torch.zeros(n, 3, device=device, dtype=torch.float32)

        width_padded = ctx.tiles_x * ctx.tile_size
        height_padded = ctx.tiles_y * ctx.tile_size

        if n > 0 and width_padded > 0 and height_padded > 0:
            lib = load_kernel("rasterize")
            lib.rasterize_backward(
                means2d,
                conics,
                opacities,
                colors,
                sorted_ids,
                tile_bins,
                int(ctx.tiles_x),
                int(ctx.img_width),
                int(ctx.img_height),
                int(ctx.tile_size),
                background,
                final_T,
                last_contributor,
                grad_out_image.contiguous(),
                d_means2d,
                d_conics,
                d_opacities,
                d_colors,
                threads=(width_padded, height_padded),
                group_size=(ctx.tile_size, ctx.tile_size),
            )

        return d_means2d, d_conics, d_opacities, d_colors, None, None, None, None, None, None, None


def rasterize_gaussians(
    means2d: torch.Tensor,
    depths: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    radii: torch.Tensor,
    valid: torch.Tensor,
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: torch.Tensor | None = None,
    return_aux: bool = False,
):
    """Tile-based differentiable alpha-compositing rasterization.

    `means2d`, `conics`, `opacities`, `colors` are the differentiable
    per-gaussian tensors (e.g. from `project_gaussians`); `depths`, `radii`,
    `valid` are used only for (non-differentiable) tile binning.

    If `return_aux` is True, also returns the per-pixel final transmittance
    `final_T` (H, W) -- close to 1 means (almost) no gaussian contributed
    to that pixel, used by metalsplat.seed to find uncovered regions.
    """
    binning = bin_and_sort_gaussians(
        means2d.detach(), depths.detach(), radii.detach(), valid, img_width, img_height, tile_size
    )
    device = means2d.device
    if background is None:
        background = torch.zeros(3, device=device, dtype=torch.float32)

    image, final_T = _RasterizeGaussiansImpl.apply(
        means2d,
        conics,
        opacities,
        colors,
        binning.sorted_gaussian_ids,
        binning.tile_bins,
        binning.tiles_x,
        img_width,
        img_height,
        tile_size,
        background,
    )
    if return_aux:
        return image, final_T
    return image
