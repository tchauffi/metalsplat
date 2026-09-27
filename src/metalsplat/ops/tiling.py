"""Tile binning and sorting for the rasterizer, using Metal kernels for the
per-pair work and torch's sort for the ordering.

Given each gaussian's screen rectangle, depth and validity, this builds:
- a flat list of (gaussian_id, tile_id) pairs, one per tile a gaussian's
  rectangle touches, sorted by (tile_id, depth) so that within a tile the
  gaussians are front-to-back;
- per-tile [start, end) index ranges into that sorted list.

`bin_and_sort_rects` takes the rectangles directly (2DGS computes exact
ones in its projection); `bin_and_sort_gaussians` derives them from 3DGS's
conic first.

This whole stage is non-differentiable (radii/tile membership are discrete/
structural, matching gsplat), so it deliberately operates outside autograd.

The pure-PyTorch version this replaces lives in
metalsplat.reference.tiling_ref and remains the oracle it is tested
against. On the garden scene (438k gaussians, 1.97M pairs, 1297x840) that
version cost 16.3ms, of which the sort was only 4.6ms; the remaining 11.7ms
was expansion and gather traffic -- `repeat_interleave` to assign each pair
to a gaussian, integer div/mod to turn a flat index into a tile coordinate,
mask compaction, and several gathers over million-element arrays. Two
kernels collapse all of that into one pass per gaussian, leaving the sort
as the dominant remaining cost.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import _loader
from metalsplat.reference.tiling_ref import (
    DEFAULT_TILE_SIZE,
    TileBinningResult,
    sigma_extent,
)

__all__ = [
    "DEFAULT_TILE_SIZE",
    "TileBinningResult",
    "bin_and_sort_gaussians",
    "bin_and_sort_rects",
]


def _empty(tile_size: int, tiles_x: int, tiles_y: int, device) -> TileBinningResult:
    return TileBinningResult(
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        sorted_gaussian_ids=torch.empty(0, dtype=torch.int32, device=device),
        tile_bins=torch.zeros(tiles_x * tiles_y, 2, dtype=torch.int32, device=device),
    )


@torch.no_grad()
def bin_and_sort_gaussians(
    means2d: torch.Tensor,  # (N, 2)
    depths: torch.Tensor,  # (N,)
    conics: torch.Tensor,  # (N, 3) a, b, c -- inverse 2D covariance
    radii: torch.Tensor,  # (N,)
    valid: torch.Tensor,  # (N,) bool-ish
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    opacities: torch.Tensor | None = None,  # (N,); see tiling_ref.sigma_extent
) -> TileBinningResult:
    """3DGS binning: each gaussian's rectangle is its conic ellipse, out to
    the opacity-aware extent (see tiling_ref.sigma_extent)."""
    device = means2d.device
    n = means2d.shape[0]

    if device.type != "mps":  # CPU/other: fall back to the reference
        from metalsplat.reference.tiling_ref import bin_and_sort_gaussians as ref

        return ref(
            means2d,
            depths,
            conics,
            radii,
            valid,
            img_width,
            img_height,
            tile_size,
            opacities,
        )

    valid_c = _as_float_flags(valid)
    rects = torch.empty(n, 4, dtype=torch.float32, device=device)
    if n > 0:
        _loader.load("tiling").ellipse_rects(
            means2d.contiguous().float(),
            conics.contiguous().float(),
            radii.contiguous().float(),
            valid_c,
            sigma_extent(opacities, n, device).contiguous(),
            rects,
            threads=n,
        )
    return bin_and_sort_rects(rects, depths, valid_c, img_width, img_height, tile_size)


@torch.no_grad()
def bin_and_sort_rects(
    rects: torch.Tensor,  # (N, 4) xmin, ymin, xmax, ymax in pixels; empty if xmax < xmin
    depths: torch.Tensor,  # (N,)
    valid: torch.Tensor,  # (N,) bool-ish
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> TileBinningResult:
    """Bins each valid gaussian into every tile its rectangle touches."""
    device = rects.device
    n = rects.shape[0]
    tiles_x = (img_width + tile_size - 1) // tile_size
    tiles_y = (img_height + tile_size - 1) // tile_size

    if device.type != "mps":  # CPU/other: fall back to the reference
        from metalsplat.reference.tiling_ref import bin_and_sort_rects as ref

        return ref(rects, depths, valid, img_width, img_height, tile_size)

    if n == 0:
        return _empty(tile_size, tiles_x, tiles_y, device)

    rects_c = rects.contiguous().float()
    depths_c = depths.contiguous().float()
    valid_c = _as_float_flags(valid)

    lib = _loader.load("tiling")

    counts = torch.empty(n, dtype=torch.int32, device=device)
    lib.tile_counts(
        rects_c,
        valid_c,
        tiles_x,
        tiles_y,
        float(tile_size),
        counts,
        threads=n,
    )

    # Exclusive prefix sum gives each gaussian the slot its pairs start at.
    # The .item() is a genuine device sync, but unavoidable: the pair buffer's
    # length is data-dependent and has to be known on the host to allocate it.
    inclusive = torch.cumsum(counts, dim=0, dtype=torch.int32)
    total_pairs = int(inclusive[-1].item())
    if total_pairs == 0:
        return _empty(tile_size, tiles_x, tiles_y, device)
    offsets = inclusive - counts

    keys = torch.empty(total_pairs, dtype=torch.int64, device=device)
    gaussian_ids = torch.empty(total_pairs, dtype=torch.int32, device=device)
    lib.tile_pairs(
        rects_c,
        depths_c,
        valid_c,
        offsets,
        tiles_x,
        tiles_y,
        float(tile_size),
        keys,
        gaussian_ids,
        threads=n,
    )

    # Stable so that pairs tying on both tile and depth stay in ascending
    # gaussian order -- which is the order the kernel emits them in, and what
    # the reference produces too, so the two agree exactly rather than
    # only up to a permutation of ties.
    order = torch.argsort(keys, stable=True)
    sorted_gaussian_ids = gaussian_ids[order].contiguous()
    # The tile id is the key's high half; no need to have carried it separately.
    sorted_tile_ids = keys[order] >> 32

    boundaries = torch.arange(tiles_x * tiles_y + 1, device=device)
    tile_starts_ends = torch.searchsorted(sorted_tile_ids, boundaries)
    tile_bins = torch.stack([tile_starts_ends[:-1], tile_starts_ends[1:]], dim=-1).to(
        torch.int32
    )

    return TileBinningResult(
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        sorted_gaussian_ids=sorted_gaussian_ids,
        tile_bins=tile_bins,
    )


def _as_float_flags(valid: torch.Tensor) -> torch.Tensor:
    """`valid` as the contiguous 0.0/1.0 float32 buffer the kernels read."""
    if valid.dtype == torch.bool:
        return valid.to(torch.float32).contiguous()
    return valid.contiguous().float()
