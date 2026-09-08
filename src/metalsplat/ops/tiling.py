"""Tile binning and sorting for the rasterizer, using Metal kernels for the
per-pair work and torch's sort for the ordering.

Given projected gaussians (means2d, depths, radii, valid), this builds:
- a flat list of (gaussian_id, tile_id) pairs, one per tile a gaussian's
  dilated bounding box touches, sorted by (tile_id, depth) so that within
  a tile the gaussians are front-to-back;
- per-tile [start, end) index ranges into that sorted list.

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
from metalsplat.reference.tiling_ref import DEFAULT_TILE_SIZE, TileBinningResult

__all__ = ["DEFAULT_TILE_SIZE", "TileBinningResult", "bin_and_sort_gaussians"]


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
    radii: torch.Tensor,  # (N,)
    valid: torch.Tensor,  # (N,) bool-ish
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> TileBinningResult:
    device = means2d.device
    n = means2d.shape[0]
    tiles_x = (img_width + tile_size - 1) // tile_size
    tiles_y = (img_height + tile_size - 1) // tile_size

    if device.type != "mps":  # CPU/other: fall back to the reference
        from metalsplat.reference.tiling_ref import bin_and_sort_gaussians as ref

        return ref(means2d, depths, radii, valid, img_width, img_height, tile_size)

    if n == 0:
        return _empty(tile_size, tiles_x, tiles_y, device)

    means2d_c = means2d.contiguous().float()
    depths_c = depths.contiguous().float()
    radii_c = radii.contiguous().float()
    valid_c = (valid > 0.5).to(torch.float32).contiguous() if valid.dtype == torch.bool else valid.contiguous().float()

    lib = _loader.load("tiling")

    counts = torch.empty(n, dtype=torch.int32, device=device)
    lib.tile_counts(
        means2d_c, radii_c, valid_c, tiles_x, tiles_y, float(tile_size), counts, threads=n
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
        means2d_c, depths_c, radii_c, valid_c, offsets,
        tiles_x, tiles_y, float(tile_size),
        keys, gaussian_ids, threads=n,
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
    tile_bins = torch.stack([tile_starts_ends[:-1], tile_starts_ends[1:]], dim=-1).to(torch.int32)

    return TileBinningResult(
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        sorted_gaussian_ids=sorted_gaussian_ids,
        tile_bins=tile_bins,
    )
