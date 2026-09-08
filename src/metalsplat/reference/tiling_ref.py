"""Pure-PyTorch tile binning and sorting: the executable spec and numerical
oracle for the Metal implementation in metalsplat.ops.tiling.

Kept as the reference rather than deleted -- it is what the kernel version
is tested against, and it runs on any device (including CPU), so it doubles
as a fallback backend.

Given projected gaussians (means2d, depths, radii, valid), this builds:
- a flat list of (gaussian_id, tile_id) pairs, one per tile a gaussian's
  dilated bounding box touches, sorted by (tile_id, depth) so that within
  a tile the gaussians are front-to-back;
- per-tile [start, end) index ranges into that sorted list.

This whole stage is non-differentiable (radii/tile membership are discrete/
structural, matching gsplat), so it deliberately operates outside autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_TILE_SIZE = 16


@dataclass
class TileBinningResult:
    tile_size: int
    tiles_x: int
    tiles_y: int
    sorted_gaussian_ids: torch.Tensor  # (M,) int32, M = total (gaussian, tile) pairs
    tile_bins: torch.Tensor  # (tiles_x * tiles_y, 2) int32, [start, end) into sorted_gaussian_ids


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
) -> TileBinningResult:
    device = means2d.device
    n = means2d.shape[0]
    tiles_x = (img_width + tile_size - 1) // tile_size
    tiles_y = (img_height + tile_size - 1) // tile_size
    num_tiles = tiles_x * tiles_y

    valid_mask = valid > 0.5 if valid.dtype != torch.bool else valid
    valid_mask = valid_mask & (radii > 0)

    if n == 0 or not bool(valid_mask.any()):
        return TileBinningResult(
            tile_size=tile_size,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            sorted_gaussian_ids=torch.empty(0, dtype=torch.int32, device=device),
            tile_bins=torch.zeros(num_tiles, 2, dtype=torch.int32, device=device),
        )

    idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # (K,)
    means2d_v = means2d[idx]
    depths_v = depths[idx]
    radii_v = radii[idx]

    # Per-axis 3-sigma half-extents from the conic (the inverse 2D
    # covariance), not a circle of radius 3*sqrt(lambda_max). Bounding an
    # elongated gaussian by its circumscribed circle gives a box as wide as
    # the splat is long; on the garden scene the tight box produces 43%
    # fewer (gaussian, tile) pairs.
    conics_v = conics[idx]
    det = conics_v[:, 0] * conics_v[:, 2] - conics_v[:, 1] * conics_v[:, 1]
    det_safe = det.clamp_min(1e-12)
    half_w = 3.0 * (conics_v[:, 2] / det_safe).clamp_min(0.0).sqrt()
    half_h = 3.0 * (conics_v[:, 0] / det_safe).clamp_min(0.0).sqrt()
    degenerate = det <= 0
    half_w = torch.where(degenerate, torch.zeros_like(half_w), half_w)
    half_h = torch.where(degenerate, torch.zeros_like(half_h), half_h)

    min_tx = torch.clamp(((means2d_v[:, 0] - half_w) / tile_size).floor().long(), min=0)
    max_tx = torch.clamp(((means2d_v[:, 0] + half_w) / tile_size).floor().long(), max=tiles_x - 1)
    min_ty = torch.clamp(((means2d_v[:, 1] - half_h) / tile_size).floor().long(), min=0)
    max_ty = torch.clamp(((means2d_v[:, 1] + half_h) / tile_size).floor().long(), max=tiles_y - 1)

    tiles_touched_x = (max_tx - min_tx + 1).clamp(min=0)
    tiles_touched_y = (max_ty - min_ty + 1).clamp(min=0)
    counts = tiles_touched_x * tiles_touched_y  # (K,)

    keep = counts > 0
    if not bool(keep.any()):
        return TileBinningResult(
            tile_size=tile_size,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            sorted_gaussian_ids=torch.empty(0, dtype=torch.int32, device=device),
            tile_bins=torch.zeros(num_tiles, 2, dtype=torch.int32, device=device),
        )
    idx = idx[keep]
    depths_v = depths_v[keep]
    min_tx, max_tx = min_tx[keep], max_tx[keep]
    min_ty, max_ty = min_ty[keep], max_ty[keep]
    tiles_touched_x, tiles_touched_y = tiles_touched_x[keep], tiles_touched_y[keep]
    counts = counts[keep]

    total_pairs = int(counts.sum().item())
    offsets = torch.cumsum(counts, dim=0) - counts  # exclusive prefix sum, (K,)

    # For each surviving gaussian k, emit `counts[k]` (gaussian, tile) pairs
    # by expanding its local tile-grid row-major. `local_i` is the pair's
    # position within its own gaussian's block of `counts[k]` entries.
    pair_gaussian_slot = torch.repeat_interleave(
        torch.arange(idx.shape[0], device=device), counts
    )  # (total_pairs,), indexes into the *kept* (idx/depths_v/...) arrays
    local_i = torch.arange(total_pairs, device=device) - offsets[pair_gaussian_slot]

    row_span = tiles_touched_x[pair_gaussian_slot]
    local_row = local_i // row_span
    local_col = local_i % row_span

    tile_x = min_tx[pair_gaussian_slot] + local_col
    tile_y = min_ty[pair_gaussian_slot] + local_row
    tile_id = tile_y * tiles_x + tile_x

    gaussian_id = idx[pair_gaussian_slot]
    depth = depths_v[pair_gaussian_slot]

    # Single sortable key: (tile_id << 32) | depth_bits. Safe because valid
    # gaussians always have depth > near > 0, so the raw float32 bit pattern
    # already orders correctly as an unsigned integer.
    depth_bits = depth.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = (tile_id.to(torch.int64) << 32) | depth_bits

    # stable so that gaussians tying on (tile, depth) keep ascending-id
    # order, which is what the Metal implementation emits by construction.
    order = torch.argsort(key, stable=True)
    sorted_gaussian_ids = gaussian_id[order].to(torch.int32).contiguous()
    sorted_tile_ids = tile_id[order]

    boundaries = torch.arange(num_tiles + 1, device=device)
    tile_starts_ends = torch.searchsorted(sorted_tile_ids, boundaries)
    tile_bins = torch.stack([tile_starts_ends[:-1], tile_starts_ends[1:]], dim=-1).to(torch.int32)

    return TileBinningResult(
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        sorted_gaussian_ids=sorted_gaussian_ids,
        tile_bins=tile_bins,
    )
