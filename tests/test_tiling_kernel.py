"""The Metal tile-binning kernels against the pure-PyTorch reference.

The two must agree *exactly*, not approximately: these outputs are indices
and ranges, so a single element out of place silently composites the wrong
gaussians into a tile.
"""

import pytest
import torch

from metalsplat.ops.tiling import bin_and_sort_gaussians
from metalsplat.reference.tiling_ref import bin_and_sort_gaussians as bin_ref

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

W, H = 200, 130  # deliberately not a multiple of the tile size


def _scene(n, seed=0, max_radius=30.0):
    g = torch.Generator().manual_seed(seed)
    # Spread centres beyond the image on both sides so off-screen culling,
    # partial overlap and full coverage are all exercised.
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([W * 1.6, H * 1.6]) - torch.tensor(
        [W * 0.3, H * 0.3]
    )
    depths = torch.rand(n, generator=g) * 10.0 + 0.5
    radii = (torch.rand(n, generator=g) * max_radius).floor()
    valid = (torch.rand(n, generator=g) > 0.2).float()
    return means2d, depths, radii, valid


def _compare(means2d, depths, radii, valid, tile_size=16):
    ref = bin_ref(means2d, depths, radii, valid, W, H, tile_size)
    got = bin_and_sort_gaussians(
        means2d.to("mps"), depths.to("mps"), radii.to("mps"), valid.to("mps"),
        W, H, tile_size,
    )
    torch.mps.synchronize()

    assert got.tiles_x == ref.tiles_x and got.tiles_y == ref.tiles_y
    assert got.sorted_gaussian_ids.shape == ref.sorted_gaussian_ids.shape
    assert torch.equal(got.sorted_gaussian_ids.cpu(), ref.sorted_gaussian_ids)
    assert torch.equal(got.tile_bins.cpu(), ref.tile_bins)
    return ref


@pytest.mark.parametrize("n", [1, 17, 500, 5000])
def test_matches_reference(n):
    ref = _compare(*_scene(n, seed=n))
    # n=1 may legitimately cull its only gaussian; the larger cases must not
    # be silently comparing two empty results.
    if n > 1:
        assert ref.sorted_gaussian_ids.numel() > 0


@pytest.mark.parametrize("tile_size", [8, 16, 32])
def test_matches_reference_across_tile_sizes(tile_size):
    _compare(*_scene(800, seed=7), tile_size=tile_size)


def test_matches_reference_with_large_radii():
    # Radii big enough that single gaussians blanket most of the tile grid,
    # which is where the per-gaussian write loop does the most work.
    _compare(*_scene(200, seed=3, max_radius=180.0))


def test_all_invalid_returns_empty():
    means2d, depths, radii, _ = _scene(64, seed=11)
    valid = torch.zeros(64)
    ref = _compare(means2d, depths, radii, valid)
    assert ref.sorted_gaussian_ids.numel() == 0


def test_zero_radius_gaussians_are_culled():
    means2d, depths, _, valid = _scene(64, seed=12)
    radii = torch.zeros(64)
    ref = _compare(means2d, depths, radii, valid)
    assert ref.sorted_gaussian_ids.numel() == 0


def test_fully_offscreen_gaussians_are_culled():
    # Centres far off the left/top with radii too small to reach the image.
    means2d = torch.tensor([[-500.0, -500.0], [W + 500.0, H + 500.0]])
    depths = torch.tensor([1.0, 2.0])
    radii = torch.tensor([4.0, 4.0])
    valid = torch.ones(2)
    ref = _compare(means2d, depths, radii, valid)
    assert ref.sorted_gaussian_ids.numel() == 0


def test_depth_order_within_a_tile_is_front_to_back():
    # Three gaussians on the same tile, deliberately supplied back-to-front.
    means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]])
    depths = torch.tensor([5.0, 1.0, 3.0])
    radii = torch.tensor([2.0, 2.0, 2.0])
    valid = torch.ones(3)

    got = bin_and_sort_gaussians(
        means2d.to("mps"), depths.to("mps"), radii.to("mps"), valid.to("mps"), W, H
    )
    torch.mps.synchronize()
    start, end = got.tile_bins[0].tolist()
    assert got.sorted_gaussian_ids[start:end].cpu().tolist() == [1, 2, 0]


def test_empty_input():
    got = bin_and_sort_gaussians(
        torch.empty(0, 2, device="mps"), torch.empty(0, device="mps"),
        torch.empty(0, device="mps"), torch.empty(0, device="mps"), W, H,
    )
    assert got.sorted_gaussian_ids.numel() == 0
    assert got.tile_bins.shape == (got.tiles_x * got.tiles_y, 2)
    assert int(got.tile_bins.sum()) == 0
