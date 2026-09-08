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

def _isotropic_conic(radii: torch.Tensor) -> torch.Tensor:
    """Conic for a circular gaussian whose 3-sigma extent equals `radii`.

    The binner derives its per-axis half-extents from the conic, so tests
    that specify a radius need a conic that agrees with it: sigma = r/3, and
    the conic is the inverse covariance, diag(1/sigma^2).
    """
    sigma = (radii / 3.0).clamp_min(1e-6)
    inv_var = 1.0 / (sigma * sigma)
    return torch.stack([inv_var, torch.zeros_like(inv_var), inv_var], dim=-1)



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
    conics = _isotropic_conic(radii)
    ref = bin_ref(means2d, depths, conics, radii, valid, W, H, tile_size)
    got = bin_and_sort_gaussians(
        means2d.to("mps"), depths.to("mps"), conics.to("mps"), radii.to("mps"),
        valid.to("mps"), W, H, tile_size,
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
        means2d.to("mps"), depths.to("mps"), _isotropic_conic(radii).to("mps"),
        radii.to("mps"), valid.to("mps"), W, H
    )
    torch.mps.synchronize()
    start, end = got.tile_bins[0].tolist()
    assert got.sorted_gaussian_ids[start:end].cpu().tolist() == [1, 2, 0]


def test_empty_input():
    got = bin_and_sort_gaussians(
        torch.empty(0, 2, device="mps"), torch.empty(0, device="mps"),
        torch.empty(0, 3, device="mps"), torch.empty(0, device="mps"),
        torch.empty(0, device="mps"), W, H,
    )
    assert got.sorted_gaussian_ids.numel() == 0
    assert got.tile_bins.shape == (got.tiles_x * got.tiles_y, 2)
    assert int(got.tile_bins.sum()) == 0


def _anisotropic_scene(n, seed=0):
    """Elongated, arbitrarily-oriented gaussians -- the case the tight box exists for."""
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([W * 1.2, H * 1.2]) - torch.tensor(
        [W * 0.1, H * 0.1]
    )
    depths = torch.rand(n, generator=g) * 10.0 + 0.5
    # Random covariance: rotate a strongly anisotropic diagonal.
    sx = torch.rand(n, generator=g) * 20.0 + 1.0
    sy = torch.rand(n, generator=g) * 2.0 + 0.5
    theta = torch.rand(n, generator=g) * 3.14159
    ct, st = theta.cos(), theta.sin()
    vxx = ct * ct * sx**2 + st * st * sy**2
    vyy = st * st * sx**2 + ct * ct * sy**2
    vxy = ct * st * (sx**2 - sy**2)
    det = vxx * vyy - vxy * vxy
    conics = torch.stack([vyy / det, -vxy / det, vxx / det], dim=-1)  # inverse covariance
    lam = 0.5 * (vxx + vyy) + (((vxx - vyy) * 0.5) ** 2 + vxy**2).sqrt()
    radii = torch.ceil(3.0 * lam.sqrt())  # circumscribed circle, as project computes it
    valid = torch.ones(n)
    return means2d, depths, conics, radii, valid


def test_anisotropic_gaussians_match_the_reference():
    means2d, depths, conics, radii, valid = _anisotropic_scene(400, seed=5)
    ref = bin_ref(means2d, depths, conics, radii, valid, W, H, 16)
    got = bin_and_sort_gaussians(
        means2d.to("mps"), depths.to("mps"), conics.to("mps"), radii.to("mps"),
        valid.to("mps"), W, H, 16,
    )
    torch.mps.synchronize()

    assert ref.sorted_gaussian_ids.numel() > 0
    assert torch.equal(got.sorted_gaussian_ids.cpu(), ref.sorted_gaussian_ids)
    assert torch.equal(got.tile_bins.cpu(), ref.tile_bins)


def test_tight_box_emits_fewer_pairs_than_the_circumscribed_circle():
    # The whole point of binning from the conic rather than a radius: a long
    # thin splat should not claim a square region as wide as it is long.
    means2d, depths, conics, radii, valid = _anisotropic_scene(400, seed=6)

    tight = bin_ref(means2d, depths, conics, radii, valid, W, H, 16)
    # An isotropic conic of the same 3-sigma radius reproduces the old
    # circumscribed-circle behaviour exactly.
    loose = bin_ref(means2d, depths, _isotropic_conic(radii), radii, valid, W, H, 16)

    assert tight.sorted_gaussian_ids.numel() < loose.sorted_gaussian_ids.numel()
    ratio = tight.sorted_gaussian_ids.numel() / loose.sorted_gaussian_ids.numel()
    assert ratio < 0.9, f"tight box saved only {100 * (1 - ratio):.1f}%"


def test_tight_box_still_covers_every_tile_the_ellipse_reaches():
    # Tighter must not mean wrong: every tile whose area intersects the
    # gaussian's 3-sigma axis-aligned extent must still be emitted.
    means2d = torch.tensor([[100.0, 65.0]])
    depths = torch.tensor([1.0])
    # sigma_x = 20, sigma_y = 2, axis-aligned: 3-sigma box is 120 x 12 px.
    conics = torch.tensor([[1.0 / 400.0, 0.0, 1.0 / 4.0]])
    radii = torch.tensor([60.0])
    valid = torch.ones(1)

    res = bin_ref(means2d, depths, conics, radii, valid, W, H, 16)
    touched = {
        i for i in range(res.tiles_x * res.tiles_y)
        if res.tile_bins[i, 1] > res.tile_bins[i, 0]
    }
    expected = {
        ty * res.tiles_x + tx
        for tx in range(int((100 - 60) // 16), int((100 + 60) // 16) + 1)
        for ty in range(int((65 - 6) // 16), int((65 + 6) // 16) + 1)
        if 0 <= tx < res.tiles_x and 0 <= ty < res.tiles_y
    }
    assert touched == expected
