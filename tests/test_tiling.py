import pytest
import torch

from metalsplat.ops.tiling import bin_and_sort_gaussians

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _isotropic_conic(radii: torch.Tensor) -> torch.Tensor:
    """Conic for a circular gaussian whose 3-sigma extent equals `radii`.

    The binner derives its per-axis half-extents from the conic, so tests
    that specify a radius need a conic that agrees with it: sigma = r/3, and
    the conic is the inverse covariance, diag(1/sigma^2).
    """
    sigma = (radii / 3.0).clamp_min(1e-6)
    inv_var = 1.0 / (sigma * sigma)
    return torch.stack([inv_var, torch.zeros_like(inv_var), inv_var], dim=-1)



@pytest.mark.parametrize("device", DEVICES)
def test_single_gaussian_single_tile(device):
    means2d = torch.tensor([[8.0, 8.0]], device=device)
    depths = torch.tensor([1.0], device=device)
    radii = torch.tensor([2.0], device=device)
    valid = torch.tensor([1.0], device=device)

    result = bin_and_sort_gaussians(
        means2d, depths, _isotropic_conic(radii), radii, valid, img_width=32, img_height=32, tile_size=16)

    assert result.tiles_x == 2 and result.tiles_y == 2
    assert result.sorted_gaussian_ids.tolist() == [0]
    lengths = (result.tile_bins[:, 1] - result.tile_bins[:, 0]).tolist()
    assert lengths == [1, 0, 0, 0]
    assert result.tile_bins[0].tolist() == [0, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_spanning_multiple_tiles(device):
    # Centered at the intersection of all 4 tiles in a 2x2 tile grid, with a
    # radius large enough to touch all of them.
    means2d = torch.tensor([[16.0, 16.0]], device=device)
    depths = torch.tensor([1.0], device=device)
    radii = torch.tensor([10.0], device=device)
    valid = torch.tensor([1.0], device=device)

    result = bin_and_sort_gaussians(
        means2d, depths, _isotropic_conic(radii), radii, valid, img_width=32, img_height=32, tile_size=16)

    assert result.sorted_gaussian_ids.numel() == 4  # one entry per touched tile
    touched_tiles = (result.tile_bins[:, 1] - result.tile_bins[:, 0] > 0).nonzero().squeeze(-1)
    assert touched_tiles.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_depth_ordering_within_tile(device):
    means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0]], device=device)
    depths = torch.tensor([5.0, 1.0], device=device)  # gaussian 1 is closer
    radii = torch.tensor([2.0, 2.0], device=device)
    valid = torch.tensor([1.0, 1.0], device=device)

    result = bin_and_sort_gaussians(
        means2d, depths, _isotropic_conic(radii), radii, valid, img_width=16, img_height=16, tile_size=16)

    assert result.sorted_gaussian_ids.tolist() == [1, 0]  # closer gaussian first
    assert result.tile_bins[0].tolist() == [0, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_gaussians_excluded(device):
    means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0]], device=device)
    depths = torch.tensor([1.0, 1.0], device=device)
    radii = torch.tensor([2.0, 0.0], device=device)  # second gaussian culled (radius 0)
    valid = torch.tensor([1.0, 0.0], device=device)

    result = bin_and_sort_gaussians(
        means2d, depths, _isotropic_conic(radii), radii, valid, img_width=16, img_height=16, tile_size=16)

    assert result.sorted_gaussian_ids.tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_gaussians_returns_empty(device):
    means2d = torch.zeros(3, 2, device=device)
    depths = torch.ones(3, device=device)
    radii = torch.zeros(3, device=device)
    valid = torch.zeros(3, device=device)

    result = bin_and_sort_gaussians(
        means2d, depths, _isotropic_conic(radii), radii, valid, img_width=16, img_height=16, tile_size=16)

    assert result.sorted_gaussian_ids.numel() == 0
    assert torch.equal(result.tile_bins, torch.zeros_like(result.tile_bins))
