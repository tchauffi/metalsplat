import torch

from metalsplat.reference.rasterize_ref import rasterize_gaussians


def test_single_opaque_gaussian_at_center_is_close_to_its_color():
    means2d = torch.tensor([[8.5, 8.5]])  # aligned with the (8, 8) pixel's center
    depths = torch.tensor([1.0])
    conics = torch.tensor([[1.0, 0.0, 1.0]])  # tight, near-isotropic
    opacities = torch.tensor([0.99])
    colors = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([1.0])

    image = rasterize_gaussians(
        means2d, depths, conics, opacities, colors, valid, 16, 16
    )

    center = image[8, 8]
    assert torch.allclose(center, torch.tensor([0.99, 0.0, 0.0]), atol=1e-3)
    corner = image[0, 0]
    assert corner[0] < center[0]  # falls off away from the gaussian center


def test_no_gaussians_shows_background():
    means2d = torch.zeros(0, 2)
    depths = torch.zeros(0)
    conics = torch.zeros(0, 3)
    opacities = torch.zeros(0)
    colors = torch.zeros(0, 3)
    valid = torch.zeros(0)
    background = torch.tensor([0.2, 0.3, 0.4])

    image = rasterize_gaussians(
        means2d, depths, conics, opacities, colors, valid, 4, 4, background=background
    )

    assert torch.allclose(image, background.expand(4, 4, 3))


def test_farther_gaussian_occluded_by_closer_opaque_one():
    means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0]])
    depths = torch.tensor([5.0, 1.0])  # gaussian 1 is closer
    conics = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    opacities = torch.tensor([0.99, 0.99])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    valid = torch.tensor([1.0, 1.0])

    image = rasterize_gaussians(
        means2d, depths, conics, opacities, colors, valid, 16, 16
    )

    center = image[8, 8]
    assert center[1] > center[0]  # closer green gaussian dominates
