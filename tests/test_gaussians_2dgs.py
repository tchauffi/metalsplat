import pytest
import torch

from metalsplat.gaussians_2dgs import Gaussian2DModel


def _means(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g)


def test_defaults():
    model = Gaussian2DModel(_means(5))
    assert model.scales.shape == (5, 2)
    assert torch.all(model.scales > 0)
    assert torch.allclose(model.quats.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(model.opacities, torch.full((5,), 0.5), atol=1e-4)
    assert torch.allclose(model.colors, torch.full((5, 3), 0.5), atol=1e-4)
    assert model.num_points == 5


def test_normals_orthonormal_frame():
    n = 8
    g = torch.Generator().manual_seed(1)
    quats = torch.randn(n, 4, generator=g)
    model = Gaussian2DModel(_means(n), quats=quats)
    rotmat = model.rotmat
    # columns are an orthonormal frame: t_u, t_v, normal
    identity = torch.eye(3).unsqueeze(0).expand(n, -1, -1)
    assert torch.allclose(rotmat.transpose(-1, -2) @ rotmat, identity, atol=1e-5)
    assert torch.allclose(model.normals, rotmat[..., :, 2])
    assert torch.allclose(model.normals.norm(dim=-1), torch.ones(n), atol=1e-5)


def test_sh_degree0_color_raises_colors_from_view():
    model = Gaussian2DModel(_means(3))
    with pytest.raises(AttributeError):
        model.colors_from_view(torch.zeros(3, 3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sh_degree_view_dependent_color():
    n = 4
    model = Gaussian2DModel(_means(n).to("mps"), sh_degree=2)
    with pytest.raises(AttributeError):
        _ = model.colors
    view_dirs = torch.nn.functional.normalize(torch.randn(n, 3, device="mps"), dim=-1)
    colors = model.colors_from_view(view_dirs)
    assert colors.shape == (n, 3)


def test_increase_sh_degree_caps_at_sh_degree():
    model = Gaussian2DModel(_means(2), sh_degree=2, active_sh_degree=0)
    assert model.increase_sh_degree() == 1
    assert model.increase_sh_degree() == 2
    assert model.increase_sh_degree() == 2


def test_random_classmethod():
    model = Gaussian2DModel.random(10, bound=2.0)
    assert model.num_points == 10
    assert model.means.abs().max() <= 2.0 + 1e-5
