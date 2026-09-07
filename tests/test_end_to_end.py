import pytest
import torch

from metalsplat import Camera, GaussianModel, render

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

W = H = 32


def _scene(device):
    torch.manual_seed(0)
    n = 20
    means = torch.rand(n, 3, device=device) * 2 - 1
    means[:, 2] = means[:, 2].abs() + 2.0
    model = GaussianModel(means, colors=torch.rand(n, 3, device=device)).to(device)
    camera = Camera.identity(fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H).to(device)
    return model, camera


def test_render_shape_and_range():
    model, camera = _scene("mps")
    image = render(model, camera)
    torch.mps.synchronize()

    assert image.shape == (H, W, 3)
    assert torch.isfinite(image).all()


def test_gradients_flow_end_to_end():
    model, camera = _scene("mps")
    image = render(model, camera)
    loss = image.pow(2).mean()
    loss.backward()
    torch.mps.synchronize()

    assert model.means.grad is not None and torch.isfinite(model.means.grad).all()
    assert model.raw_scales.grad is not None and torch.isfinite(model.raw_scales.grad).all()
    assert model.raw_quats.grad is not None and torch.isfinite(model.raw_quats.grad).all()
    assert model.raw_opacities.grad is not None and torch.isfinite(model.raw_opacities.grad).all()
    assert model.raw_colors.grad is not None and torch.isfinite(model.raw_colors.grad).all()


def test_sh_render_and_gradients_flow_end_to_end():
    torch.manual_seed(0)
    n = 20
    device = "mps"
    means = torch.rand(n, 3, device=device) * 2 - 1
    means[:, 2] = means[:, 2].abs() + 2.0
    model = GaussianModel(means, colors=torch.rand(n, 3, device=device), sh_degree=2).to(device)
    camera = Camera.identity(fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H).to(device)

    image = render(model, camera)
    assert image.shape == (H, W, 3)
    assert torch.isfinite(image).all()

    loss = image.pow(2).mean()
    loss.backward()
    torch.mps.synchronize()

    assert model.raw_sh.grad is not None and torch.isfinite(model.raw_sh.grad).all()
    # non-DC coefficients should get real (nonzero) gradient too, not just the DC term
    assert model.raw_sh.grad[:, 1:, :].abs().sum() > 0
    assert model.means.grad is not None and torch.isfinite(model.means.grad).all()
