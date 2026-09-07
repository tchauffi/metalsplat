import pytest
import torch

from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.reference.rasterize_ref import rasterize_gaussians as rasterize_gaussians_ref

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

W = H = 32


def _random_scene(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(n, 2, generator=g) * torch.tensor([W, H])
    depths = torch.rand(n, generator=g) * 5.0 + 1.0
    # keep conics well-conditioned (moderate, positive-definite-ish spread)
    raw = torch.rand(n, generator=g) * 0.15 + 0.05
    conics = torch.stack([1.0 / raw, torch.zeros(n), 1.0 / raw], dim=-1)
    opacities = torch.rand(n, generator=g) * 0.6 + 0.3
    colors = torch.rand(n, 3, generator=g)
    radii = torch.full((n,), 8.0)
    valid = torch.ones(n)
    return means2d, depths, conics, opacities, colors, radii, valid


@pytest.mark.parametrize("n", [1, 5, 40])
def test_forward_matches_reference(n):
    means2d, depths, conics, opacities, colors, radii, valid = _random_scene(n)

    ref_image = rasterize_gaussians_ref(
        means2d, depths, conics, opacities, colors, valid, W, H
    )

    kernel_image = rasterize_gaussians(
        means2d.to("mps"), depths.to("mps"), conics.to("mps"), opacities.to("mps"),
        colors.to("mps"), radii.to("mps"), valid.to("mps"), W, H,
    )
    torch.mps.synchronize()

    assert torch.allclose(kernel_image.cpu(), ref_image, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("n", [1, 5, 40])
def test_backward_matches_reference(n):
    means2d, depths, conics, opacities, colors, radii, valid = _random_scene(n)

    means2d_ref = means2d.clone().requires_grad_()
    conics_ref = conics.clone().requires_grad_()
    opacities_ref = opacities.clone().requires_grad_()
    colors_ref = colors.clone().requires_grad_()

    ref_image = rasterize_gaussians_ref(
        means2d_ref, depths, conics_ref, opacities_ref, colors_ref, valid, W, H
    )

    g = torch.Generator().manual_seed(7)
    upstream = torch.randn(H, W, 3, generator=g)
    loss_ref = (ref_image * upstream).sum()
    loss_ref.backward()

    means2d_mps = means2d.to("mps").requires_grad_()
    conics_mps = conics.to("mps").requires_grad_()
    opacities_mps = opacities.to("mps").requires_grad_()
    colors_mps = colors.to("mps").requires_grad_()

    kernel_image = rasterize_gaussians(
        means2d_mps, depths.to("mps"), conics_mps, opacities_mps, colors_mps,
        radii.to("mps"), valid.to("mps"), W, H,
    )
    loss = (kernel_image * upstream.to("mps")).sum()
    loss.backward()
    torch.mps.synchronize()

    assert torch.allclose(means2d_mps.grad.cpu(), means2d_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(conics_mps.grad.cpu(), conics_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(opacities_mps.grad.cpu(), opacities_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(colors_mps.grad.cpu(), colors_ref.grad, atol=2e-2, rtol=2e-2)
