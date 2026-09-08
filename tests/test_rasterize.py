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


@pytest.mark.parametrize("n", [1, 5, 40])
def test_depth_matches_reference(n):
    means2d, depths, conics, opacities, colors, radii, valid = _random_scene(n)

    _, ref_depth = rasterize_gaussians_ref(
        means2d, depths, conics, opacities, colors, valid, W, H, return_depth=True
    )

    _, kernel_depth, _ = rasterize_gaussians(
        means2d.to("mps"), depths.to("mps"), conics.to("mps"), opacities.to("mps"),
        colors.to("mps"), radii.to("mps"), valid.to("mps"), W, H, return_aux=True,
    )
    torch.mps.synchronize()

    assert torch.allclose(kernel_depth.cpu(), ref_depth, atol=2e-3, rtol=2e-3)


def test_depth_picks_the_nearer_of_two_opaque_gaussians():
    # Two gaussians stacked at the same pixel, one at z=1 and one at z=9,
    # both wide enough (small conic -> large sigma) that alpha at the centre
    # pixel is ~0.99 rather than falling off. Front-to-back compositing then
    # gives the near one nearly all the weight, so expected depth sits close
    # to 1, far below the naive 5.0 midpoint.
    means2d = torch.tensor([[W / 2, H / 2], [W / 2, H / 2]])
    depths = torch.tensor([1.0, 9.0])
    conics = torch.tensor([[0.01, 0.0, 0.01], [0.01, 0.0, 0.01]])
    opacities = torch.tensor([0.99, 0.99])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii = torch.tensor([16.0, 16.0])
    valid = torch.tensor([1.0, 1.0])

    _, depth, final_T = rasterize_gaussians(
        means2d.to("mps"), depths.to("mps"), conics.to("mps"), opacities.to("mps"),
        colors.to("mps"), radii.to("mps"), valid.to("mps"), W, H, return_aux=True,
    )
    torch.mps.synchronize()

    centre_depth = depth[H // 2, W // 2].item()
    centre_alpha = 1.0 - final_T[H // 2, W // 2].item()
    normalised = centre_depth / centre_alpha
    assert 1.0 <= normalised < 1.5  # dominated by the near gaussian


def test_abs_grad_accum_mutates_in_place():
    means2d, depths, conics, opacities, colors, radii, valid = _random_scene(5)
    means2d_mps = means2d.to("mps").requires_grad_()

    abs_accum = torch.zeros(5, device="mps")
    accum_before = abs_accum  # same object, to check in-place semantics

    image = rasterize_gaussians(
        means2d_mps, depths.to("mps"), conics.to("mps"), opacities.to("mps"),
        colors.to("mps"), radii.to("mps"), valid.to("mps"), W, H,
        abs_grad_accum=abs_accum,
    )
    image.sum().backward()
    torch.mps.synchronize()

    assert abs_accum is accum_before
    assert (abs_accum > 0).any()


def test_abs_grad_accum_avoids_sign_cancellation():
    # One large gaussian covering most of the image, so many pixels
    # contribute to its d_means2d. A checkerboard upstream gradient makes
    # neighboring pixels pull its position in opposite directions --
    # exactly the case where the *signed* sum (means2d.grad) cancels but
    # the abs-accumulated sum (AbsGS-style) should not.
    means2d = torch.tensor([[W / 2, H / 2]])
    depths = torch.tensor([1.0])
    conics = torch.tensor([[1.0 / 100.0, 0.0, 1.0 / 100.0]])  # wide spread
    opacities = torch.tensor([0.9])
    colors = torch.tensor([[0.5, 0.5, 0.5]])
    radii = torch.tensor([30.0])
    valid = torch.tensor([1.0])

    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checkerboard = torch.where((xs + ys) % 2 == 0, 1.0, -1.0)
    upstream = checkerboard[:, :, None].expand(H, W, 3).contiguous()

    means2d_mps = means2d.to("mps").requires_grad_()
    abs_accum = torch.zeros(1, device="mps")

    image = rasterize_gaussians(
        means2d_mps, depths.to("mps"), conics.to("mps"), opacities.to("mps"),
        colors.to("mps"), radii.to("mps"), valid.to("mps"), W, H,
        abs_grad_accum=abs_accum,
    )
    (image * upstream.to("mps")).sum().backward()
    torch.mps.synchronize()

    signed_grad_norm = means2d_mps.grad.norm().item()
    abs_accum_value = abs_accum.item()
    assert abs_accum_value > signed_grad_norm * 2  # abs-sum meaningfully exceeds the cancelled signed sum


@pytest.mark.parametrize("n", [300, 800])
def test_backward_matches_reference_with_many_gaussians_per_tile(n):
    """More gaussians in one tile than the backward kernel's batch size.

    The backward stages the tile's gaussians into threadgroup memory 256 at
    a time, so a tile holding more than that runs the batch loop more than
    once. Every other test here puts at most a few dozen gaussians on a
    32x32 image, so the multi-batch path never ran -- while a real scene
    averages several hundred per tile.
    """
    g = torch.Generator().manual_seed(n)
    # All centres inside one 16x16 tile, so they pile into the same bin.
    means2d = torch.rand(n, 2, generator=g) * 12.0 + 2.0
    depths = torch.rand(n, generator=g) * 10.0 + 0.5
    raw = torch.rand(n, generator=g) * 4.0 + 2.0
    conics = torch.stack([1.0 / raw, torch.zeros(n), 1.0 / raw], dim=-1)
    # Low opacities so transmittance survives deep into the list rather than
    # saturating after the first few gaussians.
    opacities = torch.rand(n, generator=g) * 0.05 + 0.01
    colors = torch.rand(n, 3, generator=g)
    radii = torch.full((n,), 8.0)
    valid = torch.ones(n)

    means2d_ref = means2d.clone().requires_grad_()
    conics_ref = conics.clone().requires_grad_()
    opacities_ref = opacities.clone().requires_grad_()
    colors_ref = colors.clone().requires_grad_()
    img_ref = rasterize_gaussians_ref(
        means2d_ref, depths, conics_ref, opacities_ref, colors_ref, valid, W, H
    )
    grad_out = torch.rand(*img_ref.shape, generator=g)
    (img_ref * grad_out).sum().backward()

    means2d_mps = means2d.to("mps").requires_grad_()
    conics_mps = conics.to("mps").requires_grad_()
    opacities_mps = opacities.to("mps").requires_grad_()
    colors_mps = colors.to("mps").requires_grad_()
    img = rasterize_gaussians(
        means2d_mps, depths.to("mps"), conics_mps, opacities_mps, colors_mps,
        radii.to("mps"), valid.to("mps"), W, H,
    )
    (img * grad_out.to("mps")).sum().backward()
    torch.mps.synchronize()

    assert torch.allclose(img.cpu(), img_ref, atol=1e-4), "forward disagrees"
    for name, got, want in (
        ("colors", colors_mps.grad, colors_ref.grad),
        ("opacities", opacities_mps.grad, opacities_ref.grad),
        ("conics", conics_mps.grad, conics_ref.grad),
        ("means2d", means2d_mps.grad, means2d_ref.grad),
    ):
        assert torch.allclose(got.cpu(), want, atol=2e-3, rtol=2e-2), f"{name} gradient disagrees"
