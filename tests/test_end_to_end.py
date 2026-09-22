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
    camera = Camera.identity(
        fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H
    ).to(device)
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
    assert (
        model.raw_scales.grad is not None
        and torch.isfinite(model.raw_scales.grad).all()
    )
    assert (
        model.raw_quats.grad is not None and torch.isfinite(model.raw_quats.grad).all()
    )
    assert (
        model.raw_opacities.grad is not None
        and torch.isfinite(model.raw_opacities.grad).all()
    )
    assert (
        model.raw_colors.grad is not None
        and torch.isfinite(model.raw_colors.grad).all()
    )


def test_sh_render_and_gradients_flow_end_to_end():
    torch.manual_seed(0)
    n = 20
    device = "mps"
    means = torch.rand(n, 3, device=device) * 2 - 1
    means[:, 2] = means[:, 2].abs() + 2.0
    model = GaussianModel(
        means, colors=torch.rand(n, 3, device=device), sh_degree=2
    ).to(device)
    camera = Camera.identity(
        fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H
    ).to(device)

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


def test_render_aux_works_under_no_grad():
    # Inference-mode aux (depth / coverage) must not require a graph.
    model, camera = _scene("mps")
    with torch.no_grad():
        aux = render(model, camera, return_aux=True)
        torch.mps.synchronize()

    assert aux.image.shape == (H, W, 3)
    assert aux.depth.shape == (H, W)
    assert aux.final_T.shape == (H, W)
    assert torch.isfinite(aux.depth).all()
    assert (aux.depth >= 0).all()


def test_near_fade_suppresses_gaussians_inside_the_sphere():
    model, camera = _scene("mps")
    # _scene puts everything at z in [2, 3] straight ahead, so a band that
    # ends beyond the far end of the scene fades every gaussian to nothing.
    full = render(model, camera)
    faded = render(model, camera, near_fade=(4.0, 5.0))
    torch.mps.synchronize()

    assert full.abs().sum() > 0
    assert torch.allclose(faded, torch.zeros_like(faded), atol=1e-6)


def test_near_fade_leaves_distant_gaussians_untouched():
    model, camera = _scene("mps")
    full = render(model, camera)
    # Band entirely inside the empty space between camera and scene.
    faded = render(model, camera, near_fade=(0.1, 0.5))
    torch.mps.synchronize()

    assert torch.allclose(faded, full, atol=1e-6)


def test_near_fade_is_gradual_across_the_band():
    # The band is the whole point: a hard cut pops as the camera crosses a
    # gaussian. Partway through, the render must sit strictly between the
    # unfaded and fully-faded extremes rather than snapping to one of them.
    model, camera = _scene("mps")
    full = render(model, camera)
    partial = render(model, camera, near_fade=(2.0, 4.0))
    torch.mps.synchronize()

    energy_full = full.abs().sum().item()
    energy_partial = partial.abs().sum().item()
    assert 0.0 < energy_partial < energy_full


def test_near_fade_keeps_gradients_flowing():
    model, camera = _scene("mps")
    image = render(model, camera, near_fade=(1.0, 2.5))
    image.pow(2).mean().backward()
    torch.mps.synchronize()

    assert model.raw_opacities.grad is not None
    assert torch.isfinite(model.raw_opacities.grad).all()
    assert model.means.grad is not None and torch.isfinite(model.means.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_opaque_gaussians_just_outside_the_frame_still_reach_it():
    # Opaque gaussians centred a little left of the image, at increasing
    # distances. Their alpha stays above 1/255 out to ~3.33 sigma, so the
    # nearer ones still colour the first columns -- but a 3-sigma culling
    # radius in projection dropped them, cutting the image border clean.
    # Compared against a brute-force rasterization of the same projected
    # gaussians with no culling at all.
    from metalsplat.ops.project import project_gaussians
    from metalsplat.reference.project_ref import (
        project_gaussians as project_ref,
    )
    from metalsplat.reference.rasterize_ref import rasterize_gaussians as raster_ref

    size, f, z = 128, 200.0, 2.0
    k = 16
    # Fine steps: the gap between a 3-sigma radius (rounded up) and the true
    # reach is under a pixel wide.
    u = torch.linspace(-14.0, -18.0, k)  # pixel x of each centre, off-frame
    v = torch.arange(k) * 8.0 + 4.0
    depth = z + torch.arange(k) * 1e-3  # distinct, for a stable order
    means = torch.stack(
        [(u - size / 2) * depth / f, (v - size / 2) * depth / f, depth], dim=-1
    )
    model = GaussianModel(
        means,
        scales=torch.full((k, 3), 0.05),
        opacities=torch.full((k,), 0.99),
        colors=torch.ones(k, 3),
    ).to("mps")
    camera = Camera.identity(
        fx=f, fy=f, cx=size / 2, cy=size / 2, img_width=size, img_height=size
    ).to("mps")

    with torch.no_grad():
        image = render(model, camera, antialias=False).cpu()
        proj = project_gaussians(
            model.means,
            model.scales,
            model.quats,
            camera.R_wc,
            camera.t_wc,
            f,
            f,
            size / 2,
            size / 2,
            size,
            size,
        )
        means2d, depths, conics = proj[0].cpu(), proj[1].cpu(), proj[2].cpu()
        expected = raster_ref(
            means2d,
            depths,
            conics,
            model.opacities.cpu(),
            torch.ones(k, 3),
            torch.ones(k),
            size,
            size,
        )
        old_valid = project_ref(
            model.means.cpu(),
            model.scales.cpu(),
            model.quats.cpu(),
            camera.R_wc.cpu(),
            camera.t_wc.cpu(),
            f,
            f,
            size / 2,
            size / 2,
            size,
            size,
            radius_sigmas=3.0,
        ).valid

    # Which gaussians put alpha >= 1/255 on the nearest first-column pixel.
    d = torch.stack([0.5 - means2d[:, 0], (v.floor() + 0.5) - means2d[:, 1]], -1)
    a, b, c = conics.unbind(-1)
    power = -0.5 * (a * d[:, 0] ** 2 + 2 * b * d[:, 0] * d[:, 1] + c * d[:, 1] ** 2)
    reaches = 0.99 * torch.exp(power) >= 1.0 / 255.0
    # The case is only meaningful if some gaussian the old radius culled
    # really does reach the frame.
    assert (reaches & ~old_valid).any()
    assert torch.allclose(image, expected, atol=1e-5)
