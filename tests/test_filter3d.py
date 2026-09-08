"""Mip-Splatting's 3D smoothing filter."""

import math

import pytest
import torch

from metalsplat.camera import Camera
from metalsplat.filter3d import apply_3d_filter, compute_3d_filter
from metalsplat.gaussians import GaussianModel

W = H = 64
FX = FY = 50.0


def _camera_at(z_offset: float) -> Camera:
    # Looks down +z from the origin, shifted back so a point at the world
    # origin sits `z_offset` away.
    cam = Camera.identity(fx=FX, fy=FY, cx=W / 2, cy=H / 2, img_width=W, img_height=H)
    return Camera(
        R_wc=cam.R_wc, t_wc=torch.tensor([0.0, 0.0, z_offset]),
        fx=FX, fy=FY, cx=W / 2, cy=H / 2, img_width=W, img_height=H,
    )


def test_filter_radius_follows_the_closest_observing_camera():
    means = torch.zeros(1, 3)
    near_cam, far_cam = _camera_at(2.0), _camera_at(10.0)

    r_near = compute_3d_filter(means, [near_cam], sampling_scale=0.2)
    r_far = compute_3d_filter(means, [far_cam], sampling_scale=0.2)
    r_both = compute_3d_filter(means, [near_cam, far_cam], sampling_scale=0.2)

    assert r_near.item() == pytest.approx(0.2 * 2.0 / FX, rel=1e-5)
    assert r_far.item() == pytest.approx(0.2 * 10.0 / FX, rel=1e-5)
    # A closer view resolves finer detail, so it wins.
    assert r_both.item() == pytest.approx(r_near.item(), rel=1e-5)
    assert r_near < r_far


def test_unseen_gaussians_are_not_filtered():
    # Behind the camera, and far off to the side: no observation exists, so
    # there is no sampling rate to band-limit with.
    means = torch.tensor([[0.0, 0.0, -5.0], [500.0, 0.0, 3.0]])
    r = compute_3d_filter(means, [_camera_at(0.0)], sampling_scale=0.2)
    assert torch.equal(r, torch.zeros(2))


def test_filter_scales_with_sampling_scale():
    means = torch.zeros(1, 3)
    cam = [_camera_at(4.0)]
    assert compute_3d_filter(means, cam, sampling_scale=0.4).item() == pytest.approx(
        2 * compute_3d_filter(means, cam, sampling_scale=0.2).item(), rel=1e-5
    )


def test_apply_dilates_scales_and_preserves_total_energy():
    scales = torch.tensor([[0.01, 0.02, 0.03], [1.0, 1.0, 1.0]])
    opacities = torch.tensor([0.8, 0.8])
    r = torch.tensor([0.05, 0.05])

    dilated, compensated = apply_3d_filter(scales, opacities, r)

    # Every axis grows to sqrt(s^2 + r^2), never shrinks.
    expected = (scales**2 + 0.05**2).sqrt()
    assert torch.allclose(dilated, expected)
    assert (dilated >= scales).all()

    # opacity * det(Sigma')^(1/2) is unchanged: the filter spreads the
    # gaussian without making it denser.
    energy_before = opacities * scales.prod(dim=-1)
    energy_after = compensated * dilated.prod(dim=-1)
    assert torch.allclose(energy_before, energy_after, atol=1e-7)


def test_a_gaussian_far_below_the_sampling_rate_is_suppressed():
    # The point of the filter: sub-sampling-rate detail was never observed,
    # so it should not survive as a sharp, opaque gaussian.
    scales = torch.full((1, 3), 1e-4)
    opacities = torch.tensor([0.9])
    r = torch.tensor([0.05])

    _, compensated = apply_3d_filter(scales, opacities, r)
    assert compensated.item() < 1e-6


def test_zero_filter_is_a_no_op():
    scales = torch.rand(5, 3) + 0.1
    opacities = torch.rand(5)
    dilated, compensated = apply_3d_filter(scales, opacities, torch.zeros(5))

    assert torch.allclose(dilated, scales, atol=1e-7)
    assert torch.allclose(compensated, opacities, atol=1e-7)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_render_accepts_the_filter_and_keeps_gradients():
    from metalsplat import render

    torch.manual_seed(0)
    n = 30
    means = torch.rand(n, 3, device="mps") * 2 - 1
    means[:, 2] = means[:, 2].abs() + 2.0
    model = GaussianModel(means, colors=torch.rand(n, 3, device="mps")).to("mps")
    camera = Camera.identity(fx=FX, fy=FY, cx=W / 2, cy=H / 2, img_width=W, img_height=H).to("mps")

    filter_3d = compute_3d_filter(model.means.detach(), [camera])
    image = render(model, camera, filter_3d=filter_3d)
    image.pow(2).mean().backward()
    torch.mps.synchronize()

    assert torch.isfinite(image).all()
    assert model.raw_scales.grad is not None and torch.isfinite(model.raw_scales.grad).all()
    assert model.means.grad is not None and torch.isfinite(model.means.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_filter_suppresses_subpixel_detail_and_leaves_resolved_detail_alone():
    # antialias=False on purpose: the *2D* compensation already suppresses a
    # sub-pixel gaussian, so with it on, both renders come out empty and the
    # comparison says nothing about the 3D filter. This isolates the 3D half.
    from metalsplat import render

    camera = Camera.identity(fx=FX, fy=FY, cx=W / 2, cy=H / 2, img_width=W, img_height=H).to("mps")
    means = torch.tensor([[0.0, 0.0, 3.0]], device="mps")

    def energy(scale, use_filter):
        model = GaussianModel(
            means, scales=torch.full((1, 3), scale, device="mps"),
            colors=torch.ones(1, 3, device="mps"),
        ).to("mps")
        f = compute_3d_filter(model.means.detach(), [camera]) if use_filter else None
        with torch.no_grad():
            img = render(model, camera, antialias=False, filter_3d=f)
            torch.mps.synchronize()
        return img.sum().item()

    # Far below the sampling rate (z/f = 0.06 world units per pixel here):
    # never observed at this detail, so the filter should remove it.
    tiny_plain, tiny_filtered = energy(1e-3, False), energy(1e-3, True)
    assert tiny_plain > 1e-3, "test scene renders nothing; nothing is being compared"
    assert tiny_filtered < 0.01 * tiny_plain

    # Comfortably resolved: the filter must not touch it.
    big_plain, big_filtered = energy(0.5, False), energy(0.5, True)
    assert big_plain > 1e-3
    assert abs(big_filtered - big_plain) < 0.02 * big_plain


def test_math_matches_the_covariance_identity():
    # Sigma + r^2 I must equal R diag(s^2 + r^2) R^T -- the whole reason
    # this needs no kernel change. Checked against the explicit matrices.
    from metalsplat.utils.quaternion import quat_to_rotmat

    torch.manual_seed(3)
    scales = torch.rand(4, 3) + 0.2
    raw_q = torch.randn(4, 4)
    quats = raw_q / raw_q.norm(dim=-1, keepdim=True)
    r = torch.rand(4) * 0.3

    rot = quat_to_rotmat(quats)
    sigma = rot @ torch.diag_embed(scales**2) @ rot.transpose(-1, -2)
    expected = sigma + (r**2)[:, None, None] * torch.eye(3)

    dilated, _ = apply_3d_filter(scales, torch.ones(4), r)
    got = rot @ torch.diag_embed(dilated**2) @ rot.transpose(-1, -2)

    assert torch.allclose(got, expected, atol=1e-5)
    # And the compensation is the determinant ratio it claims to be.
    _, comp = apply_3d_filter(scales, torch.ones(4), r)
    ratio = (torch.linalg.det(sigma) / torch.linalg.det(expected)).sqrt()
    assert torch.allclose(comp, ratio, atol=1e-5)


def test_filter_is_computed_over_many_cameras_efficiently():
    means = torch.randn(500, 3)
    means[:, 2] = means[:, 2].abs() + 3.0
    cams = [_camera_at(float(d)) for d in range(1, 20)]

    r = compute_3d_filter(means, cams)
    assert r.shape == (500,)
    assert (r >= 0).all()
    assert math.isfinite(float(r.max()))


def test_carry_inherits_from_the_parent_and_zeroes_orphans():
    from metalsplat.filter3d import carry_filter_3d

    filter_3d = torch.tensor([0.1, 0.2, 0.3])
    # survivors 0 and 2, a child of 2, and a parentless seeded gaussian
    parent_index = torch.tensor([0, 2, 2, -1])

    out = carry_filter_3d(filter_3d, parent_index)
    assert torch.allclose(out, torch.tensor([0.1, 0.3, 0.3, 0.0]))


def test_carry_is_a_no_op_when_the_filter_is_disabled():
    from metalsplat.filter3d import carry_filter_3d

    assert carry_filter_3d(None, torch.arange(4)) is None


def test_densify_parent_index_points_children_at_their_parent():
    # source_index marks new gaussians -1 (Adam starts them cold);
    # parent_index instead names the gaussian they came from, which is what
    # position-derived quantities like the filter radius inherit.
    from metalsplat.densify import densify_and_prune
    from metalsplat.optim import NEW_GAUSSIAN

    n = 10
    torch.manual_seed(0)
    model = GaussianModel(
        torch.randn(n, 3), scales=torch.full((n, 3), 0.5),
        opacities=torch.full((n,), 0.5), colors=torch.rand(n, 3),
    )
    with torch.no_grad():
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # split candidate

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    grad_accum[1] = 9.0
    _, stats = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=1.0, grad_percentile=0.8
    )

    assert stats.parent_index.shape == stats.source_index.shape
    assert (stats.parent_index >= 0).all(), "every densified gaussian has a parent"
    assert int((stats.source_index == NEW_GAUSSIAN).sum()) > 0, "test isn't exercising new gaussians"
    # Where source_index names a survivor, the two agree.
    survivors = stats.source_index >= 0
    assert torch.equal(stats.parent_index[survivors], stats.source_index[survivors])


def test_chunking_does_not_change_the_result():
    means = torch.randn(300, 3)
    means[:, 2] = means[:, 2].abs() + 3.0
    cams = [_camera_at(float(d)) for d in range(1, 25)]

    a = compute_3d_filter(means, cams, camera_chunk=1)
    b = compute_3d_filter(means, cams, camera_chunk=64)
    assert torch.allclose(a, b, atol=1e-7)
