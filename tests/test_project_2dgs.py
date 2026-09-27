import pytest
import torch

from metalsplat.ops.project_2dgs import project_gaussians_2dgs
from metalsplat.reference.project_2dgs_ref import (
    project_gaussians_2dgs as project_gaussians_2dgs_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

FX = FY = 50.0
CX = CY = 32.0
W = H = 64
NEAR = 0.1


def _random_scene(n, seed=0, R_wc=None, t_wc=None):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g) * 2.0 + 2.0
    xy_limit = 0.5 * min(CX / FX, CY / FY)
    x = (torch.rand(n, generator=g) * 2 - 1) * xy_limit * z
    y = (torch.rand(n, generator=g) * 2 - 1) * xy_limit * z
    means_cam = torch.stack([x, y, z], dim=-1)

    if R_wc is None:
        means = means_cam
    else:
        means = (means_cam - t_wc) @ R_wc

    scales = torch.rand(n, 2, generator=g) * 0.3 + 0.05
    raw_quats = torch.randn(n, 4, generator=g)
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)
    return means, scales, quats


def _random_camera(seed=1):
    g = torch.Generator().manual_seed(seed)
    from metalsplat.utils.quaternion import quat_to_rotmat

    raw = torch.randn(4, generator=g)
    quat = raw / raw.norm()
    R_wc = quat_to_rotmat(quat)
    t_wc = torch.randn(3, generator=g) * 0.2
    return R_wc, t_wc


def _run_both(n, seed=0, opacities=None):
    R_wc, t_wc = _random_camera(seed=seed + 100)
    means, scales, quats = _random_scene(n, seed=seed, R_wc=R_wc, t_wc=t_wc)

    ref = project_gaussians_2dgs_ref(
        means, scales, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR,
        opacities=opacities,
    )  # fmt: skip

    out = project_gaussians_2dgs(
        means.to("mps"),
        scales.to("mps"),
        quats.to("mps"),
        R_wc.to("mps"),
        t_wc.to("mps"),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
        opacities=None if opacities is None else opacities.to("mps"),
    )
    torch.mps.synchronize()
    return ref, tuple(t.cpu() for t in out)


@pytest.mark.parametrize("n", [1, 8, 37])
@pytest.mark.parametrize("with_opacities", [False, True])
def test_forward_matches_reference(n, with_opacities):
    g = torch.Generator().manual_seed(5)
    # Spans faint-enough-to-cull (< 1/255) through opaque.
    opacities = torch.rand(n, generator=g) if with_opacities else None
    ref, (means2d, depths, rects, valid, transform, normal) = _run_both(
        n, opacities=opacities
    )

    assert torch.equal(valid, ref.valid)
    mask = ref.valid

    assert torch.allclose(means2d[mask], ref.means2d[mask], atol=1e-3, rtol=1e-3)
    assert torch.allclose(depths, ref.depths, atol=1e-4, rtol=1e-4)
    assert torch.allclose(rects, ref.rects, atol=1e-2, rtol=1e-3)
    assert torch.allclose(
        transform.reshape(n, 3, 3)[mask], ref.transform[mask], atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(normal[mask], ref.normal[mask], atol=1e-3, rtol=1e-3)


def test_border_culling_matches_reference():
    """Splats straddling the frame edge, which `_random_scene` never produces.

    Its centers are clamped to the middle half of the frame, so every
    gaussian there is comfortably in bounds and the culling test is never
    exercised near a border -- exactly where the footprint bound decides
    whether a splat lives.
    """
    R_wc, t_wc = torch.eye(3), torch.zeros(3)
    z = 3.0
    # Pixel x of each center, from well off the left edge to well off the
    # right. A face-on 0.1 disk at z=3 reaches 3.33 * 50 * 0.1 / 3 ~= 5.55px,
    # so -6 / 70 just miss the 64px frame and -5 / 69 just reach it.
    offsets = torch.tensor([-200.0, -46.0, -6.0, -5.0, 10.0, 40.0, 69.0, 70.0, 300.0])
    n = offsets.numel()
    means = torch.stack(
        [(offsets - CX) / FX * z, torch.zeros(n), torch.full((n,), z)], dim=-1
    )
    scales = torch.full((n, 2), 0.1)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0

    ref = project_gaussians_2dgs_ref(
        means, scales, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR
    )
    out = project_gaussians_2dgs(
        means.to("mps"),
        scales.to("mps"),
        quats.to("mps"),
        R_wc.to("mps"),
        t_wc.to("mps"),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
    )
    torch.mps.synchronize()
    _, _, rects, valid, _, _ = (t.cpu() for t in out)

    assert torch.equal(valid, ref.valid)
    assert torch.allclose(rects, ref.rects, atol=1e-2, rtol=1e-3)
    expected = torch.tensor([False, False, False, True, True, True, True, False, False])
    assert torch.equal(ref.valid, expected)


@pytest.mark.parametrize("n", [1, 8, 37])
def test_backward_matches_reference(n):
    R_wc, t_wc = _random_camera(seed=100)
    means, scales, quats = _random_scene(n, seed=0, R_wc=R_wc, t_wc=t_wc)

    means_ref = means.clone().requires_grad_()
    scales_ref = scales.clone().requires_grad_()
    quats_ref = quats.clone().requires_grad_()

    ref = project_gaussians_2dgs_ref(
        means_ref,
        scales_ref,
        quats_ref,
        R_wc,
        t_wc,
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
    )

    g = torch.Generator().manual_seed(42)
    up_means2d = torch.randn(n, 2, generator=g) * ref.valid[:, None]
    up_transform = torch.randn(n, 3, 3, generator=g) * ref.valid[:, None, None]
    up_normal = torch.randn(n, 3, generator=g) * ref.valid[:, None]
    # `depths` is the camera-space z of the mean -- a differentiable
    # output, not just a sort key, since rasterize_2dgs falls back to it
    # for z_hit. Masked by `valid` like the rest: the kernel returns a
    # zero gradient for a culled gaussian, where the reference would still
    # differentiate z through to `means`.
    up_depths = torch.randn(n, generator=g) * ref.valid

    loss_ref = (
        (ref.means2d * up_means2d).sum()
        + (ref.depths * up_depths).sum()
        + (ref.transform * up_transform).sum()
        + (ref.normal * up_normal).sum()
    )
    loss_ref.backward()

    means_mps = means.to("mps").requires_grad_()
    scales_mps = scales.to("mps").requires_grad_()
    quats_mps = quats.to("mps").requires_grad_()

    out = project_gaussians_2dgs(
        means_mps,
        scales_mps,
        quats_mps,
        R_wc.to("mps"),
        t_wc.to("mps"),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
    )
    means2d, depths, _rects, _valid, transform, normal = out
    loss = (
        (means2d * up_means2d.to("mps")).sum()
        + (depths * up_depths.to("mps")).sum()
        + (transform.reshape(n, 3, 3) * up_transform.to("mps")).sum()
        + (normal * up_normal.to("mps")).sum()
    )
    loss.backward()
    torch.mps.synchronize()

    assert torch.allclose(means_mps.grad.cpu(), means_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(scales_mps.grad.cpu(), scales_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(quats_mps.grad.cpu(), quats_ref.grad, atol=2e-2, rtol=2e-2)
