import pytest
import torch

from metalsplat.ops.project import project_gaussians
from metalsplat.reference.project_ref import project_gaussians as project_gaussians_ref

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

FX = FY = 50.0
CX = CY = 32.0
W = H = 64
NEAR = 0.1
EPS2D = 0.3


def _random_scene(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g)
    means[:, 2] = means[:, 2].abs() + 2.0  # keep well in front of the camera
    scales = torch.rand(n, 3, generator=g) * 0.3 + 0.05
    raw_quats = torch.randn(n, 4, generator=g)
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)
    return means, scales, quats


def _random_camera(seed=1):
    g = torch.Generator().manual_seed(seed)
    # small random rotation via a normalized quaternion -> rotmat, so R_wc is
    # a genuine orthonormal rotation rather than an arbitrary matrix.
    from metalsplat.utils.quaternion import quat_to_rotmat

    raw = torch.randn(4, generator=g)
    quat = raw / raw.norm()
    R_wc = quat_to_rotmat(quat)
    t_wc = torch.randn(3, generator=g) * 0.2
    return R_wc, t_wc


def _run_both(n, seed=0):
    means, scales, quats = _random_scene(n, seed=seed)
    R_wc, t_wc = _random_camera(seed=seed + 100)

    ref = project_gaussians_ref(
        means, scales, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D
    )

    means_mps = means.to("mps")
    scales_mps = scales.to("mps")
    quats_mps = quats.to("mps")
    R_wc_mps = R_wc.to("mps")
    t_wc_mps = t_wc.to("mps")

    means2d, depths, conics, radii, valid = project_gaussians(
        means_mps, scales_mps, quats_mps, R_wc_mps, t_wc_mps,
        FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D,
    )
    torch.mps.synchronize()

    return ref, (means2d.cpu(), depths.cpu(), conics.cpu(), radii.cpu(), valid.cpu())


@pytest.mark.parametrize("n", [1, 8, 37])
def test_forward_matches_reference(n):
    ref, (means2d, depths, conics, radii, valid) = _run_both(n)

    assert torch.equal(valid, ref.valid)
    mask = ref.valid

    assert torch.allclose(means2d[mask], ref.means2d[mask], atol=1e-3, rtol=1e-3)
    assert torch.allclose(depths, ref.depths, atol=1e-4, rtol=1e-4)
    assert torch.allclose(conics[mask], ref.conics[mask], atol=1e-3, rtol=1e-3)
    assert torch.allclose(radii, ref.radii, atol=1e-3)


@pytest.mark.parametrize("n", [1, 8, 37])
def test_backward_matches_reference(n):
    means, scales, quats = _random_scene(n, seed=0)
    R_wc, t_wc = _random_camera(seed=100)

    means_ref = means.clone().requires_grad_()
    scales_ref = scales.clone().requires_grad_()
    quats_ref = quats.clone().requires_grad_()

    ref = project_gaussians_ref(
        means_ref, scales_ref, quats_ref, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D
    )

    g = torch.Generator().manual_seed(42)
    upstream_means2d = torch.randn(n, 2, generator=g)
    upstream_conics = torch.randn(n, 3, generator=g)
    # zero-out gradient contribution from culled gaussians, since the
    # rasterizer would never route gradient to them either.
    upstream_means2d = upstream_means2d * ref.valid[:, None]
    upstream_conics = upstream_conics * ref.valid[:, None]

    loss_ref = (ref.means2d * upstream_means2d).sum() + (ref.conics * upstream_conics).sum()
    loss_ref.backward()

    means_mps = means.to("mps").requires_grad_()
    scales_mps = scales.to("mps").requires_grad_()
    quats_mps = quats.to("mps").requires_grad_()
    R_wc_mps = R_wc.to("mps")
    t_wc_mps = t_wc.to("mps")

    means2d, depths, conics, radii, valid = project_gaussians(
        means_mps, scales_mps, quats_mps, R_wc_mps, t_wc_mps,
        FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D,
    )
    loss = (means2d * upstream_means2d.to("mps")).sum() + (conics * upstream_conics.to("mps")).sum()
    loss.backward()
    torch.mps.synchronize()

    assert torch.allclose(means_mps.grad.cpu(), means_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(scales_mps.grad.cpu(), scales_ref.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(quats_mps.grad.cpu(), quats_ref.grad, atol=2e-2, rtol=2e-2)


def _off_axis_scene():
    """Gaussians off to the side of the camera, close in and fairly large.

    This is the regime the EWA affine approximation breaks down in: J's third
    column is -f*x/z^2, so without the FOV clamp these produce enormous 2D
    covariances. Half-FOV here is atan(32/50) = 33 degrees, so x/z beyond
    ~0.64 is off screen.

    Small z and a large scale are what make this dangerous rather than
    merely wrong: the projected radius grows as x/z^2 while the projected
    *centre* only grows as x/z, so past a point the radius outruns how far
    off-screen the centre is. The "does this bbox touch the image" cull then
    passes, and a gaussian nowhere near the frustum gets binned into every
    tile. That is exactly the configuration seen in the garden scene.
    """
    z = torch.full((7,), 0.3)
    x = torch.tensor([0.0, 1.0, 3.0, 5.0, 10.0, 15.0, -15.0]) * z
    means = torch.stack([x, torch.zeros_like(x), z], dim=-1)
    scales = torch.full((7, 3), 0.5)
    quats = torch.zeros(7, 4)
    quats[:, 0] = 1.0
    return means, scales, quats


def test_off_axis_gaussians_get_bounded_radii():
    # Regression: gaussians far off-axis used to project to radii in the
    # hundreds of thousands of pixels (measured peak 171,000px on a 972x630
    # render), pass the "does the bbox touch the image" test trivially, and
    # paint a huge blob over every frame they appeared in.
    means, scales, quats = _off_axis_scene()
    R_wc = torch.eye(3)
    t_wc = torch.zeros(3)

    _, _, _, radii, _ = project_gaussians(
        means.to("mps"), scales.to("mps"), quats.to("mps"), R_wc.to("mps"), t_wc.to("mps"),
        FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D,
    )
    torch.mps.synchronize()

    # The on-axis gaussian sets the scale for what a sane radius looks like.
    # These all have identical (isotropic) scales at identical depth, so with
    # the Jacobian bounded none may exceed it -- the clamp caps the off-axis
    # ones slightly *below* on-axis. Unclamped, these reach ~10x on-axis.
    on_axis = radii[0].item()
    assert on_axis > 0
    assert radii.max().item() <= 2.5 * on_axis, f"radii {radii.tolist()} vs on-axis {on_axis}"


def test_off_axis_forward_matches_reference():
    means, scales, quats = _off_axis_scene()
    R_wc, t_wc = torch.eye(3), torch.zeros(3)

    ref = project_gaussians_ref(
        means, scales, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D
    )
    means2d, _, conics, radii, valid = project_gaussians(
        means.to("mps"), scales.to("mps"), quats.to("mps"), R_wc.to("mps"), t_wc.to("mps"),
        FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D,
    )
    torch.mps.synchronize()

    assert torch.equal(valid.cpu(), ref.valid)
    assert torch.allclose(conics.cpu(), ref.conics, atol=1e-4, rtol=1e-3)
    assert torch.allclose(radii.cpu(), ref.radii, atol=1e-3)


def test_off_axis_backward_matches_reference():
    # The clamp is a piecewise function: once x/z saturates, the Jacobian
    # stops depending on x at all. If the kernel kept the unclamped
    # derivative here it would disagree with the reference's autograd.
    means, scales, quats = _off_axis_scene()
    R_wc, t_wc = torch.eye(3), torch.zeros(3)
    n = means.shape[0]

    means_ref = means.clone().requires_grad_()
    scales_ref = scales.clone().requires_grad_()
    ref = project_gaussians_ref(
        means_ref, scales_ref, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D
    )

    g = torch.Generator().manual_seed(7)
    up_means2d = torch.randn(n, 2, generator=g) * ref.valid[:, None]
    up_conics = torch.randn(n, 3, generator=g) * ref.valid[:, None]
    ((ref.means2d * up_means2d).sum() + (ref.conics * up_conics).sum()).backward()

    means_mps = means.to("mps").requires_grad_()
    scales_mps = scales.to("mps").requires_grad_()
    means2d, _, conics, _, _ = project_gaussians(
        means_mps, scales_mps, quats.to("mps"), R_wc.to("mps"), t_wc.to("mps"),
        FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D,
    )
    ((means2d * up_means2d.to("mps")).sum() + (conics * up_conics.to("mps")).sum()).backward()
    torch.mps.synchronize()

    assert torch.allclose(means_mps.grad.cpu(), means_ref.grad, atol=2e-3, rtol=2e-2)
    assert torch.allclose(scales_mps.grad.cpu(), scales_ref.grad, atol=2e-3, rtol=2e-2)
