import torch

from metalsplat.reference.project_2dgs_ref import project_gaussians_2dgs
from metalsplat.utils.quaternion import quat_to_rotmat

IDENTITY_R = torch.eye(3)
ZERO_T = torch.zeros(3)


def test_centered_disk_projects_to_principal_point():
    means = torch.tensor([[0.0, 0.0, 5.0]])
    scales = torch.tensor([[0.1, 0.1]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )

    assert bool(out.valid[0])
    assert torch.allclose(out.means2d[0], torch.tensor([50.0, 50.0]), atol=1e-4)
    assert torch.allclose(out.depths[0], torch.tensor(5.0))
    assert out.radii[0] > 0
    assert torch.isfinite(out.transform).all()


def test_disk_behind_camera_is_culled():
    means = torch.tensor([[0.0, 0.0, -5.0]])
    scales = torch.tensor([[0.1, 0.1]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )

    assert not bool(out.valid[0])
    assert out.radii[0] == 0


def test_normal_flipped_to_face_camera():
    # Identity quat -> raw normal (rotmat column 2) is world +z, but the
    # gaussian sits in front of the camera along +z, so the camera-facing
    # normal must point back at the camera, i.e. -z.
    means = torch.tensor([[0.0, 0.0, 5.0]])
    scales = torch.tensor([[0.2, 0.2]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )
    assert torch.allclose(out.normal[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    n = 5
    means = torch.randn(n, 3)
    means[:, 2] = means[:, 2].abs() + 2.0  # keep in front of camera
    means.requires_grad_()
    raw_scales = torch.randn(n, 2, requires_grad=True)
    raw_quats = torch.randn(n, 4, requires_grad=True)

    scales = raw_scales.exp()
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )

    loss = out.means2d.sum() + out.transform.sum() + out.normal.sum()
    loss.backward()

    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert raw_scales.grad is not None and torch.isfinite(raw_scales.grad).all()
    assert raw_quats.grad is not None and torch.isfinite(raw_quats.grad).all()


def test_ray_splat_intersection_matches_independent_ray_plane_calculation():
    """Cross-checks the `transform` (M) formula's cross-product ray-splat
    solve against a completely independent method: an explicit 3D
    ray/plane intersection (unproject the pixel, intersect with the
    splat's plane, project the hit point onto the tangent basis) computed
    without going through `transform`/`H`/`W` at all -- see
    project_2dgs_ref's module docstring for what's being checked.
    """
    mean = torch.tensor([0.3, -0.2, 4.0])
    s_u, s_v = 0.4, 0.6
    quat = torch.tensor([0.9, 0.2, -0.3, 0.1])
    quat = quat / quat.norm()
    fx, fy, cx, cy = 150.0, 150.0, 50.0, 50.0

    out = project_gaussians_2dgs(
        mean.unsqueeze(0),
        torch.tensor([[s_u, s_v]]),
        quat.unsqueeze(0),
        IDENTITY_R,
        ZERO_T,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=100,
        img_height=100,
    )
    row0, row1, row2 = out.transform[0, 0], out.transform[0, 1], out.transform[0, 2]

    rotmat = quat_to_rotmat(quat)
    t_u_unit, t_v_unit, normal = rotmat[:, 0], rotmat[:, 1], rotmat[:, 2]

    for px, py in [(50.0, 50.0), (30.0, 70.0), (65.0, 40.0), (20.0, 20.0)]:
        # Independent calculation: camera at the origin (identity
        # extrinsics), ray direction from the pinhole model, intersect
        # with the plane (mean, normal), then project onto the tangent
        # basis directly (dot products), with no use of `transform`.
        d = torch.tensor([(px - cx) / fx, (py - cy) / fy, 1.0])
        t = (mean @ normal) / (d @ normal)
        hit = t * d
        delta = hit - mean
        u_expected = (delta @ t_u_unit) / s_u
        v_expected = (delta @ t_v_unit) / s_v
        z_expected = hit[2]

        h_u = row0 - px * row2
        h_v = row1 - py * row2
        c = torch.linalg.cross(h_u, h_v, dim=-1)
        u = c[0] / c[2]
        v = c[1] / c[2]
        z = row2[0] * u + row2[1] * v + row2[2]

        assert torch.allclose(u, u_expected, atol=1e-4)
        assert torch.allclose(v, v_expected, atol=1e-4)
        assert torch.allclose(z, z_expected, atol=1e-4)
