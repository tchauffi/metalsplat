import torch

from metalsplat.reference.project_ref import project_gaussians

IDENTITY_R = torch.eye(3)
ZERO_T = torch.zeros(3)


def test_centered_isotropic_gaussian_projects_to_principal_point():
    means = torch.tensor([[0.0, 0.0, 5.0]])
    scales = torch.tensor([[0.1, 0.1, 0.1]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians(
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
    # conic should represent a symmetric positive-definite 2x2 inverse
    a, b, c = out.conics[0]
    assert a > 0 and c > 0
    assert (a * c - b * b) > 0


def test_gaussian_behind_camera_is_culled():
    means = torch.tensor([[0.0, 0.0, -5.0]])
    scales = torch.tensor([[0.1, 0.1, 0.1]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians(
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


def test_gaussian_far_outside_frustum_is_culled():
    means = torch.tensor([[1000.0, 0.0, 5.0]])
    scales = torch.tensor([[0.1, 0.1, 0.1]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    out = project_gaussians(
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


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    n = 5
    means = torch.randn(n, 3)
    means[:, 2] = means[:, 2].abs() + 2.0  # keep in front of camera
    means.requires_grad_()
    raw_scales = torch.randn(n, 3, requires_grad=True)
    raw_quats = torch.randn(n, 4, requires_grad=True)

    scales = raw_scales.exp()
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)

    out = project_gaussians(
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

    loss = out.means2d.sum() + out.conics.sum()
    loss.backward()

    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert raw_scales.grad is not None and torch.isfinite(raw_scales.grad).all()
    assert raw_quats.grad is not None and torch.isfinite(raw_quats.grad).all()
