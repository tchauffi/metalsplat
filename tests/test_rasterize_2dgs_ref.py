import pytest
import torch

from metalsplat.reference.project_2dgs_ref import project_gaussians_2dgs
from metalsplat.reference.project_ref import project_gaussians
from metalsplat.reference.rasterize_2dgs_ref import (
    DISTORTION_FAR,
    rasterize_gaussians_2dgs,
)
from metalsplat.reference.rasterize_ref import rasterize_gaussians

IDENTITY_R = torch.eye(3)
ZERO_T = torch.zeros(3)


def test_frontoparallel_disk_matches_3dgs_analytic_ellipse():
    """For a disk exactly perpendicular to the optical axis with tangent
    axes aligned to the camera's x/y axes, the ray-splat intersection is
    an *exact* perspective (not just first-order-affine) relation -- the
    plane has no curvature relative to the camera, so 3DGS's local-affine
    EWA approximation and 2DGS's exact ray-splat evaluation should agree
    almost exactly (up to 3DGS's small eps2d covariance regularizer). This
    cross-checks rasterize_2dgs_ref/project_2dgs_ref's whole M/H/W/
    cross-product pipeline against the independently-implemented,
    already-tested 3DGS path.
    """
    means = torch.tensor([[0.0, 0.0, 5.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    opacities = torch.tensor([0.8])
    colors = torch.tensor([[1.0, 0.5, 0.25]])
    fx, fy, cx, cy = 200.0, 200.0, 50.0, 50.0
    w, h = 100, 100

    out2d = project_gaussians_2dgs(
        means,
        torch.tensor([[0.5, 0.5]]),
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=w,
        img_height=h,
    )
    img2d = rasterize_gaussians_2dgs(
        out2d.means2d,
        out2d.depths,
        out2d.transform,
        out2d.normal,
        opacities,
        colors,
        out2d.valid,
        w,
        h,
    )["image"]

    out3d = project_gaussians(
        means,
        torch.tensor([[0.5, 0.5, 1e-6]]),
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=w,
        img_height=h,
    )
    img3d = rasterize_gaussians(
        out3d.means2d,
        out3d.depths,
        out3d.conics,
        opacities,
        colors,
        out3d.valid,
        w,
        h,
    )

    assert torch.allclose(img2d, img3d, atol=5e-3)


def test_depth_and_normal_maps_are_differentiable():
    means = torch.randn(3, 3)
    means[:, 2] = means[:, 2].abs() + 3.0
    means.requires_grad_()
    raw_scales = torch.randn(3, 2, requires_grad=True)
    raw_quats = torch.randn(3, 4, requires_grad=True)
    raw_opacities = torch.randn(3, requires_grad=True)
    colors = torch.rand(3, 3, requires_grad=True)

    scales = raw_scales.exp()
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)
    opacities = torch.sigmoid(raw_opacities)

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=100.0,
        fy=100.0,
        cx=25.0,
        cy=25.0,
        img_width=50,
        img_height=50,
    )
    result = rasterize_gaussians_2dgs(
        out.means2d,
        out.depths,
        out.transform,
        out.normal,
        opacities,
        colors,
        out.valid,
        50,
        50,
    )
    loss = (
        result["image"].sum()
        + result["depth"].sum()
        + result["normal"].sum()
        + result["distortion"].sum()
    )
    loss.backward()

    for t in (means, raw_scales, raw_quats, raw_opacities, colors):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_distortion_zero_for_single_gaussian():
    # A single contributing gaussian has no pair to compare depths
    # against, so the distortion map must be exactly zero everywhere.
    means = torch.tensor([[0.0, 0.0, 5.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = project_gaussians_2dgs(
        means,
        torch.tensor([[0.5, 0.5]]),
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=150.0,
        fy=150.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )
    result = rasterize_gaussians_2dgs(
        out.means2d,
        out.depths,
        out.transform,
        out.normal,
        torch.tensor([0.9]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        out.valid,
        100,
        100,
    )
    assert torch.allclose(result["distortion"], torch.zeros(100, 100))


def test_distortion_positive_for_overlapping_gaussians_at_different_depths():
    means = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    scales = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    opacities = torch.tensor([0.6, 0.6])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    out = project_gaussians_2dgs(
        means,
        scales,
        quats,
        IDENTITY_R,
        ZERO_T,
        fx=150.0,
        fy=150.0,
        cx=50.0,
        cy=50.0,
        img_width=100,
        img_height=100,
    )
    result = rasterize_gaussians_2dgs(
        out.means2d,
        out.depths,
        out.transform,
        out.normal,
        opacities,
        colors,
        out.valid,
        100,
        100,
    )
    # Center pixel is covered by both overlapping, differently-depthed
    # disks, so its distortion should be strictly positive.
    assert result["distortion"][50, 50] > 0


def test_distortion_uses_normalized_depth_not_metric_depth():
    """Pins the distortion regularizer's depth parameterization.

    Two front-facing disks on the optical axis, at known depths, produce a
    two-term compositing sequence at the principal-point pixel whose
    distortion is exactly `2*w1*w2*(m2 - m1)` -- with `m` the *normalized*
    inverse depth `far/(far-near)*(1 - near/z)` the official 2DGS CUDA
    rasterizer uses, not the raw metric `z`. Both alphas are exact there
    (the optical-axis ray hits each disk at u=v=0, so rho=0 and
    alpha=opacity), which makes the expected value fully analytic.

    This matters beyond parameterization taste: accumulating metric `z`
    instead inflates the term by ~4 orders of magnitude at ordinary scene
    depths, which would silently make the paper's own `lambda_dist` values
    swamp the photometric loss. Guards against a regression to metric depth.
    """
    near, z1, z2 = 0.2, 3.0, 7.0
    o1, o2 = 0.5, 0.6
    fx = fy = 100.0
    cx = cy = 1.5  # principal point lands exactly on pixel (1, 1)'s center
    w = h = 3

    means = torch.tensor([[0.0, 0.0, z1], [0.0, 0.0, z2]])
    scales = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacities = torch.tensor([o1, o2])
    colors = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    proj = project_gaussians_2dgs(
        means, scales, quats, IDENTITY_R, ZERO_T, fx, fy, cx, cy, w, h, near=near
    )
    out = rasterize_gaussians_2dgs(
        proj.means2d,
        proj.depths,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        w,
        h,
        near=near,
    )

    far = DISTORTION_FAR
    m1 = far / (far - near) * (1.0 - near / z1)
    m2 = far / (far - near) * (1.0 - near / z2)
    w1 = o1
    w2 = (1.0 - o1) * o2
    expected = 2.0 * w1 * w2 * (m2 - m1)

    assert out["distortion"][1, 1].item() == pytest.approx(expected, rel=1e-5)

    # And the metric-depth version it must NOT be -- three orders of
    # magnitude apart, so this can never pass by coincidence.
    metric = 2.0 * w1 * w2 * (z2 - z1)
    assert abs(metric / expected) > 100.0
