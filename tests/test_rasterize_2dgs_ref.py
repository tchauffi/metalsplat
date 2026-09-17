import pytest
import torch

from metalsplat.reference.project_2dgs_ref import project_gaussians_2dgs
from metalsplat.reference.project_ref import project_gaussians
from metalsplat.reference.rasterize_2dgs_ref import (
    DEFAULT_FILTER_SIZE,
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
    distortion is exactly `w1*w2*(m2 - m1)^2` -- with `m` the *normalized*
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
    expected = w1 * w2 * (m2 - m1) ** 2

    # rel=1e-4, not tighter: (m2 - m1) is a cancelling difference of two
    # O(1) float32 numbers and squaring doubles its relative error.
    assert out["distortion"][1, 1].item() == pytest.approx(expected, rel=1e-4)

    # And the metric-depth version it must NOT be -- four orders of
    # magnitude apart, so this can never pass by coincidence.
    metric = w1 * w2 * (z2 - z1) ** 2
    assert abs(metric / expected) > 100.0


def _per_gaussian_weights_and_depths(
    proj, opacities, near=0.2, filter_size=DEFAULT_FILTER_SIZE
):
    """Replays the reference rasterizer's compositing loop, but keeps each
    contributing gaussian's per-pixel `(weight, normalized depth)` instead
    of accumulating them -- so a test can brute-force the distortion's
    pairwise definition rather than re-deriving the same running sums the
    implementation uses (which would just test the code against itself).

    Returns two `(N, H, W)` tensors, in compositing order.
    """
    order = torch.argsort(
        torch.where(proj.valid, proj.depths, torch.full_like(proj.depths, float("inf")))
    )
    h, w = 40, 40  # matches _tilted_overlapping_scene's camera
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32) + 0.5,
        torch.arange(w, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    trans = torch.ones(h, w)
    weights, depths_m = [], []
    scale = DISTORTION_FAR / (DISTORTION_FAR - near)
    for i in order:
        if not bool(proj.valid[i]):
            continue
        row0, row1, row2 = proj.transform[i]
        h_u = row0 - xs.unsqueeze(-1) * row2
        h_v = row1 - ys.unsqueeze(-1) * row2
        cross = torch.linalg.cross(h_u, h_v, dim=-1)
        w_local = cross[..., 2]
        degenerate = w_local.abs() < 1e-9
        safe = torch.where(degenerate, torch.ones_like(w_local), w_local)
        u, v = cross[..., 0] / safe, cross[..., 1] / safe
        rho_uv = torch.where(
            degenerate, torch.full_like(w_local, float("inf")), u * u + v * v
        )
        d2d = torch.stack([xs, ys], dim=-1) - proj.means2d[i]
        rho_screen = (d2d[..., 0] ** 2 + d2d[..., 1] ** 2) / (filter_size**2)
        uv_active = rho_uv <= rho_screen
        rho = torch.where(uv_active, rho_uv, rho_screen)
        z_hit = torch.where(
            degenerate | ~uv_active,
            proj.depths[i].expand_as(w_local),
            row2[0] * u + row2[1] * v + row2[2],
        )
        alpha = (opacities[i] * torch.exp(-0.5 * rho)).clamp(max=0.99)
        alpha = torch.where(z_hit <= near, torch.zeros_like(alpha), alpha)
        alpha_eff = torch.where(
            (trans >= 1e-4) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha)
        )
        weights.append(trans * alpha_eff)
        depths_m.append(scale * (1.0 - near / z_hit.clamp_min(near)))
        trans = trans * (1 - alpha_eff)
    return torch.stack(weights), torch.stack(depths_m)


def _tilted_overlapping_scene(seed=0, n=12):
    """Random *steeply tilted* overlapping disks in front of a 40x40 camera.

    Tilt is the point: a disk whose plane is oblique to the view direction
    has a per-pixel ray-splat intersection depth that varies across its own
    footprint, so the compositing sequence (sorted by each gaussian's
    *mean* depth) stops being monotonic in `z_hit` at a given pixel. That
    is the configuration the distortion regularizer has to stay
    well-behaved on.
    """
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * 0.6
    means[:, 2] = means[:, 2].abs() * 0.5 + 4.0
    scales = torch.rand(n, 2, generator=g) * 0.5 + 0.4
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.rand(n, generator=g) * 0.5 + 0.3
    colors = torch.rand(n, 3, generator=g)
    proj = project_gaussians_2dgs(
        means, scales, quats, IDENTITY_R, ZERO_T, 120.0, 120.0, 20.0, 20.0, 40, 40
    )
    return proj, opacities, colors


def test_distortion_is_the_squared_pairwise_second_moment():
    """Pins the distortion map to `sum_{i<j} w_i*w_j*(m_i - m_j)^2`.

    Checked against a brute-force double loop over contributing pairs, on a
    tilted scene (see `_tilted_overlapping_scene`), rather than against the
    prefix-moment recursion the implementation itself runs.
    """
    proj, opacities, colors = _tilted_overlapping_scene()
    out = rasterize_gaussians_2dgs(
        proj.means2d,
        proj.depths,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        40,
        40,
    )
    weights, m = _per_gaussian_weights_and_depths(proj, opacities)

    k = weights.shape[0]
    brute = torch.zeros(40, 40)
    for i in range(k):
        for j in range(i + 1, k):
            brute += weights[i] * weights[j] * (m[i] - m[j]) ** 2

    assert torch.allclose(out["distortion"], brute, atol=1e-7)


def test_distortion_stays_non_negative_when_z_hit_contradicts_mean_depth_order():
    """Regression guard for the signed first-power variant this used to be.

    That variant accumulated `2*sum_{i<j} w_i*w_j*(m_j - m_i)`, which is
    signed, so it went *negative* on exactly the configuration tilted
    splats produce: a pixel whose contributing gaussians are out of `z_hit`
    order relative to the mean-depth compositing sequence. Minimizing it
    therefore rewarded tilting splats until their intersection depths
    contradicted their mean depths -- edge-on, depth-scrambling surfels --
    instead of concentrating weight along the ray.

    The scene below is built so that pathology is actually exercised: the
    signed variant is negative on a substantial fraction of its pixels
    (asserted, so this can't quietly become a test of nothing), while the
    squared second moment this now computes stays non-negative.
    """
    proj, opacities, colors = _tilted_overlapping_scene()
    out = rasterize_gaussians_2dgs(
        proj.means2d,
        proj.depths,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        40,
        40,
    )
    weights, m = _per_gaussian_weights_and_depths(proj, opacities)

    k = weights.shape[0]
    signed = torch.zeros(40, 40)
    for i in range(k):
        for j in range(i + 1, k):
            signed += 2.0 * weights[i] * weights[j] * (m[j] - m[i])

    assert (signed < -1e-7).float().mean() > 0.05, (
        "scene doesn't exercise the pathology"
    )
    # -1e-9, not 0: the prefix-moment expansion m^2*A - 2*m*M1 + M2 is a
    # cancelling form in float32, so a pixel whose weight is already
    # concentrated can land a few ulps below zero. That is ~1e-5 of this
    # scene's peak, against the signed variant's own -1e-3 minimum above.
    assert out["distortion"].min() >= -1e-9
