import math

import pytest
import torch

from metalsplat.ops.rasterize_2dgs import rasterize_gaussians_2dgs
from metalsplat.reference.project_2dgs_ref import (
    project_gaussians_2dgs as project_gaussians_2dgs_ref,
)
from metalsplat.reference.rasterize_2dgs_ref import DEFAULT_FILTER_SIZE
from metalsplat.reference.rasterize_2dgs_ref import (
    rasterize_gaussians_2dgs as rasterize_gaussians_2dgs_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

FX = FY = 50.0
CX = CY = 32.0
W = H = 64
NEAR = 0.1
EPS2D = 0.3  # projection: 2D covariance dilation, px^2
FILTER_SIZE = DEFAULT_FILTER_SIZE  # rasterizer: screen-space fallback sigma, px


def _random_scene(n, seed=0):
    """A self-consistent scene via the (already-validated) reference
    projection: rasterize_2dgs's per-pixel math genuinely depends on a
    geometrically consistent `transform`, so it's easier and more
    representative to project a real 3D scene than to fabricate one in
    screen space (unlike 3DGS's rasterize test, which can fabricate
    means2d/conics directly).
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=g) * 2.0 + 2.0
    xy_limit = 0.5 * min(CX / FX, CY / FY)
    x = (torch.rand(n, generator=g) * 2 - 1) * xy_limit * z
    y = (torch.rand(n, generator=g) * 2 - 1) * xy_limit * z
    means = torch.stack([x, y, z], dim=-1)
    scales = torch.rand(n, 2, generator=g) * 0.4 + 0.1
    raw_quats = torch.randn(n, 4, generator=g)
    quats = raw_quats / raw_quats.norm(dim=-1, keepdim=True)
    opacities = torch.rand(n, generator=g) * 0.6 + 0.3
    colors = torch.rand(n, 3, generator=g)

    R_wc, t_wc = torch.eye(3), torch.zeros(3)
    proj = project_gaussians_2dgs_ref(
        means, scales, quats, R_wc, t_wc, FX, FY, CX, CY, W, H, near=NEAR, eps2d=EPS2D
    )
    return proj, opacities, colors


@pytest.mark.parametrize("n", [1, 5, 40])
def test_forward_matches_reference(n):
    proj, opacities, colors = _random_scene(n)

    ref = rasterize_gaussians_2dgs_ref(
        proj.means2d,
        proj.depths,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )

    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(n, 9).to("mps"),
        proj.normal.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    torch.mps.synchronize()
    image, depth, normal, distortion, final_T = (t.cpu() for t in out)

    # Looser than 3DGS's rasterize test (atol=2e-3): unlike an analytic
    # conic, alpha here comes from a ray-splat intersection whose (u, v)
    # can be near-unstable close to the screen-space-fallback boundary
    # (see rasterize_2dgs.metal's uv_active gating), so a handful of pixels
    # can cross the alpha>=1/255 or T<1e-4 cutoff on one platform and not
    # the other -- ordinary GPU-vs-CPU float32 noise, not gradient bias.
    # Empirically bounded well under these across many random scenes/seeds
    # (see RADIUS_SAFETY_MARGIN in project_2dgs_ref.py for the tile-culling
    # margin that keeps this rare).
    assert torch.allclose(image, ref["image"], atol=1e-2, rtol=1e-2)
    assert torch.allclose(depth, ref["depth"], atol=1e-2, rtol=1e-2)
    assert torch.allclose(normal, ref["normal"], atol=1e-2, rtol=1e-2)
    assert torch.allclose(final_T, ref["final_T"], atol=1e-2, rtol=1e-2)
    # Distortion needs its own, much tighter bound: it is a *squared*
    # difference of *normalized* depths (see rasterize_2dgs_ref's module
    # docstring), so its values peak around 1e-4 on these scenes rather
    # than O(1) like the maps above -- the shared 1e-2 tolerance, or even
    # 1e-4, would pass no matter what the kernel computed (a zeroed output
    # included). Measured max absolute deviation here is ~1.3e-7.
    assert torch.allclose(distortion, ref["distortion"], atol=1e-6, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 5, 40])
def test_backward_matches_reference(n):
    proj, opacities, colors = _random_scene(n, seed=1)
    n_transform = proj.transform.reshape(n, 9)

    means2d_ref = proj.means2d.clone().requires_grad_()
    transform_ref = proj.transform.clone().requires_grad_()
    normal_ref = proj.normal.clone().requires_grad_()
    opacities_ref = opacities.clone().requires_grad_()
    colors_ref = colors.clone().requires_grad_()
    depths_ref = proj.depths.clone().requires_grad_()

    ref = rasterize_gaussians_2dgs_ref(
        means2d_ref,
        depths_ref,
        transform_ref,
        normal_ref,
        opacities_ref,
        colors_ref,
        proj.valid,
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )

    g = torch.Generator().manual_seed(11)
    up_image = torch.randn(H, W, 3, generator=g)
    up_depth = torch.randn(H, W, generator=g)
    up_normal = torch.randn(H, W, 3, generator=g)
    up_dist = torch.randn(H, W, generator=g)

    loss_ref = (
        (ref["image"] * up_image).sum()
        + (ref["depth"] * up_depth).sum()
        + (ref["normal"] * up_normal).sum()
        + (ref["distortion"] * up_dist).sum()
    )
    loss_ref.backward()

    means2d_mps = proj.means2d.to("mps").requires_grad_()
    transform_mps = n_transform.to("mps").requires_grad_()
    normal_mps = proj.normal.to("mps").requires_grad_()
    opacities_mps = opacities.to("mps").requires_grad_()
    colors_mps = colors.to("mps").requires_grad_()
    depths_mps = proj.depths.to("mps").requires_grad_()

    out = rasterize_gaussians_2dgs(
        means2d_mps,
        transform_mps,
        normal_mps,
        opacities_mps,
        colors_mps,
        depths_mps,
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    image, depth, normal, distortion, _final_T = out
    loss = (
        (image * up_image.to("mps")).sum()
        + (depth * up_depth.to("mps")).sum()
        + (normal * up_normal.to("mps")).sum()
        + (distortion * up_dist.to("mps")).sum()
    )
    loss.backward()
    torch.mps.synchronize()

    # Looser than 3DGS's project/rasterize backward tests (atol~2e-2): the
    # distortion loss's gradient has 1/(1-alpha)-style terms (see
    # rasterize_2dgs.metal's backward derivation note) that amplify the
    # same rare threshold-boundary noise discussed above, especially for
    # scenes with many overlapping gaussians. Empirically bounded to ~0.06
    # across 180 random scenes/seeds (n up to 40) with the current
    # RADIUS_SAFETY_MARGIN.
    assert torch.allclose(
        means2d_mps.grad.cpu(), means2d_ref.grad, atol=1e-1, rtol=5e-2
    )
    assert torch.allclose(
        transform_mps.grad.cpu(), transform_ref.grad.reshape(n, 9), atol=1e-1, rtol=5e-2
    )
    assert torch.allclose(normal_mps.grad.cpu(), normal_ref.grad, atol=1e-1, rtol=5e-2)
    assert torch.allclose(
        opacities_mps.grad.cpu(), opacities_ref.grad, atol=1e-1, rtol=5e-2
    )
    assert torch.allclose(colors_mps.grad.cpu(), colors_ref.grad, atol=1e-1, rtol=5e-2)
    # `depths` is differentiable too, via the z_hit fallback the per-pixel
    # math takes wherever the ray-splat intersection is unusable.
    assert torch.allclose(depths_mps.grad.cpu(), depths_ref.grad, atol=1e-1, rtol=5e-2)


def test_abs_grad_accum_mutates_in_place_and_nonzero():
    # Unlike 3DGS (where means2d feeds the alpha computation directly),
    # most well-resolved pixels here take the exact ray-splat branch, not
    # the screen-space fallback that a literal d_means2d would capture --
    # so this only checks basic wiring (buffer size, atomic writes land)
    # rather than replicating 3DGS's sign-cancellation test, which relies
    # on means2d being the primary differentiable path.
    n = 20
    proj, opacities, colors = _random_scene(n, seed=2)

    abs_accum = torch.zeros(n, device="mps")
    accum_before = abs_accum

    means2d_mps = proj.means2d.to("mps").requires_grad_()
    transform_mps = proj.transform.reshape(n, 9).to("mps").requires_grad_()
    out = rasterize_gaussians_2dgs(
        means2d_mps,
        transform_mps,
        proj.normal.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
        abs_grad_accum=abs_accum,
    )
    out[0].sum().backward()
    torch.mps.synchronize()

    assert abs_accum is accum_before
    assert (abs_accum >= 0).all()
    assert (abs_accum > 0).any()


@pytest.mark.parametrize("n", [5, 40])
def test_distortion_backward_matches_reference_in_isolation(n):
    """Distortion-only upstream gradient.

    `test_backward_matches_reference` sums image/depth/normal/distortion
    gradients into one loss, and since distortion runs on normalized depth
    its contribution there is ~1e-4 of the others -- well inside that
    test's tolerance, so it would not notice if the distortion backward
    (its closed-form alpha chain, and the dm/dz factor that maps its depth
    gradient back to metric units) were wrong. This isolates it.
    """
    proj, opacities, colors = _random_scene(n, seed=1)
    n_transform = proj.transform.reshape(n, 9)

    means2d_ref = proj.means2d.clone().requires_grad_()
    transform_ref = proj.transform.clone().requires_grad_()
    opacities_ref = opacities.clone().requires_grad_()

    ref = rasterize_gaussians_2dgs_ref(
        means2d_ref,
        proj.depths,
        transform_ref,
        proj.normal,
        opacities_ref,
        colors,
        proj.valid,
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    g = torch.Generator().manual_seed(11)
    up_dist = torch.randn(H, W, generator=g)
    (ref["distortion"] * up_dist).sum().backward()

    means2d_mps = proj.means2d.to("mps").requires_grad_()
    transform_mps = n_transform.to("mps").requires_grad_()
    opacities_mps = opacities.to("mps").requires_grad_()

    out = rasterize_gaussians_2dgs(
        means2d_mps,
        transform_mps,
        proj.normal.to("mps"),
        opacities_mps,
        colors.to("mps"),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    (out[3] * up_dist.to("mps")).sum().backward()
    torch.mps.synchronize()

    # The distortion term is a *squared* normalized-depth difference, so
    # both the map and its gradients are small in absolute terms (measured
    # |grad|max ~1e-3 for n=5, ~7e-3 for n=40). atol is set ~2 orders below
    # that rather than at an "ordinary" 1e-3, which here would pass even on
    # an entirely zeroed backward.
    assert opacities_ref.grad.abs().max() > 1e-5, "scene exercises no distortion"
    assert torch.allclose(
        transform_mps.grad.cpu(), transform_ref.grad.reshape(n, 9), atol=1e-5, rtol=1e-2
    )
    assert torch.allclose(
        opacities_mps.grad.cpu(), opacities_ref.grad, atol=1e-5, rtol=1e-2
    )
    assert torch.allclose(
        means2d_mps.grad.cpu(), means2d_ref.grad, atol=1e-5, rtol=1e-2
    )


def _reference_pixel_counts(proj, opacities):
    """Brute-force per-gaussian covered-pixel tally from the reference
    rasterizer's own compositing rules (alpha >= 1/255 while the pixel is
    still transmissive), independent of the kernel under test.
    """
    order = torch.argsort(
        torch.where(proj.valid, proj.depths, torch.full_like(proj.depths, float("inf")))
    )
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) + 0.5,
        torch.arange(W, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    trans = torch.ones(H, W)
    counts = torch.zeros(proj.means2d.shape[0])
    for i in order:
        if not bool(proj.valid[i]):
            continue
        row0, row1, row2 = proj.transform[i]
        hu = row0 - xs.unsqueeze(-1) * row2
        hv = row1 - ys.unsqueeze(-1) * row2
        cross = torch.linalg.cross(hu, hv, dim=-1)
        wloc = cross[..., 2]
        degen = wloc.abs() < 1e-9
        safe = torch.where(degen, torch.ones_like(wloc), wloc)
        u, v = cross[..., 0] / safe, cross[..., 1] / safe
        rho_uv = torch.where(degen, torch.full_like(wloc, float("inf")), u * u + v * v)
        d = torch.stack([xs, ys], dim=-1) - proj.means2d[i]
        rho_screen = (d[..., 0] ** 2 + d[..., 1] ** 2) / (FILTER_SIZE**2)
        uv_active = rho_uv <= rho_screen
        rho = torch.where(uv_active, rho_uv, rho_screen)
        z_hit = torch.where(
            degen | ~uv_active,
            proj.depths[i].expand_as(wloc),
            row2[0] * u + row2[1] * v + row2[2],
        )
        alpha = (opacities[i] * torch.exp(-0.5 * rho)).clamp(max=0.99)
        alpha = torch.where(z_hit <= NEAR, torch.zeros_like(alpha), alpha)
        contributes = (trans >= 1e-4) & (alpha >= 1.0 / 255.0)
        counts[i] = contributes.sum()
        trans = trans * torch.where(contributes, 1 - alpha, torch.ones_like(alpha))
    return counts


def test_pixel_count_accum_counts_covered_pixels():
    """`pixel_count_accum` must count the pixels each gaussian composited
    into, on exactly the condition that feeds `abs_grad_accum`.

    It is the denominator that turns the AbsGS sum into a per-pixel mean
    (metalsplat.densify2dgs), so the two have to agree about which
    (pixel, gaussian) pairs count: a gaussian with a non-zero signal and a
    zero count would divide by zero, and a count that included pixels the
    gaussian never touched would understate dense regions.
    """
    n = 20
    proj, opacities, colors = _random_scene(n, seed=4)
    counts = torch.zeros(n, device="mps")
    absgrad = torch.zeros(n, device="mps")

    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(n, 9).to("mps").requires_grad_(),
        proj.normal.to("mps"),
        opacities.to("mps").requires_grad_(),
        colors.to("mps").requires_grad_(),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
        abs_grad_accum=absgrad,
        pixel_count_accum=counts,
    )
    out[0].sum().backward()
    torch.mps.synchronize()

    counts_cpu, absgrad_cpu = counts.cpu(), absgrad.cpu()
    assert counts_cpu.shape == (n,)
    assert (counts_cpu >= 0).all()
    assert counts_cpu.sum() > 0, "no gaussian covered any pixel"
    # Whole numbers: this is a tally, accumulated as float only because
    # Metal's atomic_fetch_add is float here.
    assert torch.allclose(counts_cpu, counts_cpu.round())
    # No pixel is counted twice per gaussian, so the tally cannot exceed
    # the number of pixels in the image.
    assert counts_cpu.max() <= W * H
    # The two accumulators must agree on which pairs contribute, or the
    # per-pixel mean divides by zero exactly where it matters most.
    assert not ((absgrad_cpu > 0) & (counts_cpu == 0)).any()


def test_pixel_count_accum_matches_a_brute_force_count():
    """Cross-checks the tally against the reference rasterizer's own
    per-pixel loop, so it measures coverage rather than merely being
    self-consistent.
    """
    n = 12
    proj, opacities, colors = _random_scene(n, seed=5)
    counts = torch.zeros(n, device="mps")

    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(n, 9).to("mps").requires_grad_(),
        proj.normal.to("mps"),
        opacities.to("mps").requires_grad_(),
        colors.to("mps").requires_grad_(),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
        pixel_count_accum=counts,
    )
    out[0].sum().backward()
    torch.mps.synchronize()

    expected = _reference_pixel_counts(proj, opacities)
    # Exact agreement is not expected at the alpha>=1/255 and T<1e-4
    # cutoffs, where GPU-vs-CPU float32 noise flips individual pixels --
    # the same boundary effect the forward/backward tests allow for.
    diff = (counts.cpu() - expected).abs()
    assert diff.max() <= 0.02 * expected.clamp_min(1.0).max(), (
        f"max |diff| {diff.max()} against counts up to {expected.max()}"
    )


def test_abs_grad_uses_the_mean_depth_not_the_hit_depth():
    """The AbsGS signal recovers d(means2d) from the transform's mean
    column via `row0[2] = means2d.x * z_mean`, so the scale factor is the
    *mean's* camera-space depth -- the same value for every pixel the
    gaussian covers -- not the per-pixel intersection depth `z_hit`.

    The two coincide for a splat parallel to the image plane, which is why
    a random scene barely distinguishes them. This pins a single pixel
    against a steeply tilted disk, where the ray hits ~25% deeper than the
    mean, and checks the accumulated signal against the identity computed
    from the reference's own `transform` gradient. One pixel and one
    gaussian, so the kernel's per-pixel term is the whole sum and the
    reference's summed gradient is directly comparable.
    """
    fx = fy = 50.0
    cx = cy = 0.5
    w = h = 1  # exactly one pixel, at (0.5, 0.5)
    ang = torch.tensor(1.0)  # ~57 degrees about the y axis
    quats = torch.tensor([[torch.cos(ang / 2), 0.0, torch.sin(ang / 2), 0.0]])
    z = 3.0
    means = torch.tensor([[8.0 / fx * z, 0.0, z]])  # center ~8px off the pixel
    scales = torch.tensor([[0.6, 0.6]])
    opacities = torch.tensor([0.8])
    colors = torch.tensor([[0.4, 0.6, 0.9]])

    proj = project_gaussians_2dgs_ref(
        means,
        scales,
        quats,
        torch.eye(3),
        torch.zeros(3),
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near=NEAR,
        eps2d=EPS2D,
    )
    assert bool(proj.valid[0]), "scene setup: the splat must survive culling"

    transform_ref = proj.transform.clone().requires_grad_()
    means2d_ref = proj.means2d.clone().requires_grad_()
    ref = rasterize_gaussians_2dgs_ref(
        means2d_ref,
        proj.depths,
        transform_ref,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        w,
        h,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    ref["image"].sum().backward()

    absgrad = torch.zeros(1, device="mps")
    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(1, 9).to("mps").requires_grad_(),
        proj.normal.to("mps"),
        opacities.to("mps").requires_grad_(),
        colors.to("mps").requires_grad_(),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        w,
        h,
        near=NEAR,
        filter_size=FILTER_SIZE,
        abs_grad_accum=absgrad,
    )
    out[0].sum().backward()
    torch.mps.synchronize()

    z_mean = proj.depths[0]
    mean_col = transform_ref.grad[0, :2, 2]  # d(row0[2]), d(row1[2])
    g_mean2d = means2d_ref.grad[0]  # screen-space fallback term, 0 when uv_active
    expected = torch.linalg.norm(g_mean2d + mean_col * z_mean)
    assert torch.allclose(absgrad.cpu()[0], expected, rtol=2e-3, atol=1e-7)

    # The setup has to actually separate the two depths, or this test would
    # pass against the z_hit scaling it exists to rule out.
    row0, row1, row2 = proj.transform[0]
    hu = 0.5 * row2 - row0
    hv = 0.5 * row2 - row1
    wloc = hu[0] * hv[1] - hu[1] * hv[0]
    u = (hu[1] * hv[2] - hu[2] * hv[1]) / wloc
    v = (hu[2] * hv[0] - hu[0] * hv[2]) / wloc
    z_hit = row2[0] * u + row2[1] * v + row2[2]
    assert z_hit > 1.2 * z_mean


def test_depth_gradient_flows_through_the_fallback():
    """An edge-on disk, where every pixel takes the z_hit fallback.

    `test_backward_matches_reference` covers this only by accident -- its
    random scenes need ~40 gaussians before any of them produces a
    degenerate or screen-space-winning pixel, and at n=1 the depth
    gradient is identically zero on both sides, so the comparison passes
    whether or not the kernel emits one. A disk rotated 90 degrees about
    the y axis has its plane containing the view direction, so the
    ray-splat intersection is unusable everywhere and `depths` is the
    *only* thing the depth map depends on.
    """
    ang = torch.tensor(torch.pi / 2)  # edge-on
    quats = torch.tensor([[torch.cos(ang / 2), 0.0, torch.sin(ang / 2), 0.0]])
    means = torch.tensor([[0.0, 0.0, 3.0]])
    scales = torch.tensor([[0.5, 0.5]])
    opacities = torch.tensor([0.9])
    colors = torch.tensor([[0.5, 0.4, 0.3]])

    proj = project_gaussians_2dgs_ref(
        means,
        scales,
        quats,
        torch.eye(3),
        torch.zeros(3),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
        eps2d=EPS2D,
    )
    assert bool(proj.valid[0]), "scene setup: the splat must survive culling"

    depths_ref = proj.depths.clone().requires_grad_()
    ref = rasterize_gaussians_2dgs_ref(
        proj.means2d,
        depths_ref,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    ref["depth"].sum().backward()

    depths_mps = proj.depths.to("mps").requires_grad_()
    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(1, 9).to("mps"),
        proj.normal.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        depths_mps,
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    out[1].sum().backward()
    torch.mps.synchronize()

    assert depths_ref.grad.abs().max() > 0.1, "scene setup: no fallback pixels"
    assert torch.allclose(depths_mps.grad.cpu(), depths_ref.grad, atol=1e-3, rtol=1e-3)


def test_screen_space_fallback_uses_the_paper_filter_width():
    """The screen-space fallback's width is `filter_size` pixels.

    The official rasterizer's `FilterSize = 0.707106` / `FilterInvSquare
    = 2.0f` makes this a sub-pixel anti-aliasing floor (variance 0.5px^2).
    Because `rho = min(rho_uv, rho_screen)`, it is also a lower bound on
    every splat's footprint, so the exact width decides how small a splat
    can usefully get -- worth pinning rather than leaving to whichever
    constant happened to be in scope.

    An edge-on disk is degenerate at every pixel, so `rho` is the fallback
    term everywhere and alpha is a pure function of the pixel's distance
    from the projected center.
    """
    ang = torch.tensor(torch.pi / 2)  # edge-on: the ray-splat solve degenerates
    quats = torch.tensor([[torch.cos(ang / 2), 0.0, torch.sin(ang / 2), 0.0]])
    means = torch.tensor([[0.0, 0.0, 3.0]])
    scales = torch.tensor([[0.5, 0.5]])
    opacity = 0.5
    opacities = torch.tensor([opacity])
    colors = torch.tensor([[1.0, 1.0, 1.0]])

    proj = project_gaussians_2dgs_ref(
        means,
        scales,
        quats,
        torch.eye(3),
        torch.zeros(3),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
        eps2d=EPS2D,
    )
    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(1, 9).to("mps"),
        proj.normal.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    torch.mps.synchronize()
    alpha = (1.0 - out[4]).cpu()  # single gaussian, so this is its alpha

    cx_px, cy_px = proj.means2d[0].tolist()

    # means2d lands on CX/CY (integers) while pixel centers sit at n+0.5,
    # so the nearest pixel is already sqrt(0.5) away -- compute each
    # pixel's true distance rather than assuming the offset is the radius.
    # Past ~2px the fallback is below the 1/255 compositing cutoff at this
    # opacity, alpha is exactly 0, and the sample pins nothing.
    def _sample(offset):
        col, row = int(cx_px - 0.5 + offset), int(cy_px - 0.5)
        d2 = (col + 0.5 - cx_px) ** 2 + (row + 0.5 - cy_px) ** 2
        return alpha[row, col], d2

    for offset in (0.0, 1.0, 2.0):
        observed, d2 = _sample(offset)
        expected = opacity * torch.exp(torch.tensor(-0.5 * d2 / FILTER_SIZE**2))
        assert torch.allclose(observed, expected, atol=2e-3), (
            f"offset {offset}px (d^2={d2}): {observed.item()} != {expected.item()}"
        )

    # And that this pins *this* width rather than passing for any filter.
    # At the nearest pixel (d^2 = 0.5) the three candidates are far apart:
    # 0.707px (variance 0.5) -> 0.303, eps2d=0.3 read as a variance -> 0.217,
    # a 2px filter -> 0.470. The atol above is 2e-3.
    observed, d2 = _sample(0.0)
    assert abs(d2 - 0.5) < 1e-6, d2
    for rejected in (0.3**0.5, 2.0):
        other = opacity * torch.exp(torch.tensor(-0.5 * d2 / rejected**2))
        assert abs(observed - other) > 0.05, (rejected, observed.item(), other.item())


def test_transmittance_cutoff_drops_the_same_gaussian_as_the_kernel():
    """The oracle and the kernel must agree on the *boundary* gaussian.

    The kernel computes `test_T = T * (1 - alpha)` and breaks before
    compositing when that falls below 1e-4, so the gaussian that would
    exhaust the pixel contributes nothing. Gating on the pre-update
    transmittance instead composites it. Four stacked alpha-0.99 disks put
    a pixel exactly on that boundary: T runs 1 -> 1e-2 -> 1e-4, and the
    third disk is the one the two rules disagree about.
    """
    n = 4
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0  # identity: fronto-parallel disks, normal along +z
    means = torch.stack(
        [torch.zeros(n), torch.zeros(n), torch.linspace(3.0, 3.3, n)], dim=-1
    )
    scales = torch.full((n, 2), 0.5)
    opacities = torch.full((n,), 0.999)  # alpha clamps to 0.99 at the center
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )

    proj = project_gaussians_2dgs_ref(
        means,
        scales,
        quats,
        torch.eye(3),
        torch.zeros(3),
        FX,
        FY,
        CX,
        CY,
        W,
        H,
        near=NEAR,
        eps2d=EPS2D,
    )
    assert bool(proj.valid.all()), "scene setup: every disk must survive culling"

    ref = rasterize_gaussians_2dgs_ref(
        proj.means2d,
        proj.depths,
        proj.transform,
        proj.normal,
        opacities,
        colors,
        proj.valid,
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    out = rasterize_gaussians_2dgs(
        proj.means2d.to("mps"),
        proj.transform.reshape(n, 9).to("mps"),
        proj.normal.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        filter_size=FILTER_SIZE,
    )
    torch.mps.synchronize()
    image, _depth, _normal, _distortion, final_T = (t.cpu() for t in out)

    # Far tighter than test_backward_matches_reference's tolerance, which
    # is loose enough to hide a whole dropped-vs-kept gaussian.
    assert torch.allclose(image, ref["image"], atol=1e-6, rtol=1e-4)
    assert torch.allclose(final_T, ref["final_T"], atol=1e-6, rtol=1e-4)

    # The third disk is blue; the rule the kernel uses drops it entirely,
    # so no blue reaches the center pixel.
    center = image[int(CY), int(CX)]
    assert center[2] < 1e-7, f"blue leaked into the center pixel: {center.tolist()}"


@pytest.mark.parametrize("tile_size", [4, 8])
def test_backward_independent_of_tile_size(tile_size):
    # A tile list longer than the tile_size^2 threads of a small threadgroup:
    # the backward's shared-memory staging must still cover every slot.
    from metalsplat import Camera, Gaussian2DModel, render_2dgs

    torch.manual_seed(0)
    model = Gaussian2DModel.random(n=600, bound=1.0, device="mps")
    with torch.no_grad():
        model.means[:, 2] += 3.0
        model.raw_scales.fill_(math.log(0.08))
        model.raw_opacities.fill_(-2.0)
    camera = Camera.identity(
        fx=40, fy=40, cx=16, cy=16, img_width=32, img_height=32
    ).to("mps")

    def grads(ts):
        model.zero_grad()
        aux = render_2dgs(model, camera, tile_size=ts, return_aux=True)
        loss = (
            aux.image.sum() + aux.depth.sum() + aux.normal.sum() + aux.distortion.sum()
        )
        loss.backward()
        torch.mps.synchronize()
        return [p.grad.cpu().clone() for p in model.parameters()]

    for got, expected in zip(grads(tile_size), grads(16)):
        assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)
