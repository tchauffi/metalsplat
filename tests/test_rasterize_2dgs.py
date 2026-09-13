import pytest
import torch

from metalsplat.ops.rasterize_2dgs import rasterize_gaussians_2dgs
from metalsplat.reference.project_2dgs_ref import (
    project_gaussians_2dgs as project_gaussians_2dgs_ref,
)
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
EPS2D = 0.3


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
        eps2d=EPS2D,
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
        eps2d=EPS2D,
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
    # Distortion needs its own, much tighter bound: it runs on *normalized*
    # depth (see rasterize_2dgs_ref's module docstring), so its values sit
    # around 1e-2 rather than O(1) like the maps above -- the shared 1e-2
    # tolerance would pass no matter what the kernel computed. Measured max
    # absolute deviation here is ~1e-7.
    assert torch.allclose(distortion, ref["distortion"], atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 5, 40])
def test_backward_matches_reference(n):
    proj, opacities, colors = _random_scene(n, seed=1)
    n_transform = proj.transform.reshape(n, 9)

    means2d_ref = proj.means2d.clone().requires_grad_()
    transform_ref = proj.transform.clone().requires_grad_()
    normal_ref = proj.normal.clone().requires_grad_()
    opacities_ref = opacities.clone().requires_grad_()
    colors_ref = colors.clone().requires_grad_()

    ref = rasterize_gaussians_2dgs_ref(
        means2d_ref,
        proj.depths,
        transform_ref,
        normal_ref,
        opacities_ref,
        colors_ref,
        proj.valid,
        W,
        H,
        near=NEAR,
        eps2d=EPS2D,
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

    out = rasterize_gaussians_2dgs(
        means2d_mps,
        transform_mps,
        normal_mps,
        opacities_mps,
        colors_mps,
        proj.depths.to("mps"),
        proj.radii.to("mps"),
        proj.valid.to("mps"),
        proj.conics.to("mps"),
        W,
        H,
        near=NEAR,
        eps2d=EPS2D,
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
        eps2d=EPS2D,
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
        eps2d=EPS2D,
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
        eps2d=EPS2D,
    )
    (out[3] * up_dist.to("mps")).sum().backward()
    torch.mps.synchronize()

    # The gradients themselves are O(0.1-1) even though the distortion map
    # is small, so these are ordinary tolerances, not inflated ones.
    # Measured max deviation ~6e-6.
    assert opacities_ref.grad.abs().max() > 1e-3, "scene exercises no distortion"
    assert torch.allclose(
        transform_mps.grad.cpu(), transform_ref.grad.reshape(n, 9), atol=1e-3, rtol=1e-2
    )
    assert torch.allclose(
        opacities_mps.grad.cpu(), opacities_ref.grad, atol=1e-3, rtol=1e-2
    )
    assert torch.allclose(
        means2d_mps.grad.cpu(), means2d_ref.grad, atol=1e-3, rtol=1e-2
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
        rho_screen = (d[..., 0] ** 2 + d[..., 1] ** 2) / EPS2D
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
        eps2d=EPS2D,
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
        eps2d=EPS2D,
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
