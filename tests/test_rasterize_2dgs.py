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
    assert torch.allclose(distortion, ref["distortion"], atol=1e-2, rtol=1e-2)
    assert torch.allclose(final_T, ref["final_T"], atol=1e-2, rtol=1e-2)


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
