import pytest
import torch

from metalsplat.camera import Camera
from metalsplat.losses import (
    distortion_loss,
    gaussian_splatting_loss,
    normal_consistency_loss,
    ssim,
)


def _random_image(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(h, w, 3, generator=g)


def test_ssim_identical_images_is_one():
    img = _random_image()
    assert torch.allclose(ssim(img, img), torch.tensor(1.0), atol=1e-5)


def test_ssim_decreases_with_noise():
    img = _random_image()
    g = torch.Generator().manual_seed(1)
    noisy = (img + torch.randn(img.shape, generator=g) * 0.3).clamp(0, 1)

    assert ssim(img, noisy) < 0.99


def test_ssim_lower_for_more_different_images():
    img = _random_image()
    slightly_off = (img + 0.05).clamp(0, 1)
    very_off = 1.0 - img  # inverted, maximally different

    assert ssim(img, very_off) < ssim(img, slightly_off)


def test_loss_is_zero_for_identical_images():
    img = _random_image()
    loss = gaussian_splatting_loss(img, img)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_loss_gradients_flow():
    img = _random_image()
    target = _random_image(seed=1)
    pred = img.clone().requires_grad_()

    loss = gaussian_splatting_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_lambda_zero_matches_plain_l1():
    pred = _random_image()
    target = _random_image(seed=1)

    loss = gaussian_splatting_loss(pred, target, lambda_dssim=0.0)
    l1 = (pred - target).abs().mean()
    assert torch.allclose(loss, l1)


def test_distortion_loss_is_mean():
    m = torch.rand(8, 8)
    assert torch.allclose(distortion_loss(m), m.mean())


def test_distortion_loss_gradients_flow():
    m = torch.rand(8, 8, requires_grad=True)
    distortion_loss(m).backward()
    assert m.grad is not None
    assert torch.allclose(m.grad, torch.full_like(m, 1.0 / m.numel()))


def _fronto_parallel_scene(h=16, w=16, depth=5.0, fx=50.0, fy=50.0):
    camera = Camera.identity(
        fx=fx, fy=fy, cx=w / 2, cy=h / 2, img_width=w, img_height=h
    )
    rendered_depth = torch.full((h, w), depth)
    # A depth plane exactly perpendicular to the optical axis has a
    # constant camera-space (and world-space, identity camera) normal of
    # (0, 0, -1) -- facing back at the camera, matching project_2dgs_ref's
    # camera-facing convention.
    rendered_normal = torch.zeros(h, w, 3)
    rendered_normal[..., 2] = -1.0
    # Fully opaque: both rasterizer outputs are already "normalized", so
    # this scene isolates the geometry from the alpha handling that
    # test_normal_consistency_requires_expected_depth covers.
    alpha = torch.ones(h, w)
    return camera, rendered_depth, rendered_normal, alpha


def test_normal_consistency_near_zero_for_consistent_frontoparallel_plane():
    camera, depth, normal, alpha = _fronto_parallel_scene()
    loss = normal_consistency_loss(normal, depth, alpha, camera)
    assert loss.item() < 1e-4


def test_normal_consistency_positive_for_wrong_normal():
    camera, depth, normal, alpha = _fronto_parallel_scene()
    wrong_normal = torch.zeros_like(normal)
    wrong_normal[..., 0] = 1.0  # orthogonal to the true (0, 0, -1) normal
    loss = normal_consistency_loss(wrong_normal, depth, alpha, camera)
    assert loss.item() > 0.9  # 1 - dot(perpendicular vectors) == 1


def test_normal_consistency_gradients_flow_to_both_normal_and_depth():
    # The pseudo-normal (derived from depth) is *not* detached (matching
    # the official 2DGS reference implementation, see the function's
    # docstring): gradient reaches both `rendered_normal` (the direct
    # comparison) and `rendered_depth` (via the pseudo-normal's own
    # construction).
    camera, depth, normal, alpha = _fronto_parallel_scene()
    # A tilted plane so depth actually varies spatially -- a perfectly
    # flat depth's pseudo-normal gradient w.r.t. depth is degenerate at
    # the cross product's own critical point.
    h, w = depth.shape
    tilt = torch.linspace(-0.5, 0.5, w).unsqueeze(0).expand(h, w)
    depth = (depth + tilt).clone().requires_grad_()
    normal = normal.clone().requires_grad_()
    loss = normal_consistency_loss(normal, depth, alpha, camera)
    loss.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert normal.grad is not None and torch.isfinite(normal.grad).all()


def _partially_transparent_plane(h=16, w=16, depth=5.0):
    """The same fronto-parallel plane, but semi-transparent with a
    horizontal alpha ramp -- so the rasterizer's *raw* outputs (which are
    alpha-weighted sums) vary spatially even though the geometry doesn't.
    """
    camera = Camera.identity(
        fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, img_width=w, img_height=h
    )
    alpha = torch.linspace(0.2, 1.0, w).unsqueeze(0).expand(h, w).contiguous()
    rendered_depth = alpha * depth  # sum_k w_k*z_k for a single flat surface
    rendered_normal = torch.zeros(h, w, 3)
    rendered_normal[..., 2] = -1.0
    rendered_normal = rendered_normal * alpha[..., None]
    return camera, rendered_depth, rendered_normal, alpha


def test_normal_consistency_requires_expected_depth_not_alpha_weighted_sum():
    """The depth must be divided by alpha before unprojection.

    Geometry here is a perfectly flat fronto-parallel plane, so the correct
    pseudo-normal is constant and exactly anti-parallel to the camera axis,
    and the loss reduces to the analytic `mean(1 - alpha^2)` (one alpha
    from the un-normalized rendered normal, one from the alpha-scaled
    pseudo-normal). Feeding the raw alpha-weighted depth sum instead makes
    the unprojected points follow the *alpha ramp* rather than the surface,
    tilting the pseudo-normal and inflating the loss well past that.
    """
    camera, depth, normal, alpha = _partially_transparent_plane()
    loss = normal_consistency_loss(normal, depth, alpha, camera)

    interior_alpha = alpha[1:-1, 1:-1]
    expected = (1.0 - interior_alpha**2).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-4)

    # Sanity: the un-normalized variant this guards against is clearly
    # distinguishable, not within tolerance of the right answer.
    wrong = normal_consistency_loss(normal, depth, torch.ones_like(alpha), camera)
    assert abs(wrong.item() - expected.item()) > 0.05


def test_normal_consistency_gradient_scales_with_alpha():
    """Scaling the pseudo-normal by alpha is what makes the regularizer
    fade out where little was actually rendered. Without it the target
    stays unit-length no matter how transparent the pixel is, so the loss
    keeps pulling the rendered normal (and through it, opacity) upward in
    parts of the frame that are barely covered.

    Checked as a *ratio* between two otherwise-identical scenes: the
    gradient w.r.t. the rendered normal is exactly the (alpha-scaled)
    pseudo-normal, so halving alpha must halve it. Note this cannot be
    tested on a fully empty region -- there the depth is 0, the
    unprojected points collapse to the origin, and the pseudo-normal is
    already 0 from the degenerate cross product whether or not the alpha
    scaling is present.
    """
    grads = {}
    for a in (1.0, 0.02):
        h = w = 16
        camera = Camera.identity(
            fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, img_width=w, img_height=h
        )
        alpha = torch.full((h, w), a)
        depth = alpha * 5.0
        normal = torch.zeros(h, w, 3)
        normal[..., 2] = -1.0
        normal = (normal * alpha[..., None]).requires_grad_()
        normal_consistency_loss(normal, depth, alpha, camera).backward()
        grads[a] = normal.grad[1:-1, 1:-1].abs().max().item()

    assert grads[1.0] > 0.0
    assert grads[0.02] / grads[1.0] == pytest.approx(0.02, rel=1e-3)


def test_normal_consistency_depth_gradient_bounded_at_uncovered_pixels():
    """Uncovered pixels must not amplify depth gradients.

    The expected depth is `rendered_depth / alpha`, so a naive floor on the
    denominator (`alpha.clamp_min(eps)`) leaves a 1/eps factor on the
    gradient path at pixels where alpha is 0 -- on a real garden view that
    produced depth gradients of magnitude 1e8 flowing back into the
    rasterizer from parts of the frame where nothing was rendered. Guarding
    the division with a `where` instead blocks that branch entirely, which
    is also what the official implementation's `nan_to_num(0/0)` does.
    """
    h = w = 16
    camera = Camera.identity(
        fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, img_width=w, img_height=h
    )
    # Half covered, half empty -- the interesting pixels are the uncovered
    # ones adjacent to the boundary, where the finite-difference stencil
    # straddles both regions.
    alpha = torch.zeros(h, w)
    alpha[:, w // 2 :] = 0.9
    depth = alpha * 5.0
    normal = torch.zeros(h, w, 3)
    normal[..., 2] = -1.0
    normal = normal * alpha[..., None]

    depth = depth.clone().requires_grad_()
    normal_consistency_loss(normal, depth, alpha, camera).backward()

    assert torch.isfinite(depth.grad).all()
    # Comfortably above any legitimate value here (measured ~1e-3 on a real
    # 648x420 garden view) and far below the ~1e8 the unguarded divide gave.
    assert depth.grad.abs().max() < 1.0
    # Uncovered pixels contribute no depth gradient at all.
    assert torch.count_nonzero(depth.grad[:, : w // 2]) == 0
