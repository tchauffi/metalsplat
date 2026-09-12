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
    return camera, rendered_depth, rendered_normal


def test_normal_consistency_near_zero_for_consistent_frontoparallel_plane():
    camera, depth, normal = _fronto_parallel_scene()
    loss = normal_consistency_loss(normal, depth, camera)
    assert loss.item() < 1e-4


def test_normal_consistency_positive_for_wrong_normal():
    camera, depth, normal = _fronto_parallel_scene()
    wrong_normal = torch.zeros_like(normal)
    wrong_normal[..., 0] = 1.0  # orthogonal to the true (0, 0, -1) normal
    loss = normal_consistency_loss(wrong_normal, depth, camera)
    assert loss.item() > 0.9  # 1 - dot(perpendicular vectors) == 1


def test_normal_consistency_gradients_flow_to_both_normal_and_depth():
    # The pseudo-normal (derived from depth) is *not* detached (matching
    # the official 2DGS reference implementation, see the function's
    # docstring): gradient reaches both `rendered_normal` (the direct
    # comparison) and `rendered_depth` (via the pseudo-normal's own
    # construction).
    camera, depth, normal = _fronto_parallel_scene()
    # A tilted plane so depth actually varies spatially -- a perfectly
    # flat depth's pseudo-normal gradient w.r.t. depth is degenerate at
    # the cross product's own critical point.
    h, w = depth.shape
    tilt = torch.linspace(-0.5, 0.5, w).unsqueeze(0).expand(h, w)
    depth = (depth + tilt).clone().requires_grad_()
    normal = normal.clone().requires_grad_()
    loss = normal_consistency_loss(normal, depth, camera)
    loss.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert normal.grad is not None and torch.isfinite(normal.grad).all()
