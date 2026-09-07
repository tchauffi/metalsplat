import torch

from metalsplat.losses import gaussian_splatting_loss, ssim


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
