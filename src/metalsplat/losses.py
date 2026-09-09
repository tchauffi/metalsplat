"""The standard 3D Gaussian Splatting training loss: L = (1-lambda)*L1 +
lambda*D-SSIM, D-SSIM = (1 - SSIM)/2 (Kerbl et al. 2023 use lambda=0.2).

SSIM is a local-window statistic computable via conv2d, which already runs
fine on MPS -- no custom Metal kernel needed here (same reasoning as
metalsplat.ops.tiling using plain torch ops rather than a kernel).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LAMBDA_DSSIM = 0.2


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device, dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] @ g1d[None, :]
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """pred, target: (H, W, C) in [0, 1]. Returns scalar mean SSIM over the image."""
    pred_c = pred.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    target_c = target.permute(2, 0, 1).unsqueeze(0)
    channels = pred_c.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_pred = F.conv2d(pred_c, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target_c, window, padding=pad, groups=channels)
    mu_pred_sq, mu_target_sq = mu_pred.pow(2), mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = (
        F.conv2d(pred_c * pred_c, window, padding=pad, groups=channels) - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target_c * target_c, window, padding=pad, groups=channels)
        - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred_c * target_c, window, padding=pad, groups=channels)
        - mu_pred_target
    )

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


def gaussian_splatting_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_dssim: float = DEFAULT_LAMBDA_DSSIM
) -> torch.Tensor:
    """pred, target: (H, W, C) in [0, 1]. The standard 3DGS training loss."""
    l1 = (pred - target).abs().mean()
    d_ssim = (1.0 - ssim(pred, target)) / 2.0
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * d_ssim
