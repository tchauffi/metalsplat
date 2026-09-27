"""Pure-PyTorch brute-force reference rasterizer (alpha-compositing).

Not tile-based and not remotely fast (every pixel walks every gaussian in
depth order) -- this is deliberately the simplest possible correct
implementation of front-to-back alpha compositing, used only as the
executable spec / gradient oracle for the Metal rasterize kernel on small
test scenes. It is plain differentiable torch ops, so torch.autograd gives
its backward for free.
"""

from __future__ import annotations

import torch


def rasterize_gaussians(
    means2d: torch.Tensor,  # (N, 2)
    depths: torch.Tensor,  # (N,)
    conics: torch.Tensor,  # (N, 3) a, b, c
    opacities: torch.Tensor,  # (N,)
    colors: torch.Tensor,  # (N, 3)
    valid: torch.Tensor,  # (N,) bool-ish
    img_width: int,
    img_height: int,
    background: torch.Tensor | None = None,  # (3,)
    return_depth: bool = False,
):
    """Returns the (H, W, 3) image, or `(image, depth)` if `return_depth`.

    `depth` is the alpha-weighted expected depth per pixel, i.e. the same
    front-to-back compositing weights the colour uses but accumulating each
    gaussian's camera-space z instead of its colour. Pixels nothing covers
    stay 0 (divide by the accumulated alpha, 1 - final_T, to normalise).
    """
    device, dtype = means2d.device, means2d.dtype
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)

    valid_mask = valid > 0.5 if valid.dtype != torch.bool else valid
    order = torch.argsort(
        torch.where(valid_mask, depths, torch.full_like(depths, float("inf")))
    )

    ys, xs = torch.meshgrid(
        torch.arange(img_height, device=device, dtype=dtype) + 0.5,
        torch.arange(img_width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xs, ys], dim=-1)  # (H, W, 2)

    image = torch.zeros(img_height, img_width, 3, device=device, dtype=dtype)
    depth_map = torch.zeros(img_height, img_width, device=device, dtype=dtype)
    trans = torch.ones(img_height, img_width, device=device, dtype=dtype)
    # Sticky per-pixel "this pixel is finished" flag, standing in for the
    # kernel's `break`: this loop is vectorized over pixels and cannot break
    # per pixel.
    stopped = torch.zeros(img_height, img_width, device=device, dtype=torch.bool)

    for i in order:
        if not bool(valid_mask[i]):
            continue
        d = pixels - means2d[i]  # (H, W, 2)
        a, b, c = conics[i]
        power = -0.5 * (
            a * d[..., 0] ** 2 + 2 * b * d[..., 0] * d[..., 1] + c * d[..., 1] ** 2
        )
        alpha = (opacities[i] * torch.exp(power)).clamp(max=0.99)
        # Match the kernel's negligible-contribution cutoff and transmittance
        # early-termination exactly (not just approximately), so this stays
        # a precise oracle rather than a merely-close one. The kernel
        # computes `test_T = T * (1 - alpha)` and breaks *before* compositing
        # when that falls below 1e-4, so the gaussian that would exhaust the
        # pixel contributes nothing. Gating on the pre-update `trans` instead
        # composites it, leaving the oracle up to ~1% off the kernel on
        # saturated pixels. A too-faint gaussian is skipped but does *not*
        # finish the pixel (`continue`, not `break`). Same as
        # rasterize_2dgs_ref.
        visible = (alpha >= 1.0 / 255.0) & ~stopped
        exhausts = visible & (trans * (1.0 - alpha) < 1e-4)
        alpha_eff = torch.where(visible & ~exhausts, alpha, torch.zeros_like(alpha))
        weight = trans * alpha_eff
        image = image + weight[..., None] * colors[i]
        depth_map = depth_map + weight * depths[i]
        trans = trans * (1 - alpha_eff)
        stopped = stopped | exhausts

    image = image + trans[..., None] * background
    if return_depth:
        return image, depth_map
    return image
