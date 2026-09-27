"""Checks for tensors handed to the Metal kernels.

The kernels read and write raw float32 buffers, indexing them by gaussian
id times a fixed stride, and have no bounds checks of their own. A tensor
of the wrong shape or dtype is therefore not an error there: it is the
wrong memory. An int64 background reads as a denormal (renders black), a
float64 mean reads as two garbage floats, and an accumulator shorter than
the gaussian count takes atomic adds past its end. These checks turn all
of that into an exception at the op boundary instead.

Differentiable inputs are checked, not converted: casting inside an
autograd.Function would hand back float32 gradients for a float64 input,
which autograd rejects anyway, and a silent cast would hide the caller's
mistake. Non-differentiable, read-only inputs (the background colour) are
converted, since nothing depends on their identity. In-place accumulators
are checked strictly, including contiguity, because converting them would
make the kernel update a copy the caller never sees.
"""

from __future__ import annotations

import torch


def check_float32(
    name: str, tensor: torch.Tensor, shape: tuple[int, ...], device: torch.device
) -> None:
    """Raises unless `tensor` is float32, on `device`, with exactly `shape`."""
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} is on {tensor.device}, expected {device}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    """Raises unless `tensor` has exactly `shape` (any dtype/device)."""
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def kernel_background(
    background: torch.Tensor | None, device: torch.device
) -> torch.Tensor:
    """The (3,) float32 background on `device` the rasterizers bind directly."""
    if background is None:
        return torch.zeros(3, device=device, dtype=torch.float32)
    if background.numel() != 3:
        raise ValueError(
            f"background must have 3 elements, got shape {tuple(background.shape)}"
        )
    return background.reshape(3).to(device=device, dtype=torch.float32).contiguous()


def check_accumulator(
    name: str, accum: torch.Tensor | None, n: int, device: torch.device
) -> None:
    """Raises unless `accum` (if given) can take the kernel's in-place adds
    for `n` gaussians: float32, on `device`, shape (n,), contiguous.

    The usual way to trip this is densifying, seeding or pruning without
    reallocating the accumulator for the new gaussian count.
    """
    if accum is None:
        return
    check_float32(name, accum, (n,), device)
    if not accum.is_contiguous():
        raise ValueError(
            f"{name} is updated in place by the kernel and must be contiguous"
        )
