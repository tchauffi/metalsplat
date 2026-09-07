"""Compiles and caches Metal Shading Language kernel libraries via torch.mps.

Kernels live as ``.metal`` source files next to this module and are compiled
at runtime through ``torch.mps.compile_shader`` (the OS's built-in Metal
compiler) the first time they're requested, then cached for the process
lifetime.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_KERNELS_DIR = Path(__file__).parent


@functools.cache
def load(name: str):
    """Compile (or fetch from cache) the shader library ``<name>.metal``."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            f"MPS is not available on this machine; cannot compile Metal kernel '{name}'."
        )
    path = _KERNELS_DIR / f"{name}.metal"
    if not path.exists():
        raise FileNotFoundError(f"No such kernel source: {path}")
    return torch.mps.compile_shader(path.read_text())
