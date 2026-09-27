"""Compiles and caches Metal Shading Language kernel libraries via torch.mps.

Kernels live as ``.metal`` source files next to this module and are compiled
at runtime through ``torch.mps.compile_shader`` (the OS's built-in Metal
compiler) the first time they're requested, then cached for the process
lifetime.

``compile_shader`` compiles a single source string with no file path, so
the Metal compiler cannot resolve a local ``#include "x.metal"`` itself.
This module expands those textually instead (each file at most once per
library), which is what lets helpers shared by several kernels -- see
``common.metal`` -- live in one place.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import torch

_KERNELS_DIR = Path(__file__).parent
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE)


def source(filename: str, _included: set[str] | None = None) -> str:
    """The source of kernels/``filename`` with its local includes expanded."""
    included = set() if _included is None else _included
    included.add(filename)
    path = _KERNELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No such kernel source: {path}")

    def expand(match: re.Match) -> str:
        name = match.group(1)
        if name in included:
            return f"// {name}: already included"
        return source(name, included)

    return _LOCAL_INCLUDE.sub(expand, path.read_text())


@functools.cache
def load(name: str):
    """Compile (or fetch from cache) the shader library ``<name>.metal``."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            f"MPS is not available on this machine; cannot compile Metal kernel '{name}'."
        )
    return torch.mps.compile_shader(source(f"{name}.metal"))
