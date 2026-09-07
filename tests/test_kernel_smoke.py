"""Pins the torch.mps.compile_shader capabilities the whole package relies on:
atomic float accumulation (needed for backward-pass gradient scatter-add) and
explicit 2D grid/threadgroup dispatch (needed for tile-based rasterization).
If a future torch upgrade breaks either, this should fail loudly here instead
of surfacing as a mysterious numerical bug deep in the rasterizer.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

_SRC = """
#include <metal_atomic>
using namespace metal;

kernel void add_atomic(device atomic_float* out,
                        constant float& val,
                        uint idx [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&out[idx % 4], val, memory_order_relaxed);
}

kernel void grid2d(device float* out,
                    uint2 gid [[thread_position_in_grid]],
                    uint2 gsize [[threads_per_grid]]) {
    out[gid.y * gsize.x + gid.x] = float(gid.x) * 100.0 + float(gid.y);
}
"""


@pytest.fixture(scope="module")
def lib():
    return torch.mps.compile_shader(_SRC)


def test_atomic_float_accumulation(lib):
    out = torch.zeros(4, device="mps")
    lib.add_atomic(out, 1.0, threads=16)
    torch.mps.synchronize()
    assert torch.allclose(out.cpu(), torch.full((4,), 4.0))


def test_explicit_2d_grid_dispatch(lib):
    width, height = 8, 4
    out = torch.zeros(height, width, device="mps")
    lib.grid2d(out, threads=(width, height), group_size=(4, 4))
    torch.mps.synchronize()
    expected = torch.tensor(
        [[x * 100.0 + y for x in range(width)] for y in range(height)]
    )
    assert torch.allclose(out.cpu(), expected)
