import pytest
import torch

from metalsplat.kernels import _loader

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


@mps_only
def test_load_missing_kernel_raises():
    _loader.load.cache_clear()
    with pytest.raises(FileNotFoundError):
        _loader.load("does_not_exist")


def test_local_include_is_inlined_once():
    src = _loader.source("project.metal")
    assert _loader._LOCAL_INCLUDE.search(src) is None  # no directive left
    assert src.count("inline float3x3 quat_to_rotmat(") == 1
    assert "#include <metal_stdlib>" in src  # system includes are left alone


def test_nested_and_repeated_includes(tmp_path, monkeypatch):
    monkeypatch.setattr(_loader, "_KERNELS_DIR", tmp_path)
    (tmp_path / "a.metal").write_text('#include "b.metal"\n#include "c.metal"\nA\n')
    (tmp_path / "b.metal").write_text('#include "c.metal"\nB\n')
    (tmp_path / "c.metal").write_text("C\n")
    lines = [line for line in _loader.source("a.metal").splitlines() if line]
    assert [line for line in lines if not line.startswith("//")] == ["C", "B", "A"]
