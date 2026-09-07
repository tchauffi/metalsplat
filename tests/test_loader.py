import pytest

from metalsplat.kernels import _loader


def test_load_missing_kernel_raises():
    _loader.load.cache_clear()
    with pytest.raises(FileNotFoundError):
        _loader.load("does_not_exist")
