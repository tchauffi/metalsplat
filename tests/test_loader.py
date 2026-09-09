import pytest
import torch

from metalsplat.kernels import _loader

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def test_load_missing_kernel_raises():
    _loader.load.cache_clear()
    with pytest.raises(FileNotFoundError):
        _loader.load("does_not_exist")
