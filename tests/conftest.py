import functools
import os

import torch

if os.environ.get("GITHUB_ACTIONS") == "true":
    # GitHub-hosted macOS runners report MPS as available but the GPU is not
    # actually usable there -- any real allocation throws "MPS backend out
    # of memory" for even a few hundred bytes (actions/runner-images#9918).
    # Force it off so the existing `skipif(not torch.backends.mps.is_available())`
    # guards skip cleanly instead of failing.
    #
    # functools.wraps preserves `__wrapped__`: torch._dynamo's import-time
    # constant-folding list holds the original lru_cache-wrapped function and
    # accesses that attribute unconditionally, so a bare replacement breaks
    # the first `torch.optim` import.
    @functools.wraps(torch.backends.mps.is_available)
    def _mps_unavailable():
        return False

    torch.backends.mps.is_available = _mps_unavailable
