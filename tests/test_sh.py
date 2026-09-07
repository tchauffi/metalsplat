import pytest
import torch

from metalsplat.ops.sh import eval_sh
from metalsplat.reference.sh_ref import NUM_SH_COEFFS
from metalsplat.reference.sh_ref import eval_sh as eval_sh_ref

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def _random_inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    sh_coeffs = torch.randn(n, NUM_SH_COEFFS, 3, generator=g)
    raw_dirs = torch.randn(n, 3, generator=g)
    dirs = raw_dirs / raw_dirs.norm(dim=-1, keepdim=True)
    return sh_coeffs, dirs


@pytest.mark.parametrize("n", [1, 8, 33])
def test_forward_matches_reference(n):
    sh_coeffs, dirs = _random_inputs(n)

    ref = eval_sh_ref(sh_coeffs, dirs)
    kernel = eval_sh(sh_coeffs.to("mps"), dirs.to("mps"))
    torch.mps.synchronize()

    assert torch.allclose(kernel.cpu(), ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("n", [1, 8, 33])
def test_backward_matches_reference(n):
    sh_coeffs, dirs = _random_inputs(n)

    sh_ref = sh_coeffs.clone().requires_grad_()
    dirs_ref = dirs.clone().requires_grad_()
    out_ref = eval_sh_ref(sh_ref, dirs_ref)

    g = torch.Generator().manual_seed(99)
    upstream = torch.randn(n, 3, generator=g)
    (out_ref * upstream).sum().backward()

    sh_mps = sh_coeffs.to("mps").requires_grad_()
    dirs_mps = dirs.to("mps").requires_grad_()
    out_mps = eval_sh(sh_mps, dirs_mps)
    (out_mps * upstream.to("mps")).sum().backward()
    torch.mps.synchronize()

    assert torch.allclose(sh_mps.grad.cpu(), sh_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dirs_mps.grad.cpu(), dirs_ref.grad, atol=1e-3, rtol=1e-3)
