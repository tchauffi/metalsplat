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


@pytest.mark.parametrize("active_degree", [0, 1, 2])
def test_active_degree_matches_reference(active_degree):
    sh_coeffs, dirs = _random_inputs(16)

    ref = eval_sh_ref(sh_coeffs, dirs, active_degree)
    kernel = eval_sh(sh_coeffs.to("mps"), dirs.to("mps"), active_degree)
    torch.mps.synchronize()

    assert torch.allclose(kernel.cpu(), ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("active_degree", [0, 1])
def test_inactive_bands_get_zero_gradient(active_degree):
    # The point of progressive growth: coefficients above the active degree
    # must be *frozen*, not merely zero-initialised.
    sh_coeffs, dirs = _random_inputs(12)
    sh_mps = sh_coeffs.to("mps").requires_grad_()

    out = eval_sh(sh_mps, dirs.to("mps"), active_degree)
    out.sum().backward()
    torch.mps.synchronize()

    grad = sh_mps.grad.cpu()
    first_inactive = (active_degree + 1) ** 2
    assert grad[:, :first_inactive, :].abs().sum() > 0  # active bands do learn
    assert torch.count_nonzero(grad[:, first_inactive:, :]) == 0


def test_degree_zero_is_view_independent():
    sh_coeffs, dirs = _random_inputs(8)
    other = -dirs  # look from the opposite side

    a = eval_sh(sh_coeffs.to("mps"), dirs.to("mps"), 0)
    b = eval_sh(sh_coeffs.to("mps"), other.to("mps"), 0)
    torch.mps.synchronize()

    assert torch.allclose(a.cpu(), b.cpu(), atol=1e-6)
