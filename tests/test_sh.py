import math

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


@pytest.mark.parametrize("active_degree", [0, 1, 2, 3])
def test_active_degree_matches_reference(active_degree):
    sh_coeffs, dirs = _random_inputs(16)

    ref = eval_sh_ref(sh_coeffs, dirs, active_degree)
    kernel = eval_sh(sh_coeffs.to("mps"), dirs.to("mps"), active_degree)
    torch.mps.synchronize()

    assert torch.allclose(kernel.cpu(), ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("active_degree", [0, 1, 2])
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


def test_sh_basis_is_orthonormal_on_the_sphere():
    """Independent check on the basis constants themselves.

    The kernel is tested against sh_ref, so a wrong constant in *both*
    would agree with itself and pass. Real spherical harmonics are
    orthonormal over the unit sphere:

        (1/4pi) * integral over S^2 of Y_i(d) Y_j(d) dd  =  delta_ij

    which Monte-Carlo integration over uniformly sampled directions
    verifies without reference to any other implementation. This is what
    catches a mistyped degree-3 coefficient.
    """
    torch.manual_seed(0)
    m = 400_000
    d = torch.randn(m, 3, dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)

    # Recover each basis function by evaluating with a one-hot coefficient
    # vector: eval_sh is linear in the coefficients, so this reads out Y_i.
    basis = []
    for i in range(NUM_SH_COEFFS):
        coeffs = torch.zeros(m, NUM_SH_COEFFS, 3, dtype=torch.float64)
        coeffs[:, i, 0] = 1.0
        basis.append(eval_sh_ref(coeffs, d, active_degree=3)[:, 0])
    y = torch.stack(basis, dim=-1)  # (m, 16)

    gram = 4 * math.pi * (y[:, :, None] * y[:, None, :]).mean(dim=0)  # ~ delta_ij
    identity = torch.eye(NUM_SH_COEFFS, dtype=torch.float64)
    err = (gram - identity).abs().max().item()
    assert err < 0.02, f"SH basis is not orthonormal, max |gram - I| = {err:.4f}"


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_model_supports_every_degree(degree):
    from metalsplat.gaussians import GaussianModel
    from metalsplat.reference.sh_ref import num_sh_coeffs

    n = 12
    means = torch.randn(n, 3, device="mps")
    model = GaussianModel(means, colors=torch.rand(n, 3, device="mps"), sh_degree=degree).to("mps")

    if degree == 0:
        assert model.colors.shape == (n, 3)
        return

    assert model.raw_sh.shape == (n, num_sh_coeffs(degree), 3)
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, device="mps"), dim=-1)
    colors = model.colors_from_view(dirs)
    colors.pow(2).sum().backward()
    torch.mps.synchronize()

    assert colors.shape == (n, 3)
    assert torch.isfinite(colors).all()
    assert model.raw_sh.grad is not None and torch.isfinite(model.raw_sh.grad).all()


def test_degree_3_uses_all_sixteen_coefficients():
    # A degree-3 model must give every one of its 16 coefficients real
    # gradient -- if the top band were skipped, coefficients 9..15 would sit
    # at zero forever and degree 3 would silently be degree 2.
    sh_coeffs, dirs = _random_inputs(64, seed=3)
    sh_mps = sh_coeffs.to("mps").requires_grad_()
    eval_sh(sh_mps, dirs.to("mps"), 3).pow(2).sum().backward()
    torch.mps.synchronize()

    per_coeff = sh_mps.grad.abs().sum(dim=(0, 2))
    assert per_coeff.shape == (16,)
    assert (per_coeff > 0).all(), f"coefficients with no gradient: {(per_coeff == 0).nonzero().flatten().tolist()}"
