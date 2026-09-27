"""torch.autograd.Function wrapping the Metal SH-evaluation kernel
(metalsplat.kernels.sh). Forward/backward math matches
metalsplat.reference.sh_ref.eval_sh exactly -- validated against it in
tests/test_sh.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.ops._validate import check_float32
from metalsplat.reference.sh_ref import MAX_SH_DEGREE, num_sh_coeffs


def buffer_sh_degree(num_coeffs: int) -> int:
    """The SH degree a buffer of `num_coeffs` coefficients per channel holds.

    Raises ValueError unless it is exactly (degree+1)^2 for a degree in
    0..MAX_SH_DEGREE.
    """
    for degree in range(MAX_SH_DEGREE + 1):
        if num_sh_coeffs(degree) == num_coeffs:
            return degree
    raise ValueError(
        f"{num_coeffs} SH coefficients per channel is not (degree+1)^2 for any "
        f"degree in 0..{MAX_SH_DEGREE}"
    )


class EvalSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sh_coeffs: torch.Tensor, dirs: torch.Tensor, active_degree: int):
        n = sh_coeffs.shape[0]
        num_coeffs = sh_coeffs.shape[1]
        device = sh_coeffs.device
        sh_c = sh_coeffs.contiguous()
        dirs_c = dirs.contiguous()
        out_color = torch.empty(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("sh")
            lib.sh_forward(
                sh_c, dirs_c, int(active_degree), int(num_coeffs), out_color, threads=n
            )

        ctx.save_for_backward(sh_c, dirs_c)
        ctx.n = n
        ctx.num_coeffs = num_coeffs
        ctx.active_degree = active_degree
        return out_color

    @staticmethod
    def backward(ctx, grad_color):
        sh_coeffs, dirs = ctx.saved_tensors
        n = ctx.n
        device = sh_coeffs.device

        d_sh = torch.zeros(n, ctx.num_coeffs, 3, device=device, dtype=torch.float32)
        d_dirs = torch.zeros(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("sh")
            lib.sh_backward(
                sh_coeffs,
                dirs,
                grad_color.contiguous(),
                int(ctx.active_degree),
                int(ctx.num_coeffs),
                d_sh,
                d_dirs,
                threads=n,
            )

        return d_sh, d_dirs, None  # active_degree is not differentiable


def eval_sh(
    sh_coeffs: torch.Tensor, dirs: torch.Tensor, active_degree: int | None = None
) -> torch.Tensor:
    """Evaluates degree<=3 spherical harmonics color using the Metal kernel.

    sh_coeffs: (N, K, 3) where K = (degree+1)^2 for the model's own degree,
    dirs: (N, 3) unit view directions -> (N, 3) color. `active_degree`
    restricts evaluation to that degree; coefficients above it are skipped
    in both passes, so they receive zero gradient. It defaults to the degree
    K holds, and may not exceed it: the kernel indexes each gaussian's
    coefficients by degree, so a higher one would read the next gaussian's.
    """
    if sh_coeffs.dim() != 3 or sh_coeffs.shape[2] != 3:
        raise ValueError(f"sh_coeffs must be (N, K, 3), got {tuple(sh_coeffs.shape)}")
    n = sh_coeffs.shape[0]
    check_float32("sh_coeffs", sh_coeffs, (n, sh_coeffs.shape[1], 3), sh_coeffs.device)
    check_float32("dirs", dirs, (n, 3), sh_coeffs.device)
    stored_degree = buffer_sh_degree(sh_coeffs.shape[1])
    if active_degree is None:
        active_degree = stored_degree
    if not 0 <= active_degree <= stored_degree:
        raise ValueError(
            f"active_degree={active_degree} needs {num_sh_coeffs(active_degree)} "
            f"coefficients per channel, but sh_coeffs holds {sh_coeffs.shape[1]} "
            f"(degree {stored_degree})"
        )
    return EvalSH.apply(sh_coeffs, dirs, active_degree)
