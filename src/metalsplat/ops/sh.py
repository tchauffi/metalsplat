"""torch.autograd.Function wrapping the Metal SH-evaluation kernel
(metalsplat.kernels.sh). Forward/backward math matches
metalsplat.reference.sh_ref.eval_sh exactly -- validated against it in
tests/test_sh.py.
"""

from __future__ import annotations

import torch

from metalsplat.kernels import load as load_kernel
from metalsplat.reference.sh_ref import NUM_SH_COEFFS


class EvalSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sh_coeffs: torch.Tensor, dirs: torch.Tensor):
        n = sh_coeffs.shape[0]
        device = sh_coeffs.device
        sh_c = sh_coeffs.contiguous()
        dirs_c = dirs.contiguous()
        out_color = torch.empty(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("sh")
            lib.sh_forward(sh_c, dirs_c, out_color, threads=n)

        ctx.save_for_backward(sh_c, dirs_c)
        ctx.n = n
        return out_color

    @staticmethod
    def backward(ctx, grad_color):
        sh_coeffs, dirs = ctx.saved_tensors
        n = ctx.n
        device = sh_coeffs.device

        d_sh = torch.zeros(n, NUM_SH_COEFFS, 3, device=device, dtype=torch.float32)
        d_dirs = torch.zeros(n, 3, device=device, dtype=torch.float32)

        if n > 0:
            lib = load_kernel("sh")
            lib.sh_backward(sh_coeffs, dirs, grad_color.contiguous(), d_sh, d_dirs, threads=n)

        return d_sh, d_dirs


def eval_sh(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluates degree<=2 spherical harmonics color using the Metal kernel.

    sh_coeffs: (N, 9, 3), dirs: (N, 3) unit view directions -> (N, 3) color.
    """
    return EvalSH.apply(sh_coeffs, dirs)
