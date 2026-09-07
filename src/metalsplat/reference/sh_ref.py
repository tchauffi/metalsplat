"""Pure-PyTorch reference spherical-harmonics color evaluation (degree <= 2,
9 coefficients per color channel). Spec/oracle for the Metal kernel in
metalsplat.kernels.sh, same role as the other reference/ modules.

Real-valued SH basis constants match the standard normalized convention
used in 3DGS/Plenoxels.
"""

from __future__ import annotations

import torch

SH_DEGREE = 2
NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2  # 9

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)


def eval_sh(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh_coeffs: (N, 9, 3), dirs: (N, 3) unit vectors -> (N, 3) color."""
    x, y, z = dirs.unbind(-1)
    x, y, z = x[:, None], y[:, None], z[:, None]

    result = SH_C0 * sh_coeffs[:, 0, :]
    result = result - SH_C1 * y * sh_coeffs[:, 1, :] + SH_C1 * z * sh_coeffs[:, 2, :] - SH_C1 * x * sh_coeffs[:, 3, :]

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + SH_C2[0] * xy * sh_coeffs[:, 4, :]
        + SH_C2[1] * yz * sh_coeffs[:, 5, :]
        + SH_C2[2] * (2 * zz - xx - yy) * sh_coeffs[:, 6, :]
        + SH_C2[3] * xz * sh_coeffs[:, 7, :]
        + SH_C2[4] * (xx - yy) * sh_coeffs[:, 8, :]
    )
    return result
