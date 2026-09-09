"""Pure-PyTorch reference spherical-harmonics color evaluation (degree <= 3,
16 coefficients per color channel). Spec/oracle for the Metal kernel in
metalsplat.kernels.sh, same role as the other reference/ modules.

Real-valued SH basis constants match the standard normalized convention
used in 3DGS/Plenoxels. Degree 3 is what the original 3D Gaussian Splatting
implementation uses; lower degrees cost fewer coefficients per gaussian
(1, 4, 9, 16 for degree 0..3, times 3 channels) and are still supported --
a model stores exactly as many as its own degree needs.
"""

from __future__ import annotations

import torch

MAX_SH_DEGREE = 3
NUM_SH_COEFFS = (MAX_SH_DEGREE + 1) ** 2  # 16


def num_sh_coeffs(degree: int) -> int:
    """Coefficients per colour channel for a given SH degree."""
    return (degree + 1) ** 2


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(
    sh_coeffs: torch.Tensor, dirs: torch.Tensor, active_degree: int = MAX_SH_DEGREE
) -> torch.Tensor:
    """sh_coeffs: (N, K, 3) with K >= (active_degree+1)^2, dirs: (N, 3) unit
    vectors -> (N, 3) color.

    `active_degree` evaluates only up to that degree, leaving higher
    coefficients out of the graph entirely (so they get no gradient) --
    used for progressive SH growth during training.
    """
    x, y, z = dirs.unbind(-1)
    x, y, z = x[:, None], y[:, None], z[:, None]

    result = SH_C0 * sh_coeffs[:, 0, :]
    if active_degree < 1:
        return result
    result = (
        result
        - SH_C1 * y * sh_coeffs[:, 1, :]
        + SH_C1 * z * sh_coeffs[:, 2, :]
        - SH_C1 * x * sh_coeffs[:, 3, :]
    )
    if active_degree < 2:
        return result

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
    if active_degree < 3:
        return result

    result = (
        result
        + SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[:, 9, :]
        + SH_C3[1] * xy * z * sh_coeffs[:, 10, :]
        + SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[:, 11, :]
        + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12, :]
        + SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[:, 13, :]
        + SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14, :]
        + SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[:, 15, :]
    )
    return result
