"""Shared spherical-harmonics / plain-RGB color parameterization.

Used by both `GaussianModel` (3D) and `Gaussian2DModel` (2D): color is
either plain per-gaussian RGB (`sh_degree=0`: color = sigmoid(raw_color),
in [0, 1]) or degree<=2 spherical harmonics (`sh_degree=1..3`:
view-dependent color = eval_sh(raw_sh, view_dir) + 0.5, matching standard
3DGS convention -- unconstrained/unclamped internally, consumers clamp to
[0, 1] for display).

Extracted out of `GaussianModel` so the two model classes share one
implementation instead of two copies that can drift; both classes register
the returned tensor as their own `nn.Parameter` and hold their own
`sh_degree`/`active_sh_degree` state, so this module stays plain functions,
not a base class.
"""

from __future__ import annotations

import torch

from metalsplat.ops.sh import eval_sh
from metalsplat.reference.sh_ref import MAX_SH_DEGREE, SH_C0, num_sh_coeffs


def logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def validate_sh_degree(sh_degree: int) -> None:
    if not 0 <= sh_degree <= MAX_SH_DEGREE:
        raise ValueError(
            f"sh_degree must be between 0 and {MAX_SH_DEGREE}, got {sh_degree}"
        )


def init_color_param(
    n: int,
    sh_degree: int,
    colors: torch.Tensor | None,  # (N, 3), in [0, 1]; defaults to 0.5 gray
    sh_coeffs: torch.Tensor | None,  # (N, (deg+1)^2, 3) raw SH, overrides colors
    device: torch.device,
) -> tuple[str, torch.Tensor]:
    """Returns `(param_name, initial_value)`: the caller registers
    `init_value` as an `nn.Parameter` under `param_name` (`"raw_colors"` for
    `sh_degree=0`, else `"raw_sh"`).
    """
    if colors is None:
        colors = torch.full((n, 3), 0.5, device=device)

    if sh_degree == 0:
        return "raw_colors", logit(colors.clone())

    if sh_coeffs is not None:
        expected = num_sh_coeffs(sh_degree)
        if sh_coeffs.shape[1] != expected:
            raise ValueError(
                f"sh_degree={sh_degree} needs {expected} coefficients per channel, "
                f"got sh_coeffs with shape {tuple(sh_coeffs.shape)}"
            )
        return "raw_sh", sh_coeffs.clone()

    raw_sh = torch.zeros(n, num_sh_coeffs(sh_degree), 3, device=device)
    raw_sh[:, 0, :] = (colors.clone() - 0.5) / SH_C0
    return "raw_sh", raw_sh


def colors_from_view(
    raw_sh: torch.Tensor, active_sh_degree: int, view_dirs: torch.Tensor
) -> torch.Tensor:
    """view_dirs: (N, 3) unit vectors from each gaussian to the camera."""
    return eval_sh(raw_sh, view_dirs, active_sh_degree) + 0.5


def increase_sh_degree(sh_degree: int, active_sh_degree: int) -> int:
    """Returns the next active degree, up to `sh_degree`."""
    return min(active_sh_degree + 1, sh_degree)
