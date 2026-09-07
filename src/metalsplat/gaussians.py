"""Learnable 3D gaussian splat parameters.

Raw parameters are unconstrained (so any optimizer step keeps them valid)
and mapped to their physical values through fixed activations:
scale = exp(raw_scale) (positive), quat = normalize(raw_quat) (unit),
opacity = sigmoid(raw_opacity) (in [0, 1]).

Color is either plain per-gaussian RGB (`sh_degree=0`, the default: color =
sigmoid(raw_color), in [0, 1]) or degree<=2 spherical harmonics
(`sh_degree=2`: view-dependent color = eval_sh(raw_sh, view_dir) + 0.5,
matching standard 3DGS convention -- unconstrained/unclamped internally,
consumers clamp to [0, 1] for display). SH needs a view direction so it
can't be a plain property; use `colors_from_view(view_dirs)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metalsplat.ops.sh import eval_sh
from metalsplat.reference.sh_ref import NUM_SH_COEFFS, SH_C0


class GaussianModel(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor | None = None,  # (N, 3), positive; defaults to 0.02
        quats: torch.Tensor | None = None,  # (N, 4), unit; defaults to identity
        opacities: torch.Tensor | None = None,  # (N,), in [0, 1]; defaults to 0.5
        colors: torch.Tensor | None = None,  # (N, 3), in [0, 1]; defaults to 0.5 gray
        sh_degree: int = 0,  # 0 = plain RGB, 2 = view-dependent spherical harmonics
        sh_coeffs: torch.Tensor | None = None,  # (N, 9, 3) raw SH; overrides `colors`-derived DC init if given
    ):
        super().__init__()
        n = means.shape[0]
        device = means.device

        if sh_degree not in (0, 2):
            raise ValueError(f"sh_degree must be 0 or 2, got {sh_degree}")
        self.sh_degree = sh_degree

        if scales is None:
            scales = torch.full((n, 3), 0.02, device=device)
        if quats is None:
            quats = torch.zeros(n, 4, device=device)
            quats[:, 0] = 1.0
        if opacities is None:
            opacities = torch.full((n,), 0.5, device=device)
        if colors is None:
            colors = torch.full((n, 3), 0.5, device=device)

        self.means = nn.Parameter(means.clone())
        self.raw_scales = nn.Parameter(scales.clone().log())
        self.raw_quats = nn.Parameter(quats.clone())
        self.raw_opacities = nn.Parameter(_logit(opacities.clone()))

        if sh_degree == 0:
            self.raw_colors = nn.Parameter(_logit(colors.clone()))
        elif sh_coeffs is not None:
            self.raw_sh = nn.Parameter(sh_coeffs.clone())
        else:
            raw_sh = torch.zeros(n, NUM_SH_COEFFS, 3, device=device)
            raw_sh[:, 0, :] = (colors.clone() - 0.5) / SH_C0
            self.raw_sh = nn.Parameter(raw_sh)

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return self.raw_quats / self.raw_quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacities)

    @property
    def colors(self) -> torch.Tensor:
        if self.sh_degree != 0:
            raise AttributeError(
                "This model uses spherical harmonics (sh_degree=2); color depends on "
                "viewing direction, so use colors_from_view(view_dirs) instead."
            )
        return torch.sigmoid(self.raw_colors)

    def colors_from_view(self, view_dirs: torch.Tensor) -> torch.Tensor:
        """view_dirs: (N, 3) unit vectors from each gaussian to the camera."""
        if self.sh_degree == 0:
            raise AttributeError("This model has sh_degree=0; use the `colors` property instead.")
        return eval_sh(self.raw_sh, view_dirs) + 0.5

    @property
    def num_points(self) -> int:
        return self.means.shape[0]

    @classmethod
    def random(cls, n: int, bound: float = 1.0, device: str | torch.device = "cpu") -> GaussianModel:
        means = (torch.rand(n, 3, device=device) * 2 - 1) * bound
        colors = torch.rand(n, 3, device=device)
        return cls(means, colors=colors)


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))
