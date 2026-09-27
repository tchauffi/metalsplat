"""The parameterization `GaussianModel` (3DGS) and `Gaussian2DModel` (2DGS)
share: everything but the number of scale axes.

Raw parameters are unconstrained (so any optimizer step keeps them valid)
and mapped to their physical values through fixed activations:
scale = exp(raw_scale) (positive), quat = normalize(raw_quat) (unit),
opacity = sigmoid(raw_opacity) (in [0, 1]). Colour is plain RGB or
spherical harmonics, see `metalsplat.sh_color`.

The two classes used to be two copies of all of this, which is how fixes
drifted between them; each now only declares its scale-axis count and
anything genuinely its own (2DGS's tangent frame and normals).
"""

from __future__ import annotations

import torch
from torch import nn

from metalsplat import sh_color
from metalsplat.sh_color import logit


class SplatModel(nn.Module):
    """Base for gaussian models; subclasses set `NUM_SCALE_AXES`."""

    NUM_SCALE_AXES: int

    def __init__(
        self,
        means: torch.Tensor,  # (N, 3)
        scales: torch.Tensor
        | None = None,  # (N, NUM_SCALE_AXES), positive; defaults to 0.02
        quats: torch.Tensor | None = None,  # (N, 4), unit; defaults to identity
        opacities: torch.Tensor | None = None,  # (N,), in [0, 1]; defaults to 0.5
        colors: torch.Tensor | None = None,  # (N, 3), in [0, 1]; defaults to 0.5 gray
        sh_degree: int = 0,  # 0 = plain RGB; 1..3 = view-dependent spherical harmonics
        sh_coeffs: torch.Tensor
        | None = None,  # (N, (deg+1)^2, 3) raw SH; overrides `colors`-derived DC init if given
        active_sh_degree: int
        | None = None,  # defaults to sh_degree; preserved across rebuilds
    ):
        super().__init__()
        n = means.shape[0]
        device = means.device

        sh_color.validate_sh_degree(sh_degree)
        self.sh_degree = sh_degree
        # Degrees actually evaluated right now. Training can start this
        # at 0 and grow it (see increase_sh_degree): fitting all bands
        # from step 1 lets the higher ones absorb per-photo exposure and
        # white-balance drift before the diffuse base has settled, which
        # is overfitting that shows up as shimmer when the camera moves.
        self.active_sh_degree = sh_color.resolve_active_sh_degree(
            sh_degree, active_sh_degree
        )

        if scales is None:
            scales = torch.full((n, self.NUM_SCALE_AXES), 0.02, device=device)
        if quats is None:
            quats = torch.zeros(n, 4, device=device)
            quats[:, 0] = 1.0
        if opacities is None:
            opacities = torch.full((n,), 0.5, device=device)

        self.means = nn.Parameter(means.clone())
        self.raw_scales = nn.Parameter(scales.clone().log())
        self.raw_quats = nn.Parameter(quats.clone())
        self.raw_opacities = nn.Parameter(logit(opacities.clone()))

        color_param_name, color_param_init = sh_color.init_color_param(
            n, sh_degree, colors, sh_coeffs, device
        )
        setattr(self, color_param_name, nn.Parameter(color_param_init))

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales.exp()

    @property
    def quats(self) -> torch.Tensor:
        return self.raw_quats / self.raw_quats.norm(dim=-1, keepdim=True).clamp_min(
            1e-8
        )

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacities)

    @property
    def colors(self) -> torch.Tensor:
        if self.sh_degree != 0:
            raise AttributeError(
                f"This model uses spherical harmonics (sh_degree={self.sh_degree}); color depends on "
                "viewing direction, so use colors_from_view(view_dirs) instead."
            )
        return torch.sigmoid(self.raw_colors)

    def colors_from_view(self, view_dirs: torch.Tensor) -> torch.Tensor:
        """view_dirs: (N, 3) unit vectors from the camera to each gaussian."""
        if self.sh_degree == 0:
            raise AttributeError(
                "This model has sh_degree=0; use the `colors` property instead."
            )
        return sh_color.colors_from_view(self.raw_sh, self.active_sh_degree, view_dirs)

    def increase_sh_degree(self) -> int:
        """Activates one more SH band, up to this model's sh_degree."""
        self.active_sh_degree = sh_color.increase_sh_degree(
            self.sh_degree, self.active_sh_degree
        )
        return self.active_sh_degree

    @property
    def num_points(self) -> int:
        return self.means.shape[0]

    @classmethod
    def random(cls, n: int, bound: float = 1.0, device: str | torch.device = "cpu"):
        means = (torch.rand(n, 3, device=device) * 2 - 1) * bound
        colors = torch.rand(n, 3, device=device)
        return cls(means, colors=colors)
