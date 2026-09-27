"""Learnable 3D gaussian splat parameters.

Raw parameters are unconstrained (so any optimizer step keeps them valid)
and mapped to their physical values through fixed activations:
scale = exp(raw_scale) (positive), quat = normalize(raw_quat) (unit),
opacity = sigmoid(raw_opacity) (in [0, 1]).

Color is either plain per-gaussian RGB (`sh_degree=0`, the default: color =
sigmoid(raw_color), in [0, 1]) or degree<=3 spherical harmonics
(`sh_degree=1..3`: view-dependent color = max(eval_sh(raw_sh, view_dir) +
0.5, 0), matching standard 3DGS convention -- clamped below at 0 but not
above, consumers clamp to [0, 1] for display). SH needs a view direction so it
can't be a plain property; use `colors_from_view(view_dirs)`.
"""

from __future__ import annotations

from metalsplat.sh_color import logit
from metalsplat.splat_model import SplatModel

__all__ = ["GaussianModel", "logit"]


class GaussianModel(SplatModel):
    """3D gaussian splats: `scales` is (N, 3), one per ellipsoid axis. See
    `metalsplat.splat_model` for the shared parameterization."""

    NUM_SCALE_AXES = 3
