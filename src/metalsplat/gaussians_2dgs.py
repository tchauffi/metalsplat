"""Learnable 2D gaussian splat ("surfel") parameters.

A 2D gaussian splat (Huang et al. 2024, "2D Gaussian Splatting for
Geometrically Accurate Radiance Fields") is a flat, oriented disk in world
space rather than a 3D ellipsoid: a position, an orientation (quaternion),
and only *two* tangent-plane scales `(s_u, s_v)` -- there is no third,
depth-axis scale. `quat_to_rotmat(quat)`'s columns 0/1 are the disk's
tangent axes `t_u, t_v` (world-space, scaled by `s_u, s_v` at use sites);
column 2 is the disk's surface normal.

Everything but the scale count is shared with `GaussianModel` through
`metalsplat.splat_model.SplatModel` -- see that module's docstring.
"""

from __future__ import annotations

import torch

from metalsplat.splat_model import SplatModel
from metalsplat.utils.quaternion import quat_to_rotmat


class Gaussian2DModel(SplatModel):
    """2D gaussian splats: `scales` is (N, 2), the tangent-plane extents
    `(s_u, s_v)`, with no depth-axis scale."""

    NUM_SCALE_AXES = 2

    @property
    def rotmat(self) -> torch.Tensor:
        """(N, 3, 3); columns 0/1 are the tangent axes, column 2 the normal."""
        return quat_to_rotmat(self.quats)

    @property
    def normals(self) -> torch.Tensor:
        """(N, 3) unit world-space surface normal (unoriented -- not yet
        flipped to face any particular camera; see rendering.render_2dgs)."""
        return self.rotmat[..., :, 2]
