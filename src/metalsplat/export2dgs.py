"""Exports a Gaussian2DModel to the 2D Gaussian Splatting .ply format
(binary little-endian) -- the same layout the official reference
implementation (hbb1/2d-gaussian-splatting) writes, so a trained scene can
be viewed/meshed with that toolchain (or its viewers) without this package.

Property layout is identical to `metalsplat.export`'s 3DGS format (x,y,z,
nx,ny,nz, f_dc_0-2, f_rest_*, opacity, scale_*, rot_0-3) with exactly one
difference: `scale_*` has 2 entries, not 3, since a 2D splat has no third,
depth-axis scale (confirmed against the official implementation's own
`GaussianModel._scaling`, which is genuinely `(N, 2)`, and its
`construct_list_of_attributes`, which loops `range(self._scaling.shape[1])`
-- i.e. 2 entries -- for exactly this reason).

`nx,ny,nz` are written as zero, matching *both* 3DGS's and the official
2DGS implementation's own `save_ply` (`normals = np.zeros_like(xyz)`) --
despite 2DGS splats having a real, meaningful normal (`quat_to_rotmat`'s
third column), the reference implementation doesn't store it directly in
these fields either; downstream tools (e.g. mesh extraction) recompute it
from the loaded rotation quaternion instead. This module does the same,
rather than "fixing" the vestigial zero fields to be more accurate than
the format they're meant to interchange with.

The 2-scale layout is faithful to the reference toolchain but not to the
much larger ecosystem of generic 3DGS viewers/tools (SuperSplat, the
various three.js/WebGL viewers, etc.), which hardcode the standard 3DGS
property set -- including exactly 3 `scale_*` entries -- and misread the
vertex stride when one is missing (everything from `opacity` onward
shifts by a field, so a splat can render as if its opacity, size, and
rotation are all wrong at once, when the actual per-field values are
fine). `save_ply(..., viewer_compatible=True)` trades reference-format
purity for that broader interop: it adds a synthetic `scale_2`, sized a
fixed log-ratio below each splat's thinner in-plane axis so it always
reads as "flat" relative to the splat's own size rather than in absolute
units (which would be wrong for scenes at a different scale). `load_ply`
already ignores any `scale_2` it finds -- it only ever looks up
`scale_0`/`scale_1` by name -- so both layouts round-trip back into a
`Gaussian2DModel` without extra handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from metalsplat._ply import (
    color_columns,
    model_color_kwargs,
    read_ply,
    stack_columns,
    write_ply,
)
from metalsplat.gaussians_2dgs import Gaussian2DModel

# log-space gap below a splat's thinner in-plane axis used for the synthetic
# scale_2 in viewer_compatible mode -- exp(-4) ~ 1.8% of that axis, thin
# enough to read as flat without being degenerate (zero) for viewers that
# build a full 3x3 covariance and need it invertible.
_VIEWER_COMPAT_LOG_MARGIN = 4.0


def save_ply(
    model: Gaussian2DModel, path: str | Path, viewer_compatible: bool = False
) -> None:
    xyz = model.means.detach().cpu().numpy()
    dc, rest = color_columns(model)
    scale = model.raw_scales.detach().cpu().numpy()  # (N, 2), log-space
    if viewer_compatible:
        thin_axis = np.minimum(scale[:, 0], scale[:, 1]) - _VIEWER_COMPAT_LOG_MARGIN
        scale = np.concatenate([scale, thin_axis[:, None]], axis=1)  # (N, 3)

    write_ply(
        path,
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "nx": np.zeros(len(xyz)),
            "ny": np.zeros(len(xyz)),
            "nz": np.zeros(len(xyz)),
            "f_dc": dc,
            "f_rest": rest,
            "opacity": model.raw_opacities.detach().cpu().numpy(),
            "scale": scale,
            "rot": model.quats.detach().cpu().numpy(),  # (N, 4), unit, w x y z
        },
    )


def load_ply(path: str | Path, device: str = "cpu") -> Gaussian2DModel:
    """Inverse of save_ply: reads a 2DGS .ply back into a Gaussian2DModel.

    Same as `metalsplat.export.load_ply`, but builds the model from the two
    `scale_0`/`scale_1` columns; a viewer-compatible file's synthetic
    `scale_2` is ignored.
    """
    path = Path(path)
    col = read_ply(path)
    model = Gaussian2DModel(
        stack_columns(col, "x", "y", "z"),
        # stored as log-scale; Gaussian2DModel takes activated values
        scales=stack_columns(col, "scale_0", "scale_1").exp(),
        quats=stack_columns(col, "rot_0", "rot_1", "rot_2", "rot_3"),
        opacities=torch.sigmoid(torch.from_numpy(col["opacity"].copy())),
        **model_color_kwargs(col, path),
    )
    return model.to(device)
