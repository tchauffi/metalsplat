"""Exports a GaussianModel to the standard 3D Gaussian Splatting .ply
format (binary little-endian) -- the de facto interchange format most
existing 3DGS viewers and tools (SuperSplat, the antimatter15/playcanvas
web viewers, etc.) read directly, letting a trained scene be viewed and
reused without this package at all.

Property layout matches the original Kerbl et al. reference
implementation: x,y,z, nx,ny,nz (unused, zero), f_dc_0-2 (degree-0 SH,
raw/pre-activation), f_rest_* (higher-degree SH, raw, channel-major),
opacity (raw/pre-sigmoid), scale_0-2 (raw/pre-exp, i.e. log-scale),
rot_0-3 (unit quaternion, w,x,y,z).

f_rest holds (K-1)*3 entries for a model with K coefficients per channel,
so a degree-3 model writes the reference implementation's full 45 and a
degree-2 model writes 24. Viewers that read the SH degree from the
header's property count (most modern ones do) render either correctly;
older viewers that hardcode degree 3 may not render a lower-degree file. A
flat-RGB model (sh_degree=0) is exported with its colour as the DC term
(via the standard RGB2SH formula) and 24 all-zero f_rest entries, i.e. the
degree-2 layout carrying no view dependence; load_ply reads such a file back
as a flat-RGB model.
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
from metalsplat.filter3d import apply_3d_filter
from metalsplat.gaussians import GaussianModel


def save_ply(
    model: GaussianModel, path: str | Path, filter_3d: torch.Tensor | None = None
) -> None:
    """Writes `model` to `path` in the reference 3DGS .ply layout.

    Pass the `filter_3d` the model was trained and evaluated with (see
    metalsplat.filter3d) to bake it into the written scales and opacities,
    the way Mip-Splatting exports. Other viewers know nothing about the
    filter, so without this they render every gaussian thinner and the
    small ones denser than training ever saw them. A baked file must then
    be rendered *without* `filter_3d`, or the filter is applied twice.

    The screen-space anti-aliasing compensation (`render(antialias=True)`)
    depends on the view and cannot be baked.
    """
    xyz = model.means.detach().cpu().numpy()
    dc, rest = color_columns(model)

    raw_opacities = model.raw_opacities.detach()
    raw_scales = model.raw_scales.detach()
    if filter_3d is not None:
        scales, opacities = apply_3d_filter(
            model.scales.detach(), model.opacities.detach(), filter_3d.detach()
        )
        raw_scales = scales.log()
        raw_opacities = torch.logit(opacities, eps=1e-6)

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
            "opacity": raw_opacities.cpu().numpy(),
            "scale": raw_scales.cpu().numpy(),  # (N, 3), log-space
            "rot": model.quats.detach().cpu().numpy(),  # (N, 4), unit, w x y z
        },
    )


def load_ply(path: str | Path, device: str = "cpu") -> GaussianModel:
    """Inverse of save_ply: reads a 3DGS .ply back into a GaussianModel.

    See `metalsplat._ply.model_color_kwargs` for how the SH degree is
    detected; a file whose f_rest entries are all zero loads as a flat-RGB
    model.
    """
    path = Path(path)
    col = read_ply(path)
    model = GaussianModel(
        stack_columns(col, "x", "y", "z"),
        # stored as log-scale; GaussianModel takes activated values
        scales=stack_columns(col, "scale_0", "scale_1", "scale_2").exp(),
        quats=stack_columns(col, "rot_0", "rot_1", "rot_2", "rot_3"),
        opacities=torch.sigmoid(torch.from_numpy(col["opacity"].copy())),
        **model_color_kwargs(col, path),
    )
    return model.to(device)
