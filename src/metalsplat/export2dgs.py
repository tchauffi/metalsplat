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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.reference.sh_ref import MAX_SH_DEGREE, SH_C0, num_sh_coeffs


def save_ply(model: Gaussian2DModel, path: str | Path) -> None:
    path = Path(path)
    n = model.num_points

    xyz = model.means.detach().cpu().numpy().astype(np.float32)
    normals = np.zeros_like(xyz)

    if model.sh_degree == 0:
        dc = (
            ((model.colors.detach() - 0.5) / SH_C0).cpu().numpy().astype(np.float32)
        )  # (N, 3)
        rest = np.zeros((n, 24), dtype=np.float32)
    else:
        sh = model.raw_sh.detach().cpu()  # (N, K, 3)
        dc = sh[:, 0, :].numpy().astype(np.float32)  # (N, 3)
        rest = (
            sh[:, 1:, :]
            .transpose(1, 2)
            .contiguous()
            .reshape(n, -1)
            .numpy()
            .astype(np.float32)
        )

    opacity = (
        model.raw_opacities.detach().cpu().numpy().astype(np.float32).reshape(n, 1)
    )
    scale = (
        model.raw_scales.detach().cpu().numpy().astype(np.float32)
    )  # (N, 2), log-space
    rot = model.quats.detach().cpu().numpy().astype(np.float32)  # (N, 4), unit, w x y z

    data = np.concatenate([xyz, normals, dc, rest, opacity, scale, rot], axis=1).astype(
        np.float32
    )

    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(2)]
        + [f"rot_{i}" for i in range(4)]
    )
    assert data.shape[1] == len(names)

    header = (
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {n}",
                *[f"property float {name}" for name in names],
                "end_header",
            ]
        )
        + "\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def load_ply(path: str | Path, device: str = "cpu") -> Gaussian2DModel:
    """Inverse of save_ply: reads a 2DGS .ply back into a Gaussian2DModel.

    See `metalsplat.export.load_ply` for the SH-degree-detection logic --
    identical here, just building a `Gaussian2DModel` from 2 `scale_*`
    columns instead of 3.
    """
    import warnings

    path = Path(path)
    with open(path, "rb") as f:
        content = f.read()

    marker = b"end_header\n"
    header_end = content.index(marker) + len(marker)
    lines = content[:header_end].decode("ascii").splitlines()
    if lines[1] != "format binary_little_endian 1.0":
        raise ValueError(
            f"Only binary_little_endian .ply is supported, got: {lines[1]}"
        )

    n = int(
        next(line for line in lines if line.startswith("element vertex")).split()[-1]
    )
    names = [line.split()[-1] for line in lines if line.startswith("property float")]
    data = np.frombuffer(content[header_end:], dtype="<f4").reshape(n, len(names))
    col = {name: data[:, i] for i, name in enumerate(names)}

    means = torch.from_numpy(np.stack([col["x"], col["y"], col["z"]], axis=1).copy())
    scales = torch.from_numpy(
        np.stack([col["scale_0"], col["scale_1"]], axis=1).copy()
    ).exp()  # stored as log-scale; Gaussian2DModel takes activated values
    quats = torch.from_numpy(
        np.stack(
            [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]], axis=1
        ).copy()
    )
    opacities = torch.sigmoid(
        torch.from_numpy(col["opacity"].copy())
    )  # stored pre-sigmoid
    dc = torch.from_numpy(
        np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], axis=1).copy()
    )

    n_rest = sum(1 for name in names if name.startswith("f_rest_"))
    coeffs_per_channel = n_rest // 3
    max_coeffs = num_sh_coeffs(MAX_SH_DEGREE)
    if coeffs_per_channel > max_coeffs - 1:
        warnings.warn(
            f"{path.name} has {coeffs_per_channel + 1} SH coefficients/channel; this package "
            f"supports {max_coeffs}. Dropping the higher-degree ones.",
            stacklevel=2,
        )

    if n_rest == 0 or not np.any(np.stack([col[f"f_rest_{i}"] for i in range(n_rest)])):
        colors = (dc * SH_C0 + 0.5).clamp(0, 1)
        model = Gaussian2DModel(
            means, scales=scales, quats=quats, opacities=opacities, colors=colors
        )
    else:
        keep = min(coeffs_per_channel + 1, max_coeffs)
        degree = 0
        while num_sh_coeffs(degree + 1) <= keep:
            degree += 1
        n_coeffs = num_sh_coeffs(degree)

        sh = torch.zeros(n, n_coeffs, 3)
        sh[:, 0, :] = dc
        for ch in range(3):
            for k in range(n_coeffs - 1):
                sh[:, k + 1, ch] = torch.from_numpy(
                    col[f"f_rest_{ch * coeffs_per_channel + k}"].copy()
                )
        model = Gaussian2DModel(
            means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_degree=degree,
            sh_coeffs=sh,
        )

    return model.to(device)
