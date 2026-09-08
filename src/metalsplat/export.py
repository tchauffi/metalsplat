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
flat-RGB model (sh_degree=0) is exported as degree-0-only SH (f_rest all
zero), via the standard RGB2SH formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from metalsplat.gaussians import GaussianModel
from metalsplat.reference.sh_ref import MAX_SH_DEGREE, SH_C0, num_sh_coeffs


def save_ply(model: GaussianModel, path: str | Path) -> None:
    path = Path(path)
    n = model.num_points

    xyz = model.means.detach().cpu().numpy().astype(np.float32)
    normals = np.zeros_like(xyz)

    if model.sh_degree == 0:
        dc = ((model.colors.detach() - 0.5) / SH_C0).cpu().numpy().astype(np.float32)  # (N, 3)
        rest = np.zeros((n, 24), dtype=np.float32)
    else:
        sh = model.raw_sh.detach().cpu()  # (N, K, 3)
        dc = sh[:, 0, :].numpy().astype(np.float32)  # (N, 3)
        # channel-major: all of channel 0's non-DC coefficients, then 1, then 2
        rest = sh[:, 1:, :].transpose(1, 2).contiguous().reshape(n, -1).numpy().astype(np.float32)

    opacity = model.raw_opacities.detach().cpu().numpy().astype(np.float32).reshape(n, 1)
    scale = model.raw_scales.detach().cpu().numpy().astype(np.float32)  # (N, 3), log-space
    rot = model.quats.detach().cpu().numpy().astype(np.float32)  # (N, 4), unit, w x y z

    data = np.concatenate([xyz, normals, dc, rest, opacity, scale, rot], axis=1).astype(np.float32)

    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
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


def load_ply(path: str | Path, device: str = "cpu") -> GaussianModel:
    """Inverse of save_ply: reads a 3DGS .ply back into a GaussianModel.

    Reads the SH degree from the header's f_rest_* property count rather
    than assuming one, so files written by other tools (degree 3 / 45
    f_rest entries) are detected -- though only degree<=2 can currently be
    represented, so higher-degree coefficients are dropped with a warning.
    """
    import warnings

    path = Path(path)
    with open(path, "rb") as f:
        content = f.read()

    marker = b"end_header\n"
    header_end = content.index(marker) + len(marker)
    lines = content[:header_end].decode("ascii").splitlines()
    if lines[1] != "format binary_little_endian 1.0":
        raise ValueError(f"Only binary_little_endian .ply is supported, got: {lines[1]}")

    n = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
    names = [line.split()[-1] for line in lines if line.startswith("property float")]
    data = np.frombuffer(content[header_end:], dtype="<f4").reshape(n, len(names))
    col = {name: data[:, i] for i, name in enumerate(names)}

    means = torch.from_numpy(np.stack([col["x"], col["y"], col["z"]], axis=1).copy())
    scales = torch.from_numpy(
        np.stack([col["scale_0"], col["scale_1"], col["scale_2"]], axis=1).copy()
    ).exp()  # stored as log-scale; GaussianModel takes activated values
    quats = torch.from_numpy(
        np.stack([col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]], axis=1).copy()
    )
    opacities = torch.sigmoid(torch.from_numpy(col["opacity"].copy()))  # stored pre-sigmoid
    dc = torch.from_numpy(np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], axis=1).copy())

    n_rest = sum(1 for name in names if name.startswith("f_rest_"))
    coeffs_per_channel = n_rest // 3  # channel-major: [ch0 coeffs..., ch1..., ch2...]
    max_coeffs = num_sh_coeffs(MAX_SH_DEGREE)
    if coeffs_per_channel > max_coeffs - 1:
        warnings.warn(
            f"{path.name} has {coeffs_per_channel + 1} SH coefficients/channel; this package "
            f"supports {max_coeffs}. Dropping the higher-degree ones.",
            stacklevel=2,
        )

    if n_rest == 0 or not np.any(np.stack([col[f"f_rest_{i}"] for i in range(n_rest)])):
        # degree-0-only file: recover plain RGB via the inverse of RGB2SH
        colors = (dc * SH_C0 + 0.5).clamp(0, 1)
        model = GaussianModel(means, scales=scales, quats=quats, opacities=opacities, colors=colors)
    else:
        # Round the stored count *down* to a whole degree: a file carrying a
        # partial band cannot be evaluated as that degree, so keep the
        # complete bands and drop the remainder.
        keep = min(coeffs_per_channel + 1, max_coeffs)
        degree = 0
        while num_sh_coeffs(degree + 1) <= keep:
            degree += 1
        n_coeffs = num_sh_coeffs(degree)

        sh = torch.zeros(n, n_coeffs, 3)
        sh[:, 0, :] = dc
        for ch in range(3):
            for k in range(n_coeffs - 1):
                sh[:, k + 1, ch] = torch.from_numpy(col[f"f_rest_{ch * coeffs_per_channel + k}"].copy())
        model = GaussianModel(
            means, scales=scales, quats=quats, opacities=opacities,
            sh_degree=degree, sh_coeffs=sh,
        )

    return model.to(device)
