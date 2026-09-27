"""The .ply reading/writing and colour-column handling `metalsplat.export`
(3DGS) and `metalsplat.export2dgs` (2DGS) share. The two formats differ
only in how many `scale_*` properties they carry, so everything else lives
here once instead of in two copies.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from metalsplat.reference.sh_ref import MAX_SH_DEGREE, SH_C0, num_sh_coeffs
from metalsplat.splat_model import SplatModel

_FLOAT_TYPES = {"float", "float32"}


def color_columns(model: SplatModel) -> tuple[np.ndarray, np.ndarray]:
    """`(f_dc, f_rest)` float32 columns in the reference layout.

    f_rest is channel-major (all of channel 0's non-DC coefficients, then
    1, then 2). A flat-RGB model is written as its colour in the DC term
    (the standard RGB2SH formula) with 24 all-zero f_rest entries, i.e. the
    degree-2 layout carrying no view dependence.
    """
    n = model.num_points
    if model.sh_degree == 0:
        dc = ((model.colors.detach() - 0.5) / SH_C0).cpu().numpy().astype(np.float32)
        return dc, np.zeros((n, 24), dtype=np.float32)
    sh = model.raw_sh.detach().cpu()  # (N, K, 3)
    dc = sh[:, 0, :].numpy().astype(np.float32)
    # Explicit width, not -1: reshape cannot infer it for an empty model.
    n_rest = (sh.shape[1] - 1) * 3
    rest = sh[:, 1:, :].transpose(1, 2).contiguous().reshape(n, n_rest).numpy()
    return dc, rest.astype(np.float32)


def write_ply(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    """Writes `columns` (name -> (N,) or (N, k) array, in property order;
    a (N, k) array named `x` becomes properties `x_0 .. x_{k-1}`) as one
    binary little-endian float32 vertex element."""
    names, arrays = [], []
    for name, array in columns.items():
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            names.append(name)
            arrays.append(array[:, None])
        else:
            names.extend(f"{name}_{i}" for i in range(array.shape[1]))
            arrays.append(array)
    data = np.concatenate(arrays, axis=1)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {data.shape[0]}",
            *[f"property float {name}" for name in names],
            "end_header",
        ]
    )
    with open(path, "wb") as f:
        f.write((header + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """The vertex element of a binary little-endian .ply, as name -> (N,).

    Only float32 vertex properties are supported -- what 3DGS/2DGS files
    use. Anything else would change the per-vertex stride, so it is an
    error rather than a silent misread; so is a vertex element that comes
    after another element. Trailing elements are ignored.
    """
    path = Path(path)
    content = path.read_bytes()
    marker = b"end_header\n"
    header_end = content.index(marker) + len(marker)
    lines = content[:header_end].decode("ascii").splitlines()
    if len(lines) < 2 or lines[1].strip() != "format binary_little_endian 1.0":
        raise ValueError(f"{path.name}: only binary_little_endian .ply is supported")

    n, names, element = None, [], None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            if element == "vertex":
                if names or n is not None:
                    raise ValueError(f"{path.name}: more than one vertex element")
                n = int(parts[2])
            elif n is None:
                raise ValueError(
                    f"{path.name}: element '{element}' before 'vertex' is not supported"
                )
        elif parts[0] == "property" and element == "vertex":
            if parts[1] not in _FLOAT_TYPES:
                raise ValueError(
                    f"{path.name}: vertex property '{parts[-1]}' is '{parts[1]}'; "
                    "only float32 properties are supported"
                )
            names.append(parts[-1])
    if n is None:
        raise ValueError(f"{path.name}: no vertex element")

    count = n * len(names)
    data = np.frombuffer(content, dtype="<f4", count=count, offset=header_end)
    data = data.reshape(n, len(names))
    return {name: data[:, i] for i, name in enumerate(names)}


def model_color_kwargs(col: dict[str, np.ndarray], path: Path) -> dict:
    """Constructor kwargs for the colour columns: `colors` for a file with
    no view dependence (all f_rest zero, or none), else `sh_degree` and
    `sh_coeffs` at the largest whole degree the f_rest count holds.

    Reads the SH degree from the property count rather than assuming one,
    so files of any degree up to 3 (45 f_rest entries, what the reference
    and most tools write) load at their own degree; coefficients beyond
    degree 3 are dropped with a warning.
    """
    n = len(col["x"])
    dc = torch.from_numpy(
        np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], axis=1).copy()
    )
    n_rest = sum(1 for name in col if name.startswith("f_rest_"))
    coeffs_per_channel = n_rest // 3  # channel-major: [ch0 coeffs..., ch1..., ch2...]
    max_coeffs = num_sh_coeffs(MAX_SH_DEGREE)
    if coeffs_per_channel > max_coeffs - 1:
        warnings.warn(
            f"{path.name} has {coeffs_per_channel + 1} SH coefficients/channel; this package "
            f"supports {max_coeffs}. Dropping the higher-degree ones.",
            stacklevel=3,
        )

    if n_rest == 0 or not np.any(np.stack([col[f"f_rest_{i}"] for i in range(n_rest)])):
        # degree-0-only file: recover plain RGB via the inverse of RGB2SH
        return {"colors": (dc * SH_C0 + 0.5).clamp(0, 1)}

    # Round the stored count *down* to a whole degree: a file carrying a
    # partial band cannot be evaluated as that degree, so keep the complete
    # bands and drop the remainder.
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
    return {"sh_degree": degree, "sh_coeffs": sh}


def stack_columns(col: dict[str, np.ndarray], *names: str) -> torch.Tensor:
    """(N, len(names)) float32 tensor of the named columns."""
    return torch.from_numpy(np.stack([col[name] for name in names], axis=1).copy())
