import struct

import pytest
import torch

from metalsplat.export import load_ply, save_ply
from metalsplat.gaussians import GaussianModel


def _read_ply(path):
    with open(path, "rb") as f:
        content = f.read()
    header_end = content.index(b"end_header\n") + len(b"end_header\n")
    header = content[:header_end].decode("ascii")
    lines = header.splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format binary_little_endian 1.0"
    n = int(lines[2].split()[-1])
    names = [line.split()[-1] for line in lines[3:-1]]
    body = content[header_end:]
    assert len(body) == n * len(names) * 4
    values = struct.unpack(f"<{n * len(names)}f", body)
    rows = [values[i * len(names) : (i + 1) * len(names)] for i in range(n)]
    return names, rows


def test_save_ply_flat_rgb_roundtrip(tmp_path):
    means = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    scales = torch.full((2, 3), 0.02)
    opacities = torch.tensor([0.7, 0.3])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model = GaussianModel(means, scales=scales, opacities=opacities, colors=colors)

    out_path = tmp_path / "model.ply"
    save_ply(model, out_path)

    names, rows = _read_ply(out_path)
    assert len(rows) == 2
    assert names[:6] == ["x", "y", "z", "nx", "ny", "nz"]
    assert names.count("f_dc_0") == 1 and "f_rest_23" in names
    assert "opacity" in names and names[-4:] == ["rot_0", "rot_1", "rot_2", "rot_3"]

    xi, _yi, zi = names.index("x"), names.index("y"), names.index("z")
    assert rows[0][xi : zi + 1] == (1.0, 2.0, 3.0)
    assert rows[1][xi : zi + 1] == (-1.0, 0.5, 0.0)

    # opacity stored raw (pre-sigmoid) -- sigmoid(raw) should recover the input
    oi = names.index("opacity")
    recovered_opacity = [1.0 / (1.0 + 2.718281828 ** (-row[oi])) for row in rows]
    assert abs(recovered_opacity[0] - 0.7) < 1e-3
    assert abs(recovered_opacity[1] - 0.3) < 1e-3

    # f_rest should be all zero for a flat-RGB (sh_degree=0) model
    rest_idx = [names.index(f"f_rest_{i}") for i in range(24)]
    for row in rows:
        assert all(row[i] == 0.0 for i in rest_idx)

    # rotation should be the identity quaternion (w=1, x=y=z=0)
    ri = [names.index(f"rot_{i}") for i in range(4)]
    for row in rows:
        assert abs(row[ri[0]] - 1.0) < 1e-5
        assert all(abs(row[i]) < 1e-5 for i in ri[1:])


def test_save_ply_sh_model_preserves_non_dc_coefficients(tmp_path):
    means = torch.tensor([[0.0, 0.0, 0.0]])
    model = GaussianModel(means, colors=torch.tensor([[0.5, 0.5, 0.5]]), sh_degree=2)
    with torch.no_grad():
        model.raw_sh[0, 3, 1] = (
            0.42  # a specific non-DC coefficient (index 3, channel 1)
        )

    out_path = tmp_path / "model_sh.ply"
    save_ply(model, out_path)

    names, rows = _read_ply(out_path)
    # our raw_sh[:, 1:, :].transpose(1,2) layout: f_rest index for
    # (coeff=3 among 1..8 -> local index 2, channel=1) is channel*8 + local_index
    local_index = 3 - 1
    channel = 1
    expected_name = f"f_rest_{channel * 8 + local_index}"
    assert abs(rows[0][names.index(expected_name)] - 0.42) < 1e-5


def test_save_load_roundtrip_flat_rgb(tmp_path):
    torch.manual_seed(0)
    n = 7
    means = torch.randn(n, 3)
    scales = torch.rand(n, 3) * 0.1 + 0.01
    opacities = torch.rand(n) * 0.8 + 0.1
    colors = torch.rand(n, 3)
    model = GaussianModel(means, scales=scales, opacities=opacities, colors=colors)

    path = tmp_path / "rt.ply"
    save_ply(model, path)
    loaded = load_ply(path)

    assert loaded.num_points == n
    assert loaded.sh_degree == 0
    assert torch.allclose(loaded.means, model.means, atol=1e-5)
    assert torch.allclose(loaded.scales, model.scales, atol=1e-5)
    assert torch.allclose(loaded.opacities, model.opacities, atol=1e-4)
    assert torch.allclose(loaded.colors, model.colors, atol=1e-3)
    assert torch.allclose(loaded.quats, model.quats, atol=1e-5)


def test_save_load_roundtrip_sh(tmp_path):
    torch.manual_seed(1)
    n = 5
    model = GaussianModel(torch.randn(n, 3), colors=torch.rand(n, 3), sh_degree=2)
    with torch.no_grad():
        model.raw_sh.copy_(torch.randn(n, 9, 3) * 0.3)

    path = tmp_path / "rt_sh.ply"
    save_ply(model, path)
    loaded = load_ply(path)

    assert loaded.sh_degree == 2
    assert torch.allclose(loaded.raw_sh, model.raw_sh, atol=1e-5)
    assert torch.allclose(loaded.means, model.means, atol=1e-5)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_save_load_roundtrip_every_sh_degree(tmp_path, degree):
    from metalsplat.reference.sh_ref import num_sh_coeffs

    torch.manual_seed(degree)
    n = 5
    k = num_sh_coeffs(degree)
    model = GaussianModel(torch.randn(n, 3), colors=torch.rand(n, 3), sh_degree=degree)
    with torch.no_grad():
        model.raw_sh.copy_(torch.randn(n, k, 3) * 0.3)

    path = tmp_path / f"rt_sh{degree}.ply"
    save_ply(model, path)
    loaded = load_ply(path)

    assert loaded.sh_degree == degree
    assert loaded.raw_sh.shape == (n, k, 3)
    assert torch.allclose(loaded.raw_sh, model.raw_sh, atol=1e-5)


def test_degree_3_ply_has_the_reference_45_f_rest_entries(tmp_path):
    # The original 3DGS implementation writes 45 f_rest properties (15 non-DC
    # coefficients x 3 channels). Matching that exactly is what lets viewers
    # that assume degree 3 read our files.
    model = GaussianModel(torch.randn(4, 3), colors=torch.rand(4, 3), sh_degree=3)
    path = tmp_path / "deg3.ply"
    save_ply(model, path)

    header = path.read_bytes().split(b"end_header")[0].decode()
    assert sum(1 for line in header.splitlines() if "f_rest_" in line) == 45
    assert "f_rest_44" in header and "f_rest_45" not in header
