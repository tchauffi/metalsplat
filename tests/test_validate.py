"""Input checks at the Metal op boundary (metalsplat.ops._validate).

The kernels index raw buffers with no bounds checks, so each of these used
to be silently wrong memory rather than an error.
"""

import pytest
import torch

from metalsplat import Camera, Gaussian2DModel
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.project_2dgs import project_gaussians_2dgs
from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.ops.rasterize_2dgs import rasterize_gaussians_2dgs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

DEV = "mps"


def _splats_3d(n=4, size=16):
    means2d = torch.full((n, 2), size / 2, device=DEV)
    depths = torch.arange(1, n + 1, dtype=torch.float32, device=DEV)
    conics = torch.tensor([[0.1, 0.0, 0.1]] * n, device=DEV)
    opacities = torch.full((n,), 0.5, device=DEV)
    colors = torch.rand(n, 3, device=DEV)
    radii = torch.full((n,), 6.0, device=DEV)
    valid = torch.ones(n, device=DEV)
    return means2d, depths, conics, opacities, colors, radii, valid


def _splats_2d(n=8, size=32):
    model = Gaussian2DModel.random(n, bound=0.5, device=DEV)
    cam = Camera.identity(
        fx=size, fy=size, cx=size / 2, cy=size / 2, img_width=size, img_height=size
    ).to(DEV)
    cam.t_wc = torch.tensor([0.0, 0.0, 3.0], device=DEV)
    means2d, depths, rects, valid, transform, normal = project_gaussians_2dgs(
        model.means, model.scales, model.quats, cam.R_wc, cam.t_wc,
        cam.fx, cam.fy, cam.cx, cam.cy, size, size,
    )  # fmt: skip
    args = (
        means2d, transform, normal, model.opacities, model.colors, depths,
        rects, valid, size, size,
    )  # fmt: skip
    return args, n


def test_non_float_background_is_converted_not_reinterpreted():
    # An int64 background used to be bound as-is and read back as float
    # bits: torch.ones(3, dtype=int64) rendered black.
    far = list(_splats_3d())
    far[0] = far[0] + 1000.0  # off-screen, so every pixel is pure background
    as_float = rasterize_gaussians(*far, 4, 4, background=torch.ones(3, device=DEV))
    as_int = rasterize_gaussians(
        *far, 4, 4, background=torch.ones(3, device=DEV, dtype=torch.int64)
    )
    assert torch.equal(as_int.cpu(), as_float.cpu())
    assert torch.all(as_float == 1.0)


def test_background_must_have_three_values():
    with pytest.raises(ValueError, match="background"):
        rasterize_gaussians(*_splats_3d(), 16, 16, background=torch.ones(4, device=DEV))


@pytest.mark.parametrize(
    "make_accum",
    [
        lambda: torch.zeros(3, device=DEV),  # stale size, e.g. after a densify
        lambda: torch.zeros(4, device=DEV, dtype=torch.float16),
        lambda: torch.zeros(4, 2, device=DEV)[:, 0],  # non-contiguous view
    ],
    ids=["stale-size", "float16", "non-contiguous"],
)
def test_bad_abs_grad_accum_raises(make_accum):
    with pytest.raises((ValueError, TypeError), match="abs_grad_accum"):
        rasterize_gaussians(*_splats_3d(n=4), 16, 16, abs_grad_accum=make_accum())


def test_2dgs_accumulators_are_checked():
    args, n = _splats_2d()
    with pytest.raises(ValueError, match="abs_grad_accum"):
        rasterize_gaussians_2dgs(*args, abs_grad_accum=torch.zeros(n + 1, device=DEV))
    with pytest.raises(ValueError, match="pixel_count_accum"):
        rasterize_gaussians_2dgs(
            *args, pixel_count_accum=torch.zeros(n - 1, device=DEV)
        )


def test_mismatched_per_gaussian_shapes_raise():
    splats = list(_splats_3d(n=4))
    splats[4] = splats[4][:3]  # colors for 3 gaussians, 4 everywhere else
    with pytest.raises(ValueError, match="colors"):
        rasterize_gaussians(*splats, 16, 16)


def test_non_float32_projection_input_raises():
    n = 2
    with pytest.raises(TypeError, match="means"):
        project_gaussians(
            torch.zeros(n, 3, device=DEV, dtype=torch.float16),
            torch.ones(n, 3, device=DEV),
            torch.tensor([[1.0, 0, 0, 0]] * n, device=DEV),
            torch.eye(3, device=DEV),
            torch.zeros(3, device=DEV),
            10.0, 10.0, 8.0, 8.0, 16, 16,
        )  # fmt: skip


@pytest.mark.parametrize("pipeline", ["3dgs", "2dgs"])
def test_rendering_on_cpu_says_it_needs_mps(pipeline):
    # Used to surface as torch's generic "Passed CPU tensor to MPS op".
    from metalsplat import GaussianModel, render, render_2dgs

    cam = Camera.identity(fx=16, fy=16, cx=8, cy=8, img_width=16, img_height=16)
    if pipeline == "3dgs":
        call = lambda: render(GaussianModel.random(4), cam)
    else:
        call = lambda: render_2dgs(Gaussian2DModel.random(4), cam)
    with pytest.raises(ValueError, match="'mps' device"):
        call()
