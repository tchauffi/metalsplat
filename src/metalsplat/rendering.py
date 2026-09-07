"""High-level glue: GaussianModel + Camera -> project -> rasterize -> image."""

from __future__ import annotations

import torch

from metalsplat.camera import Camera
from metalsplat.gaussians import GaussianModel
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE


def render(
    model: GaussianModel,
    camera: Camera,
    near: float = 0.2,
    eps2d: float = 0.3,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: torch.Tensor | None = None,
    return_aux: bool = False,
):
    """Renders `model` from `camera`'s viewpoint. Returns an (H, W, 3) image.

    If `return_aux` is True, instead returns `(image, means2d, valid,
    final_T)`, with `means2d.retain_grad()` already called. `means2d` /
    `valid` are used by training loops that need per-gaussian screen-space
    gradient magnitudes for densification (metalsplat.densify), since
    means2d is otherwise just an internal intermediate tensor discarded
    after this call; `final_T` (per-pixel transmittance, close to 1 where
    ~no gaussian contributed) is used by metalsplat.seed to find uncovered
    regions.
    """
    means2d, depths, conics, radii, valid = project_gaussians(
        model.means,
        model.scales,
        model.quats,
        camera.R_wc,
        camera.t_wc,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.img_width,
        camera.img_height,
        near=near,
        eps2d=eps2d,
    )
    if return_aux:
        means2d.retain_grad()

    if model.sh_degree == 0:
        colors = model.colors
    else:
        view_dirs = torch.nn.functional.normalize(model.means - camera.position, dim=-1)
        colors = model.colors_from_view(view_dirs)

    result = rasterize_gaussians(
        means2d,
        depths,
        conics,
        model.opacities,
        colors,
        radii,
        valid,
        camera.img_width,
        camera.img_height,
        tile_size=tile_size,
        background=background,
        return_aux=return_aux,
    )
    if return_aux:
        image, final_T = result
        return image, means2d, valid, final_T
    return result
