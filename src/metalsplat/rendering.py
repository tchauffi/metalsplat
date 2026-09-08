"""High-level glue: GaussianModel + Camera -> project -> rasterize -> image."""

from __future__ import annotations

from typing import NamedTuple

import torch

from metalsplat.camera import Camera
from metalsplat.filter3d import apply_3d_filter
from metalsplat.gaussians import GaussianModel
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.ops.tiling import DEFAULT_TILE_SIZE


class RenderAux(NamedTuple):
    """Auxiliary render outputs (see `render(..., return_aux=True)`)."""

    image: torch.Tensor  # (H, W, 3)
    means2d: torch.Tensor  # (N, 2), with retain_grad() called
    valid: torch.Tensor  # (N,)
    final_T: torch.Tensor  # (H, W) per-pixel transmittance
    depth: torch.Tensor  # (H, W) alpha-weighted expected depth, forward-only


def render(
    model: GaussianModel,
    camera: Camera,
    near: float = 0.2,
    eps2d: float = 0.3,
    tile_size: int = DEFAULT_TILE_SIZE,
    background: torch.Tensor | None = None,
    return_aux: bool = False,
    abs_grad_accum: torch.Tensor | None = None,
    near_fade: tuple[float, float] | None = None,
    antialias: bool = True,
    filter_3d: torch.Tensor | None = None,
):
    """Renders `model` from `camera`'s viewpoint. Returns an (H, W, 3) image.

    If `return_aux` is True, instead returns a `RenderAux` (a NamedTuple, so
    both `aux.depth` and positional unpacking work), with
    `means2d.retain_grad()` already called. `valid` is used by training
    loops to count how many steps each gaussian was visible for; `final_T`
    (per-pixel transmittance, close to 1 where ~no gaussian contributed) is
    used by metalsplat.seed to find uncovered regions; `depth` is the
    alpha-weighted expected depth (forward-only, no gradient).

    `abs_grad_accum`, if given, is passed through to rasterize_gaussians:
    an (N,) tensor that backward() atomically adds each gaussian's
    |screen-space gradient contribution| into (AbsGS-style densification
    signal, metalsplat.densify) -- prefer this over means2d.grad.norm()
    for that purpose, since contributions from different pixels can have
    opposite signs and cancel out in means2d.grad's signed sum.

    `near_fade`, if given, is `(r0, r1)` in world units: a sphere around the
    camera inside which gaussians are suppressed. Opacity is scaled by a
    smoothstep that is 0 closer than `r0` and 1 beyond `r1`, so gaussians
    sitting in the empty space just in front of the camera stop contributing.
    Reconstruction puts spurious gaussians there -- they are only ever seen
    by a couple of training views, so nothing constrains them -- and because
    they are near the camera they sweep across the whole frame during a
    camera move, which is what makes an orbit flicker.

    The fade band is the point: a hard cut makes each gaussian vanish in a
    single frame as the camera crosses it, which trades flicker for popping.
    `r1` must be comfortably below the distance to the nearest real geometry
    (for an orbit, the camera's height above the ground) or this punches a
    hole in the scene instead.

    `antialias` (default True) scales each gaussian's opacity by the
    Mip-Splatting compensation factor, correcting for the energy the eps2d
    low-pass filter would otherwise add to sub-pixel gaussians. Set False to
    reproduce renders from before this existed, or to match a model trained
    without it.

    `filter_3d`, if given, is the (N,) per-gaussian filter radius from
    metalsplat.filter3d.compute_3d_filter: the *world-space* half of
    Mip-Splatting, band-limiting each gaussian to the finest detail any
    training camera resolved it at. Recompute it when the gaussian count
    changes, and keep it consistent between training and rendering -- a
    model trained with it expects to be rendered with it.
    """
    scales = model.scales
    opacities = model.opacities
    if filter_3d is not None:
        scales, opacities = apply_3d_filter(scales, opacities, filter_3d)

    means2d, depths, conics, radii, valid, compensation = project_gaussians(
        model.means,
        scales,
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
    if return_aux and means2d.requires_grad:
        # No-op under torch.no_grad() / inference, where aux is still useful
        # for depth and coverage but there's no graph to retain a grad on.
        means2d.retain_grad()

    if antialias:
        # Mip-Splatting's anti-aliasing factor. project_gaussians dilates the
        # 2D covariance by eps2d as a low-pass filter; that also inflates the
        # integral of the gaussian's density, so a sub-pixel gaussian renders
        # stronger than it should and flickers as it crosses pixel
        # boundaries. `compensation` is the factor that takes that extra
        # energy back out of the peak opacity. ~1 for anything comfortably
        # larger than a pixel, falling to 0 for the smallest gaussians.
        opacities = opacities * compensation
    if near_fade is not None:
        r0, r1 = near_fade
        dist = (model.means - camera.position).norm(dim=-1)
        w = ((dist - r0) / max(r1 - r0, 1e-6)).clamp(0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep: flat at both ends, no crease
        opacities = opacities * w
        # Fully-faded gaussians contribute nothing, so drop them from tile
        # binning too rather than paying to composite a zero.
        valid = valid * (w > 0.0).to(valid.dtype)

    if model.sh_degree == 0:
        colors = model.colors
    else:
        view_dirs = torch.nn.functional.normalize(model.means - camera.position, dim=-1)
        colors = model.colors_from_view(view_dirs)

    result = rasterize_gaussians(
        means2d,
        depths,
        conics,
        opacities,
        colors,
        radii,
        valid,
        camera.img_width,
        camera.img_height,
        tile_size=tile_size,
        background=background,
        return_aux=return_aux,
        abs_grad_accum=abs_grad_accum,
    )
    if return_aux:
        image, depth, final_T = result
        return RenderAux(image=image, means2d=means2d, valid=valid, final_T=final_T, depth=depth)
    return result
