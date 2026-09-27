"""Training-loop plumbing shared by the example training scripts.

`GaussianTrainingState` keeps everything that is indexed by the model's
gaussians -- the optimizer's moments, the densification accumulators and
the 3D filter -- following the model when densify, seed or prune replace
it. That used to be done by hand at every such call site in every script,
and missing one piece there is not an error: an accumulator left at the
old size takes the rasterizer's in-place atomic adds for the wrong
gaussians (now rejected at the op boundary, see metalsplat.ops._validate),
and a stale 3D filter band-limits the wrong ones.

The initialization helpers (`estimate_scene_scale`,
`calibrate_initial_scale`) and `save_image` were likewise copies in each
script.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from metalsplat.filter3d import carry_filter_3d
from metalsplat.optim import migrate_optimizer_state
from metalsplat.splat_model import SplatModel


class GaussianTrainingState:
    """The model being trained plus its per-gaussian training state.

    `make_optimizer(model)` builds the optimizer; it is called again for
    every replacement model, and the old optimizer's per-group learning
    rates (e.g. a scheduled means rate) and Adam moments are carried over.

    Attributes, all sized to `model.num_points`:
    - `grad_accum`: pass as `abs_grad_accum=` to render / render_2dgs.
    - `grad_count`: steps each gaussian was visible (`record_visibility`).
    - `pixel_count`: pass as `pixel_count_accum=` to render_2dgs; None
      unless `track_pixel_count`.
    - `filter_3d`: the 3DGS 3D filter, or None.
    """

    def __init__(
        self,
        model: SplatModel,
        make_optimizer: Callable[[SplatModel], torch.optim.Optimizer],
        filter_3d: torch.Tensor | None = None,
        track_pixel_count: bool = False,
    ):
        self.model = model
        self._make_optimizer = make_optimizer
        self.optimizer = make_optimizer(model)
        self.filter_3d = filter_3d
        self._track_pixel_count = track_pixel_count
        self.reset_accumulators()

    def reset_accumulators(self) -> None:
        """Starts a new densification window: zeroes the accumulators."""
        n, device = self.model.num_points, self.model.means.device
        self.grad_accum = torch.zeros(n, device=device)
        self.grad_count = torch.zeros(n, device=device)
        self.pixel_count = (
            torch.zeros(n, device=device) if self._track_pixel_count else None
        )

    @torch.no_grad()
    def record_visibility(self, visible: torch.Tensor) -> None:
        """Counts this step towards `grad_count` for the `visible` gaussians."""
        self.grad_count[visible] += 1.0

    def replace_model(
        self,
        model: SplatModel,
        source_index: torch.Tensor,
        parent_index: torch.Tensor | None = None,
    ) -> None:
        """Swaps in `model`, whose gaussians came from the current one.

        `source_index` / `parent_index` are what densify, seed and prune
        return (see DensifyStats): the optimizer's moments follow
        `source_index` (new gaussians start cold), the 3D filter follows
        `parent_index` (children inherit their parent's radius), and the
        accumulators restart at the new size. A model returned unchanged
        (the same object) keeps everything as it is.
        """
        if model is self.model:
            return
        new_optimizer = self._make_optimizer(model)
        for new_group, old_group in zip(
            new_optimizer.param_groups, self.optimizer.param_groups
        ):
            new_group["lr"] = old_group["lr"]
        self.optimizer = migrate_optimizer_state(
            self.optimizer, new_optimizer, source_index
        )
        self.filter_3d = carry_filter_3d(
            self.filter_3d, source_index if parent_index is None else parent_index
        )
        self.model = model
        self.reset_accumulators()


def estimate_scene_scale(points: torch.Tensor, sample_size: int = 5000) -> float:
    """Median nearest-neighbor distance between sparse points -- a natural
    "local spacing" unit for a scene's (COLMAP-arbitrary) coordinate scale,
    used both for the initial gaussian scale and to calibrate the means
    learning rate to that scale rather than a fixed absolute value.
    """
    n = points.shape[0]
    idx = torch.randperm(n, device=points.device)[: min(sample_size, n)]
    sample = points[idx]
    d = torch.cdist(sample, sample)
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.median().item()


def calibrate_initial_scale(
    points: torch.Tensor,
    cameras: list,
    scene_scale: float,
    target_pixel_radius: float = 3.0,
    sample_points: int = 3000,
    sample_cameras: int = 20,
) -> float:
    """Scales `scene_scale` so a gaussian of that size projects to roughly
    `target_pixel_radius` pixels on screen. Needed because the raw
    world-space nearest-neighbor spacing says nothing about how zoomed-in
    the cameras are -- on the garden scene it was ~0.12 world units but the
    camera sits close enough to a locally dense cluster that it projected
    to a ~23px radius, so 138k mostly-overlapping oversized gaussians
    produced an undifferentiated blur with no usable per-gaussian
    gradient signal. Pass the training cameras only.
    """
    n = points.shape[0]
    idx = torch.randperm(n, device=points.device)[: min(sample_points, n)]
    sample = points[idx]

    cam_idx = torch.randperm(len(cameras))[: min(sample_cameras, len(cameras))]
    magnifications = []
    for ci in cam_idx.tolist():
        cam = cameras[ci]
        pts_cam = sample.to(cam.R_wc.device) @ cam.R_wc.T + cam.t_wc
        z = pts_cam[:, 2]
        in_front = z > 0.1
        if in_front.any():
            magnifications.append((cam.fx / z[in_front]).median().item())

    median_magnification = torch.tensor(magnifications).median().item()
    multiplier = target_pixel_radius / (scene_scale * median_magnification)
    return scene_scale * multiplier


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Writes an (H, W, 3) image in [0, 1] (clamped) as an 8-bit file."""
    from PIL import Image

    arr = (tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
