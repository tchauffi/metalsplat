"""Loads a COLMAP sparse reconstruction (cameras/images/points3D) into
Camera objects, images, and an initial point cloud for GaussianModel.

Expects the standard COLMAP project layout:
    <scene_root>/sparse/0/{cameras,images,points3D}.bin
    <scene_root>/<image_dir>/*.jpg  (image_dir defaults to "images")

The images on disk may be downsampled relative to the resolution COLMAP was
calibrated at (common in released datasets, e.g. MipNeRF360's `images` vs.
`images_4`); intrinsics are rescaled per-image to match each image's actual
size on disk.

Only undistorted camera models (PINHOLE, SIMPLE_PINHOLE) are supported --
this pipeline's projection assumes an ideal pinhole camera with no lens
distortion (see README roadmap).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# torch and pycolmap each bundle their own OpenMP runtime; loading both in
# one process aborts with "OMP: Error #15: Initializing libomp.dylib, but
# found libomp.dylib already initialized" unless this is set before
# pycolmap is imported. Benign here -- there's no shared OpenMP state
# between the two libraries in this codebase.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pycolmap
import torch
from PIL import Image

from metalsplat.camera import Camera

_SUPPORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


class ImageStore(Sequence):
    """Training images held as uint8 on device, converted on access.

    A capture's images dominate GPU memory, not the model. On the garden
    scene (185 images at 1297x840) they take 2.42GB as float32 against
    103MB for a 438k-gaussian degree-3 model plus 207MB of Adam state --
    24x the model, and the reason an "idle" session showed 2.5GB
    allocated. Stored as uint8 they take 0.60GB.

    Indexing returns float32 in [0, 1], so callers are unchanged and there
    is no way to accidentally use raw 0-255 values as if they were
    normalised. The cost is one cast per access -- about 13MB for a single
    image, against a ~100ms training step, so it does not register.

    Use `.nbytes` to measure the store itself; iterating it materialises
    every image as float and defeats the point.
    """

    def __init__(self, images_u8: list[torch.Tensor]):
        self._images = images_u8

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._images[index].float() / 255.0

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._images)

    @property
    def raw(self) -> list[torch.Tensor]:
        """The underlying uint8 tensors, for callers that want them undivided."""
        return self._images


@dataclass
class ColmapScene:
    cameras: list[Camera]
    images: ImageStore  # indexes to (H, W, 3) float32 in [0, 1], aligned with `cameras`
    image_names: list[str]
    points: torch.Tensor  # (P, 3) float32
    colors: torch.Tensor  # (P, 3) float32 in [0, 1]


def _camera_intrinsics(cam: pycolmap.Camera) -> tuple[float, float, float, float]:
    model = cam.model.name
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported COLMAP camera model '{model}' -- only {sorted(_SUPPORTED_MODELS)} "
            "are supported (this pipeline assumes an undistorted pinhole camera)."
        )
    if model == "PINHOLE":
        fx, fy, cx, cy = cam.params
    else:  # SIMPLE_PINHOLE
        f, cx, cy = cam.params
        fx = fy = f
    return float(fx), float(fy), float(cx), float(cy)


def load_colmap_scene(
    scene_root: str | Path,
    image_dir: str = "images",
    sparse_subdir: str = "sparse/0",
    device: str | torch.device = "cpu",
    downscale: float = 1.0,
) -> ColmapScene:
    """`downscale` (>=1.0) resizes every loaded image by `1/downscale`
    (e.g. `downscale=2.0` halves both dimensions) -- a plain training-speed
    knob (fewer pixels per render/backward), matching 3DGS/2DGS's
    convention of a resolution downscale factor. Intrinsics don't need any
    special handling for this: the existing `scale = actual_width /
    colmap_cam.width` rescaling below already reacts to whatever size the
    image actually is after resizing, the same way it already handles
    scenes shipped with pre-downsampled image directories (e.g.
    MipNeRF360's `images_4`).
    """
    scene_root = Path(scene_root)
    rec = pycolmap.Reconstruction(str(scene_root / sparse_subdir))

    images_by_name = sorted(rec.images.values(), key=lambda im: im.name)

    cameras: list[Camera] = []
    images: list[torch.Tensor] = []
    image_names: list[str] = []

    for colmap_image in images_by_name:
        image_path = scene_root / image_dir / colmap_image.name
        pil_image = Image.open(image_path).convert("RGB")
        if downscale != 1.0:
            new_size = (
                round(pil_image.width / downscale),
                round(pil_image.height / downscale),
            )
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
        actual_width, actual_height = pil_image.size

        colmap_cam = rec.cameras[colmap_image.camera_id]
        fx, fy, cx, cy = _camera_intrinsics(colmap_cam)
        scale = actual_width / colmap_cam.width
        fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale

        cam_from_world = colmap_image.cam_from_world()
        R_wc = torch.from_numpy(
            np.array(cam_from_world.rotation.matrix(), dtype=np.float32)
        )
        t_wc = torch.from_numpy(np.array(cam_from_world.translation, dtype=np.float32))

        cameras.append(
            Camera(
                R_wc=R_wc,
                t_wc=t_wc,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_width=actual_width,
                img_height=actual_height,
            ).to(device)
        )
        # Kept as uint8; ImageStore converts on access. See its docstring.
        image_tensor = torch.from_numpy(np.array(pil_image, dtype=np.uint8))
        images.append(image_tensor.to(device))
        image_names.append(colmap_image.name)

    points3d = list(rec.points3D.values())
    points = torch.from_numpy(
        np.stack([p.xyz for p in points3d]).astype(np.float32)
    ).to(device)
    colors = torch.from_numpy(
        (np.stack([p.color for p in points3d]).astype(np.float32)) / 255.0
    ).to(device)

    return ColmapScene(
        cameras=cameras,
        images=ImageStore(images),
        image_names=image_names,
        points=points,
        colors=colors,
    )
