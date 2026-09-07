from pathlib import Path

import pytest
import torch

from metalsplat.data.colmap import load_colmap_scene

GARDEN_ROOT = Path(__file__).parent.parent / "data" / "garden"

pytestmark = pytest.mark.skipif(
    not GARDEN_ROOT.exists(), reason="data/garden not present locally"
)


def test_load_garden_scene():
    scene = load_colmap_scene(GARDEN_ROOT)

    n = len(scene.cameras)
    assert n > 0
    assert len(scene.images) == n
    assert len(scene.image_names) == n

    assert scene.points.shape[0] > 0
    assert scene.points.shape == (scene.points.shape[0], 3)
    assert scene.colors.shape == scene.points.shape
    assert scene.colors.min() >= 0.0 and scene.colors.max() <= 1.0

    cam = scene.cameras[0]
    image = scene.images[0]
    assert image.shape == (cam.img_height, cam.img_width, 3)
    assert image.min() >= 0.0 and image.max() <= 1.0

    # world-to-camera rotation should be a proper rotation matrix
    det = torch.linalg.det(cam.R_wc)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-3)
    should_be_identity = cam.R_wc @ cam.R_wc.T
    assert torch.allclose(should_be_identity, torch.eye(3), atol=1e-3)

    # intrinsics should be positive and roughly image-sized
    assert cam.fx > 0 and cam.fy > 0
    assert 0 < cam.cx < cam.img_width
    assert 0 < cam.cy < cam.img_height
