import torch

from metalsplat.camera import Camera


def _look_at(eye, target, up=(0.0, -1.0, 0.0)):
    return Camera.look_at(
        eye=torch.tensor(eye),
        target=torch.tensor(target),
        up=torch.tensor(up),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        img_width=100, img_height=100,
    )


def test_look_at_is_a_proper_rotation():
    cam = _look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0])
    assert torch.allclose(cam.R_wc @ cam.R_wc.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(cam.R_wc), torch.tensor(1.0), atol=1e-5)


def test_look_at_points_camera_at_target():
    eye, target = [3.0, 1.0, -4.0], [0.5, 0.2, 0.1]
    cam = _look_at(eye, target)
    target_cam = cam.R_wc @ torch.tensor(target) + cam.t_wc

    assert target_cam[2] > 0  # in front of the camera
    # and on the optical axis, so it projects to the principal point
    assert abs(target_cam[0]) < 1e-5 and abs(target_cam[1]) < 1e-5


def test_look_at_position_roundtrips():
    eye = [2.0, -1.0, 3.0]
    cam = _look_at(eye, [0.0, 0.0, 0.0])
    assert torch.allclose(cam.position, torch.tensor(eye), atol=1e-5)


def test_look_at_is_not_upside_down():
    # With world up = -y, a point above the target must land *above* the
    # principal point in image space, i.e. at a smaller row index (v grows
    # downward). Catches the 180-degree roll from using up where the
    # camera's down axis belongs.
    up = torch.tensor([0.0, -1.0, 0.0])
    cam = _look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], up=tuple(up.tolist()))

    higher = torch.tensor([0.0, 0.0, 0.0]) + up  # one unit further "up"
    p_cam = cam.R_wc @ higher + cam.t_wc
    v = cam.fy * p_cam[1] / p_cam[2] + cam.cy

    assert v < cam.cy
