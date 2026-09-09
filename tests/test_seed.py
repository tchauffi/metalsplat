import torch

from metalsplat.camera import Camera
from metalsplat.gaussians import GaussianModel
from metalsplat.seed import seed_uncovered_regions

H = W = 32


def _model_and_camera():
    means = torch.tensor([[0.0, 0.0, 5.0]])
    model = GaussianModel(means, colors=torch.tensor([[0.5, 0.5, 0.5]]))
    camera = Camera.identity(
        fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H
    )
    return model, camera


def test_seeds_only_high_residual_uncovered_pixels():
    model, camera = _model_and_camera()
    pred = torch.zeros(H, W, 3)
    target = torch.zeros(H, W, 3)
    target[0, 0] = torch.tensor([1.0, 0.0, 0.0])  # a single bright red uncovered pixel
    final_T = torch.ones(H, W)  # nothing covered anywhere

    new_model, stats = seed_uncovered_regions(
        model,
        camera,
        pred,
        target,
        final_T,
        init_scale=0.05,
        residual_thresh=0.1,
        coverage_thresh=0.8,
    )

    assert stats.n_seeded == 1
    assert new_model.num_points == model.num_points + 1
    assert torch.allclose(
        new_model.colors[-1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-3
    )


def test_no_seeding_when_residual_low():
    model, camera = _model_and_camera()
    pred = torch.full((H, W, 3), 0.5)
    target = torch.full((H, W, 3), 0.5)  # perfect match everywhere
    final_T = torch.ones(H, W)

    new_model, stats = seed_uncovered_regions(
        model, camera, pred, target, final_T, init_scale=0.05
    )

    assert stats.n_seeded == 0
    assert new_model.num_points == model.num_points


def test_no_seeding_when_already_covered():
    model, camera = _model_and_camera()
    pred = torch.zeros(H, W, 3)
    target = torch.ones(H, W, 3)  # high residual everywhere...
    final_T = torch.zeros(H, W)  # ...but fully covered everywhere

    _new_model, stats = seed_uncovered_regions(
        model, camera, pred, target, final_T, init_scale=0.05
    )

    assert stats.n_seeded == 0


def test_max_seeds_per_call_caps_growth():
    model, camera = _model_and_camera()
    pred = torch.zeros(H, W, 3)
    target = torch.ones(H, W, 3)  # every pixel needs seeding
    final_T = torch.ones(H, W)

    new_model, stats = seed_uncovered_regions(
        model,
        camera,
        pred,
        target,
        final_T,
        init_scale=0.05,
        max_seeds_per_call=10,
    )

    assert stats.n_seeded == 10
    assert new_model.num_points == model.num_points + 10


def test_max_points_stops_seeding():
    model, camera = _model_and_camera()
    pred = torch.zeros(H, W, 3)
    target = torch.ones(H, W, 3)
    final_T = torch.ones(H, W)

    new_model, stats = seed_uncovered_regions(
        model,
        camera,
        pred,
        target,
        final_T,
        init_scale=0.05,
        max_points=model.num_points,
    )

    assert stats.n_seeded == 0
    assert new_model.num_points == model.num_points


def test_sh_model_seeds_preserve_dc_color():
    means = torch.tensor([[0.0, 0.0, 5.0]])
    model = GaussianModel(means, colors=torch.tensor([[0.5, 0.5, 0.5]]), sh_degree=2)
    camera = Camera.identity(
        fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H
    )

    pred = torch.zeros(H, W, 3)
    target = torch.zeros(H, W, 3)
    target[5, 5] = torch.tensor([0.2, 0.8, 0.3])
    final_T = torch.ones(H, W)

    new_model, stats = seed_uncovered_regions(
        model,
        camera,
        pred,
        target,
        final_T,
        init_scale=0.05,
        residual_thresh=0.1,
    )

    assert stats.n_seeded == 1
    assert new_model.sh_degree == 2
    assert new_model.raw_sh.shape == (2, 9, 3)


def test_seeding_preserves_active_sh_degree():
    means = torch.tensor([[0.0, 0.0, 5.0]])
    model = GaussianModel(means, colors=torch.tensor([[0.5, 0.5, 0.5]]), sh_degree=2)
    model.active_sh_degree = 1
    camera = Camera.identity(
        fx=32.0, fy=32.0, cx=W / 2, cy=H / 2, img_width=W, img_height=H
    )

    pred = torch.zeros(H, W, 3)
    target = torch.zeros(H, W, 3)
    target[5, 5] = torch.tensor([0.9, 0.1, 0.2])

    seeded, stats = seed_uncovered_regions(
        model,
        camera,
        pred,
        target,
        torch.ones(H, W),
        init_scale=0.05,
        residual_thresh=0.1,
    )

    assert stats.n_seeded == 1
    assert seeded.active_sh_degree == 1
