import torch

from metalsplat.densify import densify_and_prune
from metalsplat.gaussians import GaussianModel

SCENE_SCALE = 1.0


def _model(n=10):
    means = torch.zeros(n, 3)
    scales = torch.full((n, 3), 0.5)  # below scene_scale by default
    opacities = torch.full((n,), 0.5)
    colors = torch.rand(n, 3)
    return GaussianModel(means, scales=scales, opacities=opacities, colors=colors)


def test_split_clone_and_prune():
    n = 10
    model = _model(n)
    with torch.no_grad():
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # gaussian 0: large -> split candidate
        model.raw_opacities[5] = -10.0  # gaussian 5: opacity ~0 -> pruned

    grad_count = torch.ones(n)
    grad_count[3] = 0  # never visible
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0  # highest grad, large scale -> split
    grad_accum[1] = 9.0  # second-highest grad, small scale -> clone

    new_model, stats = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.8,
    )

    assert stats.n_before == n
    assert stats.n_split == 1
    assert stats.n_cloned == 1
    assert stats.n_pruned == 1  # gaussian 5, low opacity
    # net change: -1 (split original removed) + 2 (split children) + 1 (clone) - 1 (pruned) = +1
    assert stats.n_after == n + 1
    assert new_model.num_points == n + 1


def test_no_visible_gaussians_is_a_no_op():
    n = 5
    model = _model(n)
    grad_count = torch.zeros(n)
    grad_accum = torch.zeros(n)

    new_model, stats = densify_and_prune(model, grad_accum, grad_count, scene_scale=SCENE_SCALE)

    assert stats.n_split == 0 and stats.n_cloned == 0 and stats.n_pruned == 0
    assert new_model.num_points == n


def test_max_points_stops_densification():
    n = 10
    model = _model(n)
    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0

    new_model, stats = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, max_points=n,
    )

    assert new_model.num_points == n
    assert stats.n_split == 0 and stats.n_cloned == 0


def test_split_clone_and_prune_preserves_sh_coefficients():
    n = 10
    means = torch.zeros(n, 3)
    scales = torch.full((n, 3), 0.5)
    opacities = torch.full((n,), 0.5)
    colors = torch.rand(n, 3)
    model = GaussianModel(means, scales=scales, opacities=opacities, colors=colors, sh_degree=2)
    with torch.no_grad():
        model.raw_sh[:, 1, :] = 0.7  # a non-DC coefficient, should survive densify
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # split candidate
        model.raw_opacities[5] = -10.0  # pruned

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    grad_accum[1] = 9.0

    new_model, stats = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.8,
    )

    assert new_model.sh_degree == 2
    assert new_model.raw_sh.shape == (stats.n_after, 9, 3)
    assert torch.allclose(new_model.raw_sh[:, 1, :], torch.full((stats.n_after, 3), 0.7))
