import torch

from metalsplat.densify import densify_and_prune, prune_low_opacity, reset_opacity
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


def test_reset_opacity_caps_high_opacities_in_place():
    n = 5
    model = _model(n)
    with torch.no_grad():
        model.raw_opacities[0] = 10.0  # opacity ~1.0, above the reset value
        model.raw_opacities[1] = -5.0  # opacity ~0.0067, already below the reset value
    already_low_opacity = model.opacities[1].item()

    param_before = model.raw_opacities  # same nn.Parameter object, to check in-place semantics
    reset_opacity(model, value=0.01)

    assert model.raw_opacities is param_before  # in-place: no new Parameter, optimizer state stays valid
    assert model.opacities[0].item() <= 0.01 + 1e-6
    assert abs(model.opacities[1].item() - already_low_opacity) < 1e-6  # already below value, untouched
    assert model.opacities[2].item() <= 0.01 + 1e-6  # baseline 0.5 is above the reset value too


def test_prune_low_opacity_removes_only_below_threshold():
    n = 5
    model = _model(n)
    with torch.no_grad():
        model.raw_opacities[2] = -10.0  # opacity ~0, should be pruned

    new_model, n_pruned, _ = prune_low_opacity(model, prune_opacity_thresh=0.005)

    assert n_pruned == 1
    assert new_model.num_points == n - 1


def test_prune_low_opacity_no_op_when_nothing_below_threshold():
    n = 5
    model = _model(n)

    new_model, n_pruned, _ = prune_low_opacity(model, prune_opacity_thresh=0.005)

    assert n_pruned == 0
    assert new_model is model  # returned unchanged, no rebuild needed


def test_rebuilds_preserve_active_sh_degree():
    # densify/prune/seed all rebuild the model through the GaussianModel
    # constructor. If they let active_sh_degree default back to sh_degree,
    # a progressive-SH schedule is silently disabled the first time
    # densification runs -- no error, just a quietly ignored feature.
    n = 10
    means = torch.zeros(n, 3)
    model = GaussianModel(
        means, scales=torch.full((n, 3), 0.5), opacities=torch.full((n,), 0.5),
        colors=torch.rand(n, 3), sh_degree=2,
    )
    model.active_sh_degree = 1

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    densified, _ = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.8
    )
    assert densified.active_sh_degree == 1

    with torch.no_grad():
        densified.raw_opacities[0] = -10.0
    pruned, n_pruned, _ = prune_low_opacity(densified)
    assert n_pruned > 0
    assert pruned.active_sh_degree == 1
