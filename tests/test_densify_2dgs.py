import torch

from metalsplat.densify import reset_opacity
from metalsplat.densify2dgs import densify_and_prune_2dgs, prune_low_opacity_2dgs
from metalsplat.gaussians_2dgs import Gaussian2DModel

SCENE_SCALE = 1.0


def _model(n=10):
    means = torch.zeros(n, 3)
    scales = torch.full((n, 2), 0.5)  # below scene_scale by default
    opacities = torch.full((n,), 0.5)
    colors = torch.rand(n, 3)
    return Gaussian2DModel(means, scales=scales, opacities=opacities, colors=colors)


def test_split_clone_and_prune():
    n = 10
    model = _model(n)
    with torch.no_grad():
        model.raw_scales[0] = torch.log(
            torch.tensor(2.0)
        )  # gaussian 0: large -> split candidate
        model.raw_opacities[5] = -10.0  # gaussian 5: opacity ~0 -> pruned

    grad_count = torch.ones(n)
    grad_count[3] = 0  # never visible
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0  # highest grad, large scale -> split
    grad_accum[1] = 9.0  # second-highest grad, small scale -> clone

    new_model, stats = densify_and_prune_2dgs(
        model,
        grad_accum,
        grad_count,
        scene_scale=SCENE_SCALE,
        grad_percentile=0.8,
    )

    assert stats.n_before == n
    assert stats.n_split == 1
    assert stats.n_cloned == 1
    assert stats.n_pruned == 1  # gaussian 5, low opacity
    assert stats.n_after == n + 1
    assert new_model.num_points == n + 1
    assert new_model.scales.shape == (n + 1, 2)


def test_split_child_offset_lies_in_parent_tangent_plane():
    n = 5
    model = _model(n)
    with torch.no_grad():
        # A non-trivial orientation, so the tangent plane isn't axis-aligned.
        raw = torch.tensor([0.7, 0.2, -0.5, 0.3])
        model.raw_quats[0] = raw / raw.norm()
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # split candidate

    parent_mean = model.means[0].clone()
    parent_normal = model.rotmat[0, :, 2].clone()

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0

    new_model, stats = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.8
    )
    assert stats.n_split == 1

    # The two split children are the last two rows in the concatenation
    # order (keep, split, clone) -- with no clones here, they're the last 2.
    children = new_model.means[-2:]
    offsets = children - parent_mean
    dots = (offsets * parent_normal).sum(-1)
    assert torch.allclose(dots, torch.zeros(2), atol=1e-5)


def test_no_visible_gaussians_is_a_no_op():
    n = 5
    model = _model(n)
    grad_count = torch.zeros(n)
    grad_accum = torch.zeros(n)

    new_model, stats = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE
    )

    assert stats.n_split == 0 and stats.n_cloned == 0 and stats.n_pruned == 0
    assert new_model.num_points == n


def test_max_points_stops_densification():
    n = 10
    model = _model(n)
    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0

    new_model, stats = densify_and_prune_2dgs(
        model,
        grad_accum,
        grad_count,
        scene_scale=SCENE_SCALE,
        max_points=n,
    )

    assert new_model.num_points == n
    assert stats.n_split == 0 and stats.n_cloned == 0


def test_split_clone_and_prune_preserves_sh_coefficients():
    n = 10
    means = torch.zeros(n, 3)
    scales = torch.full((n, 2), 0.5)
    opacities = torch.full((n,), 0.5)
    colors = torch.rand(n, 3)
    model = Gaussian2DModel(
        means, scales=scales, opacities=opacities, colors=colors, sh_degree=2
    )
    with torch.no_grad():
        model.raw_sh[:, 1, :] = 0.7  # a non-DC coefficient, should survive densify
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # split candidate
        model.raw_opacities[5] = -10.0  # pruned

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    grad_accum[1] = 9.0

    new_model, stats = densify_and_prune_2dgs(
        model,
        grad_accum,
        grad_count,
        scene_scale=SCENE_SCALE,
        grad_percentile=0.8,
    )

    assert new_model.sh_degree == 2
    assert new_model.raw_sh.shape == (stats.n_after, 9, 3)
    assert torch.allclose(
        new_model.raw_sh[:, 1, :], torch.full((stats.n_after, 3), 0.7)
    )


def test_reset_opacity_caps_high_opacities_in_place():
    n = 5
    model = _model(n)
    with torch.no_grad():
        model.raw_opacities[0] = 10.0  # opacity ~1.0, above the reset value
        model.raw_opacities[1] = -5.0  # opacity ~0.0067, already below the reset value
    already_low_opacity = model.opacities[1].item()

    param_before = model.raw_opacities
    reset_opacity(model, value=0.01)

    assert model.raw_opacities is param_before
    assert model.opacities[0].item() <= 0.01 + 1e-6
    assert abs(model.opacities[1].item() - already_low_opacity) < 1e-6
    assert model.opacities[2].item() <= 0.01 + 1e-6


def test_prune_low_opacity_removes_only_below_threshold():
    n = 5
    model = _model(n)
    with torch.no_grad():
        model.raw_opacities[2] = -10.0  # opacity ~0, should be pruned

    new_model, n_pruned, _ = prune_low_opacity_2dgs(model, prune_opacity_thresh=0.005)

    assert n_pruned == 1
    assert new_model.num_points == n - 1


def test_prune_low_opacity_no_op_when_nothing_below_threshold():
    n = 5
    model = _model(n)

    new_model, n_pruned, _ = prune_low_opacity_2dgs(model, prune_opacity_thresh=0.005)

    assert n_pruned == 0
    assert new_model is model  # returned unchanged, no rebuild needed


def test_rebuilds_preserve_active_sh_degree():
    n = 10
    means = torch.zeros(n, 3)
    model = Gaussian2DModel(
        means,
        scales=torch.full((n, 2), 0.5),
        opacities=torch.full((n,), 0.5),
        colors=torch.rand(n, 3),
        sh_degree=2,
    )
    model.active_sh_degree = 1

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    densified, _ = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.8
    )
    assert densified.active_sh_degree == 1

    with torch.no_grad():
        densified.raw_opacities[0] = -10.0
    pruned, n_pruned, _ = prune_low_opacity_2dgs(densified)
    assert n_pruned > 0
    assert pruned.active_sh_degree == 1


def _grad_scene(n, top_frac=0.1, big=10.0, small=1.0):
    model = _model(n)
    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), small)
    grad_accum[: int(n * top_frac)] = big
    return model, grad_accum, grad_count


def test_absolute_threshold_selects_only_gaussians_above_it():
    n = 100
    model, grad_accum, grad_count = _grad_scene(n)

    _, stats = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_threshold=5.0
    )
    assert stats.n_split + stats.n_cloned == 10

    _, none_stats = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_threshold=50.0
    )
    assert none_stats.n_split + none_stats.n_cloned == 0
    assert none_stats.n_after == none_stats.n_before


def test_percentile_promotes_a_fixed_fraction_however_well_fit():
    n = 100
    model = _model(n)
    grad_count = torch.ones(n)

    for magnitude in (10.0, 1e-6):
        _, stats = densify_and_prune_2dgs(
            model,
            torch.full((n,), magnitude),
            grad_count,
            scene_scale=SCENE_SCALE,
            grad_percentile=0.9,
        )
        assert stats.n_split + stats.n_cloned > 0, (
            "percentile densified nothing; the test no longer shows the problem"
        )


def test_calibrated_threshold_makes_densification_decay():
    n = 200
    model, grad_accum, grad_count = _grad_scene(n)

    _, first = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.9
    )
    bar = first.grad_threshold
    assert bar > 0

    promoted = []
    for decay in (1.0, 0.5, 0.1):
        m = _model(n)
        _, stats = densify_and_prune_2dgs(
            m,
            grad_accum * decay,
            grad_count,
            scene_scale=SCENE_SCALE,
            grad_threshold=bar,
        )
        promoted.append(stats.n_split + stats.n_cloned)

    assert promoted[0] > 0
    assert promoted == sorted(promoted, reverse=True), promoted
    assert promoted[-1] == 0, "a settled model should stop densifying entirely"


def test_reported_threshold_round_trips():
    n = 50
    model, grad_accum, grad_count = _grad_scene(n)
    _, stats = densify_and_prune_2dgs(
        model, grad_accum, grad_count, scene_scale=SCENE_SCALE, grad_percentile=0.9
    )
    m2 = _model(n)
    _, again = densify_and_prune_2dgs(
        m2,
        grad_accum,
        grad_count,
        scene_scale=SCENE_SCALE,
        grad_threshold=stats.grad_threshold,
    )
    assert again.n_split + again.n_cloned == stats.n_split + stats.n_cloned
