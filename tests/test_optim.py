"""Adam state surviving a change in gaussian count."""

import torch

from metalsplat.densify import densify_and_prune, prune_low_opacity
from metalsplat.gaussians import GaussianModel
from metalsplat.optim import NEW_GAUSSIAN, migrate_optimizer_state


def _model(n=10):
    # Non-zero means on purpose: the loss below is quadratic, so means
    # sitting exactly at the origin would get zero gradient and Adam would
    # accumulate nothing, making these tests pass on empty state.
    torch.manual_seed(n)
    return GaussianModel(
        torch.randn(n, 3),
        scales=torch.full((n, 3), 0.5),
        opacities=torch.full((n,), 0.5),
        colors=torch.rand(n, 3),
    )


def _optimizer(m):
    return torch.optim.Adam(
        [
            {"params": [m.means], "lr": 1e-3},
            {"params": [m.raw_scales, m.raw_quats, m.raw_colors], "lr": 1e-2},
            {"params": [m.raw_opacities], "lr": 5e-2},
        ]
    )


def _take_steps(model, opt, steps=5):
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (
            model.means.pow(2).sum()
            + model.raw_scales.pow(2).sum()
            + model.raw_quats.pow(2).sum()
            + model.raw_colors.pow(2).sum()
            + model.raw_opacities.pow(2).sum()
        )
        loss.backward()
        opt.step()


def test_moments_are_carried_across_a_prune():
    model = _model(10)
    opt = _optimizer(model)
    _take_steps(model, opt)
    before = opt.state[model.means]["exp_avg"].clone()
    assert before.abs().sum() > 0  # Adam actually accumulated something

    with torch.no_grad():
        model.raw_opacities[3] = -10.0
    pruned, n_pruned, source_index = prune_low_opacity(model)
    assert n_pruned == 1

    new_opt = migrate_optimizer_state(opt, _optimizer(pruned), source_index)
    after = new_opt.state[pruned.means]["exp_avg"]

    keep = [i for i in range(10) if i != 3]
    assert torch.allclose(after, before[keep])


def test_new_gaussians_start_from_zero_state():
    model = _model(6)
    opt = _optimizer(model)
    _take_steps(model, opt)

    # Two survivors, one brand-new gaussian.
    source_index = torch.tensor([0, 2, NEW_GAUSSIAN])
    grown = _model(3)
    new_opt = migrate_optimizer_state(opt, _optimizer(grown), source_index)

    before = opt.state[model.means]["exp_avg"]
    after = new_opt.state[grown.means]["exp_avg"]
    assert torch.allclose(after[0], before[0])
    assert torch.allclose(after[1], before[2])
    assert torch.equal(after[2], torch.zeros(3))


def test_step_count_is_preserved():
    # Resetting `step` restarts Adam's bias correction, which inflates the
    # effective step size right after every densification.
    model = _model(4)
    opt = _optimizer(model)
    _take_steps(model, opt, steps=7)
    step_before = opt.state[model.means]["step"]

    new_opt = migrate_optimizer_state(opt, _optimizer(model), torch.arange(4))
    step_after = new_opt.state[model.means]["step"]

    assert float(step_after) == float(step_before) > 0


def test_every_parameter_group_is_migrated():
    model = _model(8)
    opt = _optimizer(model)
    _take_steps(model, opt)

    source_index = torch.arange(8)
    new_opt = migrate_optimizer_state(opt, _optimizer(model), source_index)

    for param in (
        model.means,
        model.raw_scales,
        model.raw_quats,
        model.raw_colors,
        model.raw_opacities,
    ):
        old_state = opt.state[param]
        new_state = new_opt.state[param]
        assert torch.allclose(new_state["exp_avg"], old_state["exp_avg"])
        assert torch.allclose(new_state["exp_avg_sq"], old_state["exp_avg_sq"])


def test_densify_source_index_maps_survivors_correctly():
    n = 10
    model = _model(n)
    opt = _optimizer(model)
    _take_steps(model, opt)
    with torch.no_grad():
        model.raw_scales[0] = torch.log(torch.tensor(2.0))  # split candidate

    grad_count = torch.ones(n)
    grad_accum = torch.full((n,), 1.0)
    grad_accum[0] = 10.0
    grad_accum[1] = 9.0
    densified, stats = densify_and_prune(
        model, grad_accum, grad_count, scene_scale=1.0, grad_percentile=0.8
    )

    assert stats.source_index.shape[0] == densified.num_points
    # Survivors point at real old rows; split children and clones are new.
    assert (
        int((stats.source_index == NEW_GAUSSIAN).sum())
        == 2 * stats.n_split + stats.n_cloned
    )
    existing = stats.source_index[stats.source_index >= 0]
    assert existing.max() < n
    assert existing.unique().numel() == existing.numel()  # no old gaussian reused twice

    new_opt = migrate_optimizer_state(opt, _optimizer(densified), stats.source_index)
    before = opt.state[model.means]["exp_avg"]
    after = new_opt.state[densified.means]["exp_avg"]
    assert torch.allclose(after[: existing.numel()], before[existing])
    assert after[existing.numel() :].abs().sum() == 0


def test_a_no_op_prune_still_yields_a_usable_identity_mapping():
    model = _model(5)
    opt = _optimizer(model)
    _take_steps(model, opt)

    same, n_pruned, source_index = prune_low_opacity(model)
    assert n_pruned == 0 and same is model
    assert torch.equal(source_index, torch.arange(5))

    new_opt = migrate_optimizer_state(opt, _optimizer(model), source_index)
    assert torch.allclose(
        new_opt.state[model.means]["exp_avg"], opt.state[model.means]["exp_avg"]
    )
