import torch

from metalsplat.cleanup import damp_view_dependence, prune_isolated
from metalsplat.gaussians import GaussianModel


def _clustered_model_with_floaters(sh_degree=0):
    # A dense 4x4x4 cluster (64 gaussians in a tight grid) plus 3 lone
    # gaussians far away -- the floaters.
    grid = torch.stack(
        torch.meshgrid(torch.arange(4.0), torch.arange(4.0), torch.arange(4.0), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3) * 0.05
    floaters = torch.tensor([[10.0, 10.0, 10.0], [-8.0, 4.0, 9.0], [20.0, -5.0, 3.0]])
    means = torch.cat([grid, floaters])
    colors = torch.rand(means.shape[0], 3)
    return GaussianModel(means, colors=colors, sh_degree=sh_degree), grid.shape[0]


def test_prune_isolated_removes_floaters_keeps_cluster():
    model, n_cluster = _clustered_model_with_floaters()
    pruned, n_pruned = prune_isolated(model, cell_size=0.5, min_per_cell=8)

    assert n_pruned == 3
    assert pruned.num_points == n_cluster
    # every survivor is from the tight cluster near the origin
    assert pruned.means.abs().max() < 1.0


def test_prune_isolated_is_a_no_op_when_nothing_is_isolated():
    means = torch.rand(50, 3) * 0.1
    model = GaussianModel(means, colors=torch.rand(50, 3))

    pruned, n_pruned = prune_isolated(model, cell_size=1.0, min_per_cell=2)

    assert n_pruned == 0
    assert pruned is model


def test_prune_isolated_preserves_sh_coefficients():
    model, n_cluster = _clustered_model_with_floaters(sh_degree=2)
    with torch.no_grad():
        model.raw_sh[:, 2, :] = 0.33

    pruned, n_pruned = prune_isolated(model, cell_size=0.5, min_per_cell=8)

    assert n_pruned == 3
    assert pruned.sh_degree == 2
    assert torch.allclose(pruned.raw_sh[:, 2, :], torch.full((n_cluster, 3), 0.33))


def test_damp_view_dependence_scales_only_non_dc():
    model = GaussianModel(torch.randn(6, 3), colors=torch.rand(6, 3), sh_degree=2)
    with torch.no_grad():
        model.raw_sh.copy_(torch.randn(6, 9, 3))
    original = model.raw_sh.detach().clone()

    damped = damp_view_dependence(model, factor=0.5)

    assert torch.allclose(damped.raw_sh[:, 0, :], original[:, 0, :], atol=1e-6)  # DC untouched
    assert torch.allclose(damped.raw_sh[:, 1:, :], original[:, 1:, :] * 0.5, atol=1e-6)
    assert damped.num_points == model.num_points


def test_damp_view_dependence_is_a_no_op_for_flat_rgb():
    model = GaussianModel(torch.randn(4, 3), colors=torch.rand(4, 3))
    assert damp_view_dependence(model, factor=0.5) is model


def test_cleanup_preserves_active_sh_degree():
    model, _ = _clustered_model_with_floaters(sh_degree=2)
    model.active_sh_degree = 1

    pruned, _ = prune_isolated(model, cell_size=0.5, min_per_cell=8)
    assert pruned.active_sh_degree == 1
    assert damp_view_dependence(model, 0.5).active_sh_degree == 1
