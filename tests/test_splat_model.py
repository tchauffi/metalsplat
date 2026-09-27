"""Lossless row operations on the shared model base (SplatModel).

Densify, prune, seed and cleanup used to rebuild models through the
constructor, i.e. through the activations: logit clamps at 1e-4, so every
raw opacity was capped at +-9.2, and quaternions came back normalized.
They now copy raw parameters, so carried-over gaussians are bit-identical.
"""

import math

import pytest
import torch

from metalsplat.cleanup import damp_view_dependence, prune_isolated
from metalsplat.densify import densify_and_prune, prune_low_opacity
from metalsplat.densify2dgs import densify_and_prune_2dgs, prune_low_opacity_2dgs
from metalsplat.gaussians import GaussianModel
from metalsplat.gaussians_2dgs import Gaussian2DModel


def _extreme(model_cls, n=6, sh_degree=0):
    """A model whose raw values the old activation round trip altered."""
    g = torch.Generator().manual_seed(0)
    model = model_cls(torch.randn(n, 3, generator=g), sh_degree=sh_degree)
    with torch.no_grad():
        model.raw_opacities.copy_(torch.linspace(-3.0, 15.0, n))  # 15 > logit cap 9.2
        model.raw_quats.mul_(2.5)  # non-unit norm
        model.raw_quats.add_(torch.randn(n, 4, generator=g) * 0.1)
        if sh_degree == 0:
            model.raw_colors.fill_(12.0)  # also past the cap
        else:
            model.raw_sh.normal_(generator=g)
    return model


@pytest.mark.parametrize("model_cls", [GaussianModel, Gaussian2DModel])
@pytest.mark.parametrize("sh_degree", [0, 2])
def test_select_and_cat_copy_raw_parameters_exactly(model_cls, sh_degree):
    model = _extreme(model_cls, sh_degree=sh_degree)
    model.active_sh_degree = min(1, sh_degree)
    idx = torch.tensor([4, 0, 5])

    picked = model.select(idx)
    both = picked.cat(model.select(torch.tensor([1])))

    assert type(both) is model_cls
    assert (
        both.sh_degree == sh_degree and both.active_sh_degree == model.active_sh_degree
    )
    for name, value in model.raw_rows().items():
        assert torch.equal(
            getattr(both, name).detach(), value[torch.tensor([4, 0, 5, 1])]
        )
    assert [n for n, _ in both.named_parameters()] == [
        n for n, _ in model.named_parameters()
    ]


def test_with_rows_rejects_mismatched_rows():
    model = _extreme(GaussianModel)
    rows = model.raw_rows()
    rows["raw_scales"] = rows["raw_scales"][:, :2]  # a 2DGS shape on a 3DGS model
    with pytest.raises(ValueError, match="raw_scales"):
        model.with_rows(rows)
    with pytest.raises(ValueError, match="same type"):
        model.cat(_extreme(Gaussian2DModel))


@pytest.mark.parametrize(
    "densify, model_cls",
    [
        (
            lambda m, a, c: densify_and_prune(
                m, a, c, scene_scale=0.5, grad_threshold=5.0
            ),
            GaussianModel,
        ),
        (
            lambda m, a, c: densify_and_prune_2dgs(m, a, c, grad_threshold=5.0),
            Gaussian2DModel,
        ),
    ],
    ids=["3dgs", "2dgs"],
)
def test_densify_carries_untouched_gaussians_over_bit_identical(densify, model_cls):
    model = _extreme(model_cls)
    n = model.num_points
    with torch.no_grad():
        model.raw_scales[3] = math.log(2.0)  # large -> split if selected
    grad_accum = torch.ones(n)
    grad_accum[3] = 10.0  # only gaussian 3 is a candidate
    new_model, stats = densify(model, grad_accum, torch.ones(n))

    assert stats.n_split == 1 and stats.n_pruned == 0
    kept = stats.source_index >= 0
    for name, value in model.raw_rows().items():
        carried = getattr(new_model, name).detach()[kept]
        assert torch.equal(carried, value[stats.source_index[kept]]), name
    # Split children: parent's raw values, scales shrunk by exactly 1.6.
    children = ~kept
    assert torch.allclose(
        new_model.raw_scales.detach()[children],
        (model.raw_scales.detach()[3] - math.log(1.6)).expand(2, -1),
    )
    assert torch.equal(
        new_model.raw_opacities.detach()[children],
        model.raw_opacities.detach()[3].expand(2),
    )


@pytest.mark.parametrize(
    "prune, model_cls",
    [(prune_low_opacity, GaussianModel), (prune_low_opacity_2dgs, Gaussian2DModel)],
)
def test_prune_carries_survivors_over_bit_identical(prune, model_cls):
    model = _extreme(model_cls)
    with torch.no_grad():
        model.raw_opacities[1] = -10.0
    pruned, n_pruned, index = prune(model)
    assert n_pruned == 1
    for name, value in model.raw_rows().items():
        assert torch.equal(getattr(pruned, name).detach(), value[index])


def test_cleanup_is_lossless_outside_what_it_edits():
    model = _extreme(GaussianModel, n=20, sh_degree=2)
    with torch.no_grad():
        model.means.mul_(0.01)  # one dense cell, so nothing is isolated
    kept, n_pruned = prune_isolated(model, cell_size=1.0, min_per_cell=1)
    assert n_pruned == 0 and kept is model

    damped = damp_view_dependence(model, 0.5)
    for name in ("means", "raw_scales", "raw_quats", "raw_opacities"):
        assert torch.equal(getattr(damped, name), getattr(model, name))
    assert torch.equal(damped.raw_sh[:, 0], model.raw_sh[:, 0])
    assert torch.allclose(damped.raw_sh[:, 1:], model.raw_sh[:, 1:] * 0.5)
