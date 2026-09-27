import pytest
import torch

from metalsplat.gaussians import GaussianModel
from metalsplat.gaussians_2dgs import Gaussian2DModel
from metalsplat.optim import NEW_GAUSSIAN, SparseAdam
from metalsplat.training import GaussianTrainingState, estimate_scene_scale


def _make_optimizer(m):
    return SparseAdam(
        [
            {"params": [m.means], "lr": 0.1},
            {"params": [m.raw_scales], "lr": 0.01},
            {"params": [m.raw_quats], "lr": 0.01},
            {"params": [m.raw_colors], "lr": 0.01},
            {"params": [m.raw_opacities], "lr": 0.01},
        ]
    )


def _stepped_state(model_cls=GaussianModel, n=5, **kwargs):
    model = model_cls(torch.randn(n, 3))
    state = GaussianTrainingState(model, _make_optimizer, **kwargs)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    state.optimizer.step(torch.ones(n, dtype=torch.bool))
    return state


@pytest.mark.parametrize("model_cls", [GaussianModel, Gaussian2DModel])
def test_replace_model_carries_everything_per_gaussian(model_cls):
    state = _stepped_state(model_cls, filter_3d=torch.arange(5.0))
    state.optimizer.param_groups[0]["lr"] = 0.0123  # e.g. a scheduled means rate
    old_model, old_optimizer = state.model, state.optimizer

    source = torch.tensor([4, 1, NEW_GAUSSIAN])
    parent = torch.tensor([4, 1, 1])
    new_model = old_model.select(torch.tensor([4, 1, 1]))
    state.replace_model(new_model, source, parent)

    assert state.model is new_model
    assert state.optimizer.param_groups[0]["lr"] == 0.0123
    old_avg = old_optimizer.state[old_model.means]["exp_avg"]
    new_avg = state.optimizer.state[new_model.means]["exp_avg"]
    assert torch.equal(new_avg[:2], old_avg[torch.tensor([4, 1])])
    assert torch.count_nonzero(new_avg[2]) == 0  # a new gaussian starts cold
    assert torch.equal(state.filter_3d, torch.tensor([4.0, 1.0, 1.0]))
    assert state.grad_accum.shape == state.grad_count.shape == (3,)


def test_replacing_with_the_same_model_keeps_state():
    state = _stepped_state()
    state.record_visibility(torch.tensor([True, False, True, False, False]))
    optimizer = state.optimizer

    state.replace_model(state.model, torch.arange(5))

    assert state.optimizer is optimizer
    assert torch.equal(state.grad_count, torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]))


def test_pixel_count_only_when_tracked():
    assert _stepped_state().pixel_count is None
    tracked = _stepped_state(Gaussian2DModel, track_pixel_count=True)
    assert tracked.pixel_count.shape == (5,)


def test_estimate_scene_scale_is_the_nearest_neighbour_spacing():
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(6.0)] * 3, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    assert estimate_scene_scale(grid * 0.25) == pytest.approx(0.25)
