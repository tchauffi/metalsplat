"""Carrying optimizer state across changes in gaussian count.

Densification, seeding and pruning all rebuild the GaussianModel, which
produces new nn.Parameter objects. An optimizer built over the old
parameters cannot be reused, so the obvious thing -- constructing a fresh
one -- is what this package used to do. That silently throws away Adam's
per-parameter moment estimates for *every* gaussian, not just the ones
that changed, and resets the step counter so bias correction starts over.

That matters far more than it looks. On the garden scene those three
operations fired 65 times in a 5000-step run, once every 77 steps, while
Adam's second moment with beta2=0.999 has an effective averaging window of
about a thousand steps. The optimizer therefore spends the entire run in
its warmup transient and never reaches a steady state.

`migrate_optimizer_state` fixes that by copying the moment tensors across,
row-selected to follow the gaussians. It mirrors what the reference 3DGS
implementation does in its `cat_tensors_to_optimizer` / `_prune_optimizer`
helpers.
"""

from __future__ import annotations

import torch

# Source index for a gaussian that did not exist before this operation, and
# so starts with zeroed optimizer state.
NEW_GAUSSIAN = -1


def _select(state: torch.Tensor, source_index: torch.Tensor) -> torch.Tensor:
    """Rows of `state` picked by `source_index`, zeros where the index is -1."""
    out = state.new_zeros(source_index.shape[0], *state.shape[1:])
    existing = source_index >= 0
    if bool(existing.any()):
        out[existing] = state[source_index[existing]]
    return out


@torch.no_grad()
def migrate_optimizer_state(
    old_optimizer: torch.optim.Optimizer,
    new_optimizer: torch.optim.Optimizer,
    source_index: torch.Tensor,
) -> torch.optim.Optimizer:
    """Copies Adam moments from `old_optimizer` onto `new_optimizer`.

    `source_index` is (n_after,) int64: for each gaussian in the new model,
    the index it came from in the old one, or NEW_GAUSSIAN (-1) for a
    gaussian that has just been created and should start from zero.

    The two optimizers must have the same group layout and the same
    parameter ordering within each group -- build `new_optimizer` with the
    same factory that built the old one and that holds. Returns
    `new_optimizer` for convenience.

    Parameters whose first dimension is not the gaussian count (there are
    none today, but a future global parameter would be one) are copied
    across unchanged rather than row-selected.
    """
    if len(old_optimizer.param_groups) != len(new_optimizer.param_groups):
        raise ValueError(
            "optimizer group layout changed: "
            f"{len(old_optimizer.param_groups)} -> {len(new_optimizer.param_groups)} groups"
        )

    n_before_expected = None
    for g_old, g_new in zip(old_optimizer.param_groups, new_optimizer.param_groups):
        if len(g_old["params"]) != len(g_new["params"]):
            raise ValueError("optimizer group layout changed: differing parameter counts")

        for p_old, p_new in zip(g_old["params"], g_new["params"]):
            state = old_optimizer.state.get(p_old)
            if not state or "exp_avg" not in state:
                continue  # never stepped; nothing to carry

            if n_before_expected is None:
                n_before_expected = state["exp_avg"].shape[0]

            migrated = {}
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                if tensor.shape[0] == n_before_expected and p_new.shape[0] == source_index.shape[0]:
                    migrated[key] = _select(tensor, source_index)
                else:
                    migrated[key] = tensor.clone()
            # Keep the step count: resetting it restarts Adam's bias
            # correction, which is half the reason the rebuild hurt.
            step = state.get("step")
            if step is not None:
                migrated["step"] = step.clone() if torch.is_tensor(step) else step
            for key, value in state.items():
                migrated.setdefault(key, value)

            new_optimizer.state[p_new] = migrated

    return new_optimizer
