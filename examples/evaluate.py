"""Scores a trained .ply against a COLMAP scene's held-out views.

Reports PSNR and SSIM on every held-out view (and, with --train, on the
training views too, so the generalisation gap is visible). Optionally
applies metalsplat.cleanup first, to check what the post-training passes
actually cost or buy.

Usage:
    uv run python examples/evaluate.py
    uv run python examples/evaluate.py --ply examples/garden.ply --train
    uv run python examples/evaluate.py --sh-damp 0.75 --prune
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from metalsplat import render
from metalsplat.cleanup import damp_view_dependence, prune_isolated
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.export import load_ply
from metalsplat.losses import ssim

DEVICE = "mps"
DEFAULT_DATA = Path(__file__).parent.parent / "data" / "garden"
DEFAULT_PLY = Path(__file__).parent / "garden.ply"
HOLDOUT_STRIDE = 8


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred - target).pow(2).mean().item()
    return float("inf") if mse <= 0 else -10.0 * torch.log10(torch.tensor(mse)).item()


def score(model, scene, indices: list[int], background: torch.Tensor) -> tuple[float, float]:
    psnrs, ssims = [], []
    with torch.no_grad():
        for i in indices:
            pred = render(model, scene.cameras[i], background=background)
            torch.mps.synchronize()
            psnrs.append(psnr(pred, scene.images[i]))
            ssims.append(ssim(pred.clamp(0, 1), scene.images[i]).item())
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train", action="store_true", help="also score training views")
    parser.add_argument("--prune", action="store_true", help="apply isolation pruning first")
    parser.add_argument("--sh-damp", type=float, default=1.0, help="scale non-DC SH before scoring")
    args = parser.parse_args()

    model = load_ply(args.ply, device=DEVICE)
    print(f"{args.ply.name}: {model.num_points} gaussians, sh_degree={model.sh_degree}", flush=True)

    if args.prune:
        model, n_pruned = prune_isolated(model)
        print(f"pruned {n_pruned} isolated -> {model.num_points}", flush=True)
    if args.sh_damp != 1.0:
        model = damp_view_dependence(model, args.sh_damp)
        print(f"damped non-DC SH by {args.sh_damp}", flush=True)

    scene = load_colmap_scene(args.data, device=DEVICE)
    n = len(scene.cameras)
    held_out = list(range(0, n, HOLDOUT_STRIDE))
    print(f"scoring {len(held_out)} held-out views...", flush=True)

    background = torch.zeros(3, device=DEVICE)
    p, s = score(model, scene, held_out, background)
    print(f"  held-out : PSNR {p:.2f}  SSIM {s:.4f}", flush=True)

    if args.train:
        train = [i for i in range(n) if i % HOLDOUT_STRIDE != 0][: len(held_out)]
        tp, ts = score(model, scene, train, background)
        print(f"  train    : PSNR {tp:.2f}  SSIM {ts:.4f}", flush=True)
        print(f"  gap      : PSNR {tp - p:+.2f}  SSIM {ts - s:+.4f}", flush=True)


if __name__ == "__main__":
    main()
