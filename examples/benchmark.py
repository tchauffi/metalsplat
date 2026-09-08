"""Benchmarks the MetalSplat pipeline on a real trained scene.

Reports a per-stage breakdown (projection, tile binning + sort,
rasterization; forward and backward), end-to-end render throughput, and how
both scale with gaussian count and image resolution.

Every measurement calls torch.mps.synchronize() around the timed region --
MPS dispatches asynchronously, so without it you time the enqueue, not the
work. Each configuration is warmed up first (the first call to a kernel
pays runtime MSL compilation) and reported as a median over repeats.

Usage:
    uv run python examples/benchmark.py
    uv run python examples/benchmark.py --ply examples/garden.ply --repeats 30
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from metalsplat import Camera, GaussianModel, render
from metalsplat.data.colmap import load_colmap_scene
from metalsplat.export import load_ply
from metalsplat.ops.project import project_gaussians
from metalsplat.ops.rasterize import rasterize_gaussians
from metalsplat.ops.tiling import bin_and_sort_gaussians

DEVICE = "mps"
DATA_ROOT = Path(__file__).parent.parent / "data" / "garden"
DEFAULT_PLY = Path(__file__).parent / "garden.ply"


def timeit(fn, repeats: int, warmup: int = 3) -> tuple[float, float]:
    """Median and min wall time in ms, synchronising around each call."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples)


def scaled_camera(cam: Camera, s: float) -> Camera:
    return Camera(
        R_wc=cam.R_wc, t_wc=cam.t_wc,
        fx=cam.fx * s, fy=cam.fy * s, cx=cam.cx * s, cy=cam.cy * s,
        img_width=int(round(cam.img_width * s)), img_height=int(round(cam.img_height * s)),
    ).to(DEVICE)


def subsample(model: GaussianModel, n: int) -> GaussianModel:
    if n >= model.num_points:
        return model
    idx = torch.randperm(model.num_points, device=model.means.device)[:n]
    kw = {
        'scales': model.scales.detach()[idx], 'quats': model.quats.detach()[idx],
        'opacities': model.opacities.detach()[idx],
    }
    if model.sh_degree == 0:
        kw["colors"] = model.colors.detach()[idx]
    else:
        kw["sh_degree"] = model.sh_degree
        kw["sh_coeffs"] = model.raw_sh.detach()[idx]
    return GaussianModel(model.means.detach()[idx], **kw).to(DEVICE)


def stage_breakdown(model: GaussianModel, cam: Camera, repeats: int) -> None:
    """Times each pipeline stage in isolation, forward and backward."""
    bg = torch.zeros(3, device=DEVICE)
    args = (cam.R_wc, cam.t_wc, cam.fx, cam.fy, cam.cx, cam.cy, cam.img_width, cam.img_height)

    means = model.means.detach().clone().requires_grad_()
    scales = model.scales.detach().clone().requires_grad_()
    quats = model.quats.detach().clone().requires_grad_()

    def project_fwd():
        with torch.no_grad():
            project_gaussians(means, scales, quats, *args)

    p_fwd, _ = timeit(project_fwd, repeats)

    def project_bwd():
        m2d, _, conics, _, _, _ = project_gaussians(means, scales, quats, *args)
        (m2d.sum() + conics.sum()).backward()

    p_both, _ = timeit(project_bwd, repeats)

    with torch.no_grad():
        means2d, depths, conics, radii, valid, _ = project_gaussians(means, scales, quats, *args)
        torch.mps.synchronize()

    def binning():
        bin_and_sort_gaussians(means2d, depths, radii, valid, cam.img_width, cam.img_height)

    b_ms, _ = timeit(binning, repeats)

    colors = model.colors_from_view(
        torch.nn.functional.normalize(model.means - cam.position, dim=-1)
    ) if model.sh_degree else model.colors
    colors_d = colors.detach()
    opac_d = model.opacities.detach()

    def raster_fwd():
        with torch.no_grad():
            rasterize_gaussians(
                means2d, depths, conics, opac_d, colors_d, radii, valid,
                cam.img_width, cam.img_height, background=bg,
            )

    r_fwd, _ = timeit(raster_fwd, repeats)

    m2d_g = means2d.detach().clone().requires_grad_()
    con_g = conics.detach().clone().requires_grad_()
    op_g = opac_d.clone().requires_grad_()
    col_g = colors_d.clone().requires_grad_()

    def raster_bwd():
        img = rasterize_gaussians(
            m2d_g, depths, con_g, op_g, col_g, radii, valid,
            cam.img_width, cam.img_height, background=bg,
        )
        img.sum().backward()

    r_both, _ = timeit(raster_bwd, repeats)

    def full_fwd():
        with torch.no_grad():
            render(model, cam, background=bg)

    f_ms, _ = timeit(full_fwd, repeats)

    n_pairs = (bin_and_sort_gaussians(
        means2d, depths, radii, valid, cam.img_width, cam.img_height
    ).sorted_gaussian_ids.numel())
    torch.mps.synchronize()

    print(f"  visible gaussians   : {int((valid > 0).sum())} / {model.num_points}")
    print(f"  (gaussian, tile) pairs: {n_pairs}")
    print()
    print(f"  {'stage':<26}{'forward':>10}{'fwd+bwd':>10}{'backward':>10}")
    print(f"  {'-' * 56}")
    print(f"  {'project':<26}{p_fwd:>9.2f}ms{p_both:>9.2f}ms{p_both - p_fwd:>9.2f}ms")
    print(f"  {'tile bin + sort':<26}{b_ms:>9.2f}ms{'--':>10}{'--':>10}")
    print(f"  {'rasterize':<26}{r_fwd:>9.2f}ms{r_both:>9.2f}ms{r_both - r_fwd:>9.2f}ms")
    print(f"  {'-' * 56}")
    print(f"  {'render() end-to-end':<26}{f_ms:>9.2f}ms  ({1000 / f_ms:.1f} fps)")


def sweep_points(model: GaussianModel, cam: Camera, repeats: int) -> None:
    bg = torch.zeros(3, device=DEVICE)
    print(f"  {'gaussians':>12}{'forward':>12}{'fps':>9}{'per 100k':>11}")
    print(f"  {'-' * 44}")
    for n in (50_000, 100_000, 200_000, 400_000, model.num_points):
        if n > model.num_points:
            continue
        sub = subsample(model, n)
        ms, _ = timeit(lambda m=sub: render(m, cam, background=bg), repeats)
        print(f"  {n:>12,}{ms:>11.2f}ms{1000 / ms:>9.1f}{ms / (n / 100_000):>10.2f}ms")
        del sub


def sweep_resolution(model: GaussianModel, cam: Camera, repeats: int) -> None:
    bg = torch.zeros(3, device=DEVICE)
    print(f"  {'resolution':>14}{'pixels':>12}{'forward':>12}{'fps':>9}{'per Mpix':>11}")
    print(f"  {'-' * 58}")
    for s in (0.25, 0.5, 0.75, 1.0):
        c = scaled_camera(cam, s)
        mpix = c.img_width * c.img_height / 1e6
        ms, _ = timeit(lambda cc=c: render(model, cc, background=bg), repeats)
        print(
            f"  {f'{c.img_width}x{c.img_height}':>14}{c.img_width * c.img_height:>12,}"
            f"{ms:>11.2f}ms{1000 / ms:>9.1f}{ms / mpix:>10.2f}ms"
        )


def training_step(model: GaussianModel, cam: Camera, target: torch.Tensor, repeats: int) -> None:
    from metalsplat.losses import gaussian_splatting_loss

    bg = torch.zeros(3, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        img = render(model, cam, background=bg)
        gaussian_splatting_loss(img, target).backward()
        opt.step()

    ms, best = timeit(step, repeats)
    print("  full training step (render + L1/D-SSIM loss + backward + Adam)")
    print(f"    median {ms:.1f}ms   best {best:.1f}ms   -> {1000 / ms:.1f} steps/s, "
          f"{5000 * ms / 1000 / 60:.1f} min per 5000 iterations")


def memory_report(model: GaussianModel, cam: Camera, images: list | None = None) -> None:
    """Measures what a render and a training step actually cost in memory.

    Apple Silicon has unified memory, so "VRAM" is just system RAM the GPU
    driver has taken. torch.mps.current_allocated_memory() is what tensors
    hold; driver_allocated_memory() includes the allocator's cached blocks,
    which is what shows up as memory pressure.
    """
    bg = torch.zeros(3, device=DEVICE)
    n = model.num_points

    # Per-gaussian parameter cost, from the model's own tensors.
    params = {
        "means (N,3)": model.means,
        "raw_scales (N,3)": model.raw_scales,
        "raw_quats (N,4)": model.raw_quats,
        "raw_opacities (N,)": model.raw_opacities,
    }
    params["raw_sh (N,9,3)" if model.sh_degree else "raw_colors (N,3)"] = (
        model.raw_sh if model.sh_degree else model.raw_colors
    )
    total_param_bytes = sum(t.numel() * t.element_size() for t in params.values())

    print(f"  {'parameter':<22}{'bytes/gaussian':>16}{'total':>12}")
    print(f"  {'-' * 50}")
    for name, t in params.items():
        b = t.numel() * t.element_size()
        print(f"  {name:<22}{b / n:>15.0f}B{b / 1e6:>11.1f}MB")
    print(f"  {'-' * 50}")
    print(f"  {'model total':<22}{total_param_bytes / n:>15.0f}B{total_param_bytes / 1e6:>11.1f}MB")

    base = torch.mps.current_allocated_memory()
    with torch.no_grad():
        render(model, cam, background=bg)
        torch.mps.synchronize()
    after_fwd = torch.mps.current_allocated_memory()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    img = render(model, cam, background=bg)
    img.sum().backward()
    opt.step()
    torch.mps.synchronize()
    after_train = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()

    with torch.no_grad():
        means2d, depths, _, radii, valid, _ = project_gaussians(
            model.means, model.scales, model.quats, cam.R_wc, cam.t_wc,
            cam.fx, cam.fy, cam.cx, cam.cy, cam.img_width, cam.img_height,
        )
        n_pairs = bin_and_sort_gaussians(
            means2d, depths, radii, valid, cam.img_width, cam.img_height
        ).sorted_gaussian_ids.numel()
        n_visible = int((valid > 0).sum())
        torch.mps.synchronize()

    print()
    if images:
        img_bytes = sum(t.numel() * t.element_size() for t in images)
        print(f"  {f'training images ({len(images)}, {images[0].dtype})':<40}{img_bytes / 1e6:>9.1f}MB")
        print(f"  {'  ... the same images as uint8':<40}{img_bytes / 4 / 1e6:>9.1f}MB")
    print(f"  {'model parameters':<40}{total_param_bytes / 1e6:>9.1f}MB")
    print(f"  {'+ Adam state (2 moments per param)':<40}{2 * total_param_bytes / 1e6:>9.1f}MB")
    print(f"  {'live tensors, idle':<40}{base / 1e6:>9.1f}MB")
    print(f"  {'live tensors, after forward render':<40}{after_fwd / 1e6:>9.1f}MB")
    print(f"  {'live tensors, after a training step':<40}{after_train / 1e6:>9.1f}MB")
    print(f"  {'driver allocation (incl. cache)':<40}{driver / 1e6:>9.1f}MB")
    print()
    print(f"  tile binning expands {n:,} gaussians to {n_pairs:,} (gaussian, tile) pairs")
    print(f"  at {cam.img_width}x{cam.img_height}: {n_pairs * 8 * 2 / 1e6:.1f}MB for the sort keys and ids alone")
    print(f"  ({n_pairs / max(n_visible, 1):.1f} tiles touched per visible gaussian on average)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    if not args.ply.exists():
        raise FileNotFoundError(f"No trained scene at {args.ply}; run train_garden.py first.")

    model = load_ply(args.ply, device=DEVICE)
    scene = load_colmap_scene(DATA_ROOT, device=DEVICE)
    cam = scene.cameras[0]

    print(f"scene   : {args.ply.name}, {model.num_points:,} gaussians, sh_degree={model.sh_degree}")
    print(f"camera  : {cam.img_width}x{cam.img_height}")
    print(f"repeats : {args.repeats} (median reported, 3 warmup calls discarded)")

    print("\n== stage breakdown (full resolution, all gaussians) ==")
    stage_breakdown(model, cam, args.repeats)

    print("\n== scaling with gaussian count (full resolution) ==")
    sweep_points(model, cam, args.repeats)

    print("\n== scaling with resolution (all gaussians) ==")
    sweep_resolution(model, cam, args.repeats)

    print("\n== training throughput (full resolution, as train_garden.py uses) ==")
    training_step(model, cam, scene.images[0], args.repeats)

    print("\n== memory ==")
    memory_report(model, cam, scene.images)


if __name__ == "__main__":
    main()
