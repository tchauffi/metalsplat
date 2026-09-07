# MetalSplat

A differentiable 3D Gaussian Splatting rasterizer for PyTorch, targeting
Apple Silicon via Metal instead of CUDA -- a `gsplat`-equivalent that runs
entirely on the MPS backend, with real hand-written Metal compute kernels
for the performance-critical stages.

## Requirements

- macOS on Apple Silicon (uses the `mps` PyTorch backend).
- `torch.mps.compile_shader` (present in this project's pinned torch
  version). Kernels are Metal Shading Language source compiled at *runtime*
  via PyTorch's built-in Metal compiler -- no Xcode command-line tools or
  `xcrun metal` toolchain required, no native build step.

```bash
uv sync
uv run pytest
uv run python examples/fit_image.py
```

## Usage

```python
import torch
from metalsplat import Camera, GaussianModel, render

model = GaussianModel.random(n=5000, bound=1.0, device="mps")
camera = Camera.identity(fx=128, fy=128, cx=64, cy=64, img_width=128, img_height=128).to("mps")

image = render(model, camera)  # (H, W, 3), differentiable
loss = image.pow(2).mean()
loss.backward()  # gradients flow to model.means, .raw_scales, .raw_quats, .raw_opacities, .raw_colors
```

See `examples/fit_image.py` for a full training loop (Adam-optimizing
gaussians to reproduce a target image).

### Real scenes (COLMAP)

`metalsplat/data/colmap.py` loads a standard COLMAP project
(`<scene_root>/sparse/0/{cameras,images,points3D}.bin` +
`<scene_root>/images/`, via `pycolmap`) into `Camera` objects, images, and
an initial point cloud:

```python
from metalsplat.data.colmap import load_colmap_scene

scene = load_colmap_scene("data/garden", device="mps")
# scene.cameras: list[Camera], scene.images: list[Tensor], scene.points/.colors for init
```

Intrinsics are automatically rescaled if the images on disk are
downsampled relative to COLMAP's calibration resolution. Only undistorted
camera models (PINHOLE, SIMPLE_PINHOLE) are supported. See
`examples/train_garden.py` for a full training script against a real
scene: random training view per step, Adam, periodic loss/PSNR logging
against held-out eval views.

```bash
uv run python examples/train_garden.py
```

Training uses adaptive density control (`metalsplat/densify.py`): every
`DENSIFY_INTERVAL` steps, gaussians with high accumulated screen-space
gradient are split (if already large -- over-reconstruction) or cloned (if
still small -- under-reconstruction), and low-opacity gaussians are
pruned. Without this, a fixed gaussian count can't add detail where
reconstruction is poor or drop gaussians that have become useless;
overloaded gaussians compensate by growing/recoloring in unstable ways
instead, which shows up as training-time color drift and falling held-out
PSNR (observed empirically on this scene before densification was added).
The means learning rate is also decayed exponentially over training
(`lr_means_init -> lr_means_final`), matching standard 3DGS practice, for
the same reason -- positions should make large exploratory moves early and
settle down for fine detail late, not keep taking large steps throughout.

Set `SH_DEGREE = 2` at the top of `examples/train_garden.py` to train
view-dependent color (see below) instead of the default plain RGB.

The training loss (`metalsplat/losses.py`) is the standard 3DGS objective:
`(1 - lambda) * L1 + lambda * D-SSIM`, `D-SSIM = (1 - SSIM) / 2`, with the
paper's default `lambda = 0.2` (`LAMBDA_DSSIM` in the script). SSIM is a
local-window statistic computed via `conv2d` -- no custom Metal kernel
needed for it, plain torch ops already run fine on MPS here.

## Architecture

Pipeline: `GaussianModel + Camera -> project -> tile-bin/sort -> rasterize -> image`,
mirroring gsplat's own architecture with Metal kernels standing in for its
CUDA kernels.

- **`metalsplat/ops/project.py`** (Metal kernel, `kernels/project.metal`):
  projects 3D gaussians to 2D screen space -- camera-space transform, 3D
  covariance from scale+quaternion, the EWA/affine perspective
  approximation, and the resulting 2D conic (inverse covariance) and pixel
  radius. Forward and backward are both hand-written MSL kernels.
- **`metalsplat/ops/tiling.py`** (plain torch/MPS ops, no kernel): bins
  projected gaussians into 16x16-pixel tiles and sorts them by (tile,
  depth) via a single packed sort key -- gsplat itself doesn't use a
  custom kernel for this stage either, relying on a generic sort.
- **`metalsplat/ops/rasterize.py`** (Metal kernel, `kernels/rasterize.metal`):
  tile-based front-to-back alpha compositing. One thread per pixel, one
  threadgroup per tile. Backward accumulates per-gaussian gradients via
  Metal atomics (many pixels write to the same gaussian).
- **`metalsplat/ops/sh.py`** (Metal kernel, `kernels/sh.metal`): optional
  degree<=2 spherical-harmonics view-dependent color (`GaussianModel(...,
  sh_degree=2)`). Evaluates the SH basis given each gaussian's raw
  coefficients and its unit view direction to the camera (`Camera.position`
  gives the world-space camera center); the view-direction normalize
  itself is left to plain PyTorch autograd (cheap, no need for a custom
  kernel), so the Metal kernel's backward only needs to produce
  `d_sh_coeffs` and `d_dirs` -- gradient flows from there through the
  normalize op back into `model.means` automatically. Uses the standard
  3DGS color convention (`color = eval_sh(...) + 0.5`, unconstrained/
  unclamped internally) rather than the plain-RGB path's sigmoid.

Every kernel-backed stage has a pure-PyTorch reference implementation
(`metalsplat/reference/`) that is the executable spec and the numerical
oracle the kernels are tested against (`tests/test_project.py`,
`tests/test_rasterize.py`) -- both forward values and backward gradients,
via `torch.autograd` on the reference vs. the kernel's hand-written
backward. It also works as a CPU-compatible fallback.

## Roadmap

Deliberately out of scope for this pass:

- SH degree 3 (only degree<=2, 9 coefficients/channel, is implemented; the
  kernel pattern extends directly, just more basis-function terms).
- Exact anti-aliasing compensation factor (currently a small `eps * I`
  regularizer on the 2D covariance for numerical stability instead).
- Opacity reset (3DGS periodically resets opacity during training to clear
  out stale/occluding gaussians; not implemented here).
- Multi-camera batching.
- Depth/alpha auxiliary render outputs and depth supervision.
- Camera lens distortion (only undistorted PINHOLE/SIMPLE_PINHOLE COLMAP
  cameras are supported).
