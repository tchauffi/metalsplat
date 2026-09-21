# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-21

### Added

- 2D Gaussian Splatting pipeline (Huang et al. 2024): `Gaussian2DModel`,
  `render_2dgs`, and hand-written Metal kernels for projection and
  rasterization, parallel to the existing 3DGS pipeline.
- Exact per-pixel ray-splat intersection (no EWA/affine approximation),
  yielding genuinely differentiable depth and normal outputs.
- Depth distortion and normal-consistency regularizers
  (`distortion_loss`, `normal_consistency_loss`), gated to activate
  partway through training.
- AbsGS-style densification and adaptive density control for
  `Gaussian2DModel` (`densify_and_prune_2dgs`, `prune_low_opacity_2dgs`).
- Loss-driven gaussian seeding for regions with little or no initial
  coverage.
- `.ply` export/import for `Gaussian2DModel`, matching the official 2DGS
  reference implementation's layout, plus viewer-compatible synthetic
  `scale_2` output.
- Real-scene training example (`examples/train_garden_2dgs.py`) using the
  paper's own hyperparameters, and an image-fitting smoke test
  (`examples/fit_image_2dgs.py`).
- Orbit video rendering with RGB and surface-normal panes
  (`examples/render_video_2dgs.py`).
- Pure-PyTorch reference implementations for every new kernel-backed
  stage, used as the numerical oracle in tests and as a CPU-compatible
  fallback.

## [0.1.0] - 2026-09-09

Initial release.

### Added

- Differentiable 3D Gaussian Splatting rasterizer for PyTorch, running on
  Apple Silicon via the MPS backend with hand-written Metal compute kernels
  (project, tiling, rasterize, spherical harmonics).
- `GaussianModel` / `Camera` / `render` core API, with forward and backward
  passes for the projection and rasterization stages.
- Depth pass alongside the RGB render, and AbsGS-style absolute
  screen-space gradients for densification.
- Adaptive density control: gradient-driven split/clone, opacity reset,
  and low-opacity pruning.
- Loss-driven gaussian seeding for regions with little or no initial
  coverage.
- Spherical harmonics up to degree 2 for view-dependent color, plus a
  3D smoothing (Mip-Splatting) filter and a frustum-clamped EWA Jacobian.
- `SparseAdam` optimizer, visibility-aware to speed up training on
  partially-visible gaussians.
- COLMAP scene loading (`metalsplat.data.colmap`) via `pycolmap`.
- `.ply` export/import and post-training cleanup utilities.
- Pure-PyTorch reference implementations for every kernel-backed stage,
  used as the numerical oracle in tests and as a CPU-compatible fallback.
- Example scripts: image fitting, real-scene training, orbit video
  rendering, evaluation, and a pipeline benchmark.
- GitHub Actions CI running the test suite on macOS.
- Ruff-based linting/formatting with pre-commit hooks.
- MIT license, PyPI classifiers, keywords, and project URLs.

[Unreleased]: https://github.com/tchauffi/metalsplat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tchauffi/metalsplat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tchauffi/metalsplat/releases/tag/v0.1.0
