# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tchauffi/metalsplat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tchauffi/metalsplat/releases/tag/v0.1.0
