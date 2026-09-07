"""Quaternion helpers shared by the pure-PyTorch reference and (mirrored in
MSL) the Metal kernels. Quaternion layout is ``(w, x, y, z)``.
"""

from __future__ import annotations

import torch


def normalize_quat(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(..., 4) raw quaternion -> (..., 4) unit quaternion."""
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """(..., 4) unit quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix."""
    w, x, y, z = quat.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
