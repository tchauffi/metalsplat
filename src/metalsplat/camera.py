"""Minimal pinhole camera used by rendering.render()."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    R_wc: torch.Tensor  # (3, 3) world-to-camera rotation
    t_wc: torch.Tensor  # (3,) world-to-camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    img_width: int
    img_height: int

    @property
    def position(self) -> torch.Tensor:
        """World-space camera center, i.e. the inverse of `t_wc = -R_wc @ position`."""
        return -self.R_wc.T @ self.t_wc

    def to(self, device) -> Camera:
        return Camera(
            R_wc=self.R_wc.to(device),
            t_wc=self.t_wc.to(device),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            img_width=self.img_width,
            img_height=self.img_height,
        )

    @staticmethod
    def look_at(
        eye: torch.Tensor,  # (3,)
        target: torch.Tensor,  # (3,)
        up: torch.Tensor,  # (3,)
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_width: int,
        img_height: int,
    ) -> Camera:
        """Builds a camera looking from `eye` toward `target`.

        `up` is the world-space up direction; it only needs to be roughly
        right, it gets orthogonalised against the view direction.
        """
        forward = target - eye
        forward = forward / forward.norm()
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        # Camera +y is image-*down* (v = fy*y/z + cy, and v grows downward),
        # so the middle row is the down direction, not the up one. It also
        # has to be cross(forward, right), not cross(right, forward): the
        # basis must satisfy right x down = forward to be a rotation
        # (det +1) rather than a reflection (det -1).
        down = torch.linalg.cross(forward, right)

        # Rows are the camera axes expressed in world space, so R_wc
        # (world -> camera) is exactly this stacked matrix.
        R_wc = torch.stack([right, down, forward], dim=0)
        t_wc = -R_wc @ eye
        return Camera(
            R_wc=R_wc,
            t_wc=t_wc,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_width=img_width,
            img_height=img_height,
        )

    @staticmethod
    def identity(
        fx: float, fy: float, cx: float, cy: float, img_width: int, img_height: int
    ) -> Camera:
        return Camera(
            R_wc=torch.eye(3),
            t_wc=torch.zeros(3),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_width=img_width,
            img_height=img_height,
        )
