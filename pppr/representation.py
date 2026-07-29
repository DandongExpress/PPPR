"""PPPR Gaussian primitive parameterization (Sec. 4.1.2, Eqs. 4–5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import JOINT_NAMES, JOINT_SCALES, NUM_JOINTS, t_pose_positions


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix (footnote 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


class JointGaussian(nn.Module):
    """Single-joint Gaussian primitive Θ_j = {p, s, q, v, β, ω}."""

    def __init__(
        self,
        joint_idx: int,
        position: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        n_doppler: int = 8,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.joint_idx = joint_idx
        self.joint_name = JOINT_NAMES[joint_idx]
        init_scale = scale if scale is not None else JOINT_SCALES[self.joint_name]

        self.position = nn.Parameter(position.float().clone())
        self.scale = nn.Parameter(torch.full((3,), init_scale, dtype=torch.float32))
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        if velocity is None:
            velocity = torch.zeros(3)
        self.velocity = nn.Parameter(velocity.float().clone())
        self.beta = nn.Parameter(torch.tensor(0.0))  # opacity logit; α = σ(β)
        self.omega = nn.Parameter(torch.zeros(n_doppler))

    def get_rotation_matrix(self) -> torch.Tensor:
        return quaternion_to_rotation_matrix(self.rotation)

    def get_covariance(self) -> torch.Tensor:
        """Σ = R S S^T R^T with S = diag(s) (Sec. 4.1.2)."""
        s = self.scale.abs().clamp(min=1e-4)
        S = torch.diag(s)
        R = self.get_rotation_matrix()
        return R @ S @ S.T @ R.T

    def get_opacity(self) -> torch.Tensor:
        """α_j = σ(β_j) — normalized RCS (Eq. 5)."""
        return torch.sigmoid(self.beta)

    def gaussian_density(self, x: torch.Tensor) -> torch.Tensor:
        """G_j(x) in Eq. 4. x: [..., 3]."""
        cov = self.get_covariance()
        diff = x - self.position
        # Use Cholesky for numerical stability
        eps = 1e-6 * torch.eye(3, device=cov.device, dtype=cov.dtype)
        cov = cov + eps
        try:
            L = torch.linalg.cholesky(cov)
            y = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            quad = (diff * y).sum(dim=-1)
            log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
        except RuntimeError:
            inv = torch.linalg.pinv(cov)
            quad = torch.einsum("...i,ij,...j->...", diff, inv, diff)
            log_det = torch.logdet(cov.clamp(min=1e-8))
        norm = -0.5 * (3 * torch.log(torch.tensor(2.0 * torch.pi, device=x.device)) + log_det)
        return torch.exp(norm - 0.5 * quad)

    def complex_return_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Phase term exp(i ω^T [v ⊙ (x - p)]) from Eq. 5 (returns complex)."""
        diff = x - self.position
        n = min(self.omega.numel(), 3)
        feat = self.velocity[:n] * diff[..., :n]
        # Pad / truncate omega to match
        omega = self.omega[:n]
        phase = (omega * feat).sum(dim=-1)
        return torch.exp(1j * phase)

    def forward(self) -> Dict[str, torch.Tensor]:
        return {
            "position": self.position,
            "scale": self.scale.abs(),
            "rotation": F.normalize(self.rotation, dim=-1),
            "velocity": self.velocity,
            "beta": self.beta,
            "opacity": self.get_opacity(),
            "omega": self.omega,
            "covariance": self.get_covariance(),
        }


class PPPRInstance(nn.Module):
    """Full-body PPPR for one person: N_j joint Gaussians."""

    def __init__(
        self,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        n_doppler: int = 8,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if positions is None:
            positions = t_pose_positions()
        if velocities is None:
            velocities = torch.zeros_like(positions)
        assert positions.shape == (NUM_JOINTS, 3)
        joints = []
        for i in range(NUM_JOINTS):
            joints.append(
                JointGaussian(
                    joint_idx=i,
                    position=positions[i],
                    velocity=velocities[i],
                    n_doppler=n_doppler,
                )
            )
        self.joints = nn.ModuleList(joints)
        if device is not None:
            self.to(device)

    @property
    def positions(self) -> torch.Tensor:
        return torch.stack([j.position for j in self.joints], dim=0)

    @property
    def velocities(self) -> torch.Tensor:
        return torch.stack([j.velocity for j in self.joints], dim=0)

    def centroid(self) -> torch.Tensor:
        return self.positions.mean(dim=0)

    def parameters_vector(self) -> torch.Tensor:
        """Flatten all Θ into a single vector for HPE model input."""
        return pack_pppr(self)


def pack_pppr(instance: PPPRInstance) -> torch.Tensor:
    """Pack one person into a flat feature tensor [N_j * D].

    Layout per joint: p(3), s(3), q(4), v(3), β(1), ω(8) → 22 dims.
    """
    feats = []
    for j in instance.joints:
        feats.append(
            torch.cat(
                [
                    j.position,
                    j.scale.abs(),
                    F.normalize(j.rotation, dim=-1),
                    j.velocity,
                    j.beta.view(1),
                    j.omega,
                ],
                dim=0,
            )
        )
    return torch.cat(feats, dim=0)


def unpack_pppr(vec: torch.Tensor, n_doppler: int = 8) -> PPPRInstance:
    """Inverse of pack_pppr."""
    d = 3 + 3 + 4 + 3 + 1 + n_doppler
    assert vec.numel() == NUM_JOINTS * d
    positions = []
    velocities = []
    scales = []
    rotations = []
    betas = []
    omegas = []
    flat = vec.view(NUM_JOINTS, d)
    for i in range(NUM_JOINTS):
        row = flat[i]
        positions.append(row[0:3])
        scales.append(row[3:6])
        rotations.append(row[6:10])
        velocities.append(row[10:13])
        betas.append(row[13:14])
        omegas.append(row[14 : 14 + n_doppler])
    inst = PPPRInstance(
        positions=torch.stack(positions),
        velocities=torch.stack(velocities),
        n_doppler=n_doppler,
    )
    with torch.no_grad():
        for i, j in enumerate(inst.joints):
            j.scale.copy_(scales[i])
            j.rotation.copy_(rotations[i])
            j.beta.copy_(betas[i].squeeze())
            j.omega.copy_(omegas[i])
    return inst


def mean_pool_pppr_features(vec: torch.Tensor, n_doppler: int = 8) -> torch.Tensor:
    """Appendix A.6: concatenate per-joint params then mean-pool → dim 48.

    Uses a reduced 48-D descriptor: mean of [p(3),s(3),v(3),α(1),ω_mean(1)]
    expanded with std stats to reach 48, matching the MLP+PPPR baseline.
    """
    d = 3 + 3 + 4 + 3 + 1 + n_doppler
    joints = vec.view(NUM_JOINTS, d)
    p = joints[:, 0:3]
    s = joints[:, 3:6]
    v = joints[:, 10:13]
    alpha = torch.sigmoid(joints[:, 13:14])
    omega = joints[:, 14:]
    parts = [
        p.mean(0),
        p.std(0),
        s.mean(0),
        s.std(0),
        v.mean(0),
        v.std(0),
        alpha.mean(0),
        alpha.std(0),
        omega.mean(0)[:8],
        omega.std(0)[:8],
        p.max(0).values,
        p.min(0).values,
    ]
    feat = torch.cat([t.reshape(-1) for t in parts], dim=0)
    # Pad / trim to 48
    if feat.numel() < 48:
        feat = F.pad(feat, (0, 48 - feat.numel()))
    return feat[:48]
