"""Dual physics-informed objectives (Sec. 4.3–4.4, Eqs. 12–20)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .config import OptimConfig
from .radar_simulation import match_heatmap_shape
from .representation import PPPRInstance
from .skeleton import (
    JOINT_ANGLE_TRIPLETS,
    SKELETON_CONNECTIONS,
    SKELETON_LENGTHS,
)


def bone_length_loss(instance: PPPRInstance) -> torch.Tensor:
    """L_bone = Σ_{(m,n)∈E} (||p_m - p_n|| - ℓ_mn)  (Eq. 12)."""
    loss = instance.positions.new_zeros(())
    pos = instance.positions
    for m, n in SKELETON_CONNECTIONS:
        key = (min(m, n), max(m, n))
        ell = SKELETON_LENGTHS.get(key, SKELETON_LENGTHS.get((m, n), 0.3))
        dist = torch.norm(pos[m] - pos[n])
        loss = loss + (dist - ell)
    return loss


def rigid_bone_loss(instance: PPPRInstance) -> torch.Tensor:
    """L_rigid = Σ ||(v_m - v_n) · b̂_mn||  (Eq. 13)."""
    loss = instance.positions.new_zeros(())
    pos = instance.positions
    vel = instance.velocities
    for m, n in SKELETON_CONNECTIONS:
        bone = pos[m] - pos[n]
        bone_hat = bone / (torch.norm(bone) + 1e-6)
        loss = loss + torch.abs(torch.dot(vel[m] - vel[n], bone_hat))
    return loss


def joint_angle_loss(instance: PPPRInstance, theta_max: float) -> torch.Tensor:
    """L_joint = Σ max(0, |θ_mno| - θ_max)^2  (Eq. 14)."""
    loss = instance.positions.new_zeros(())
    pos = instance.positions
    for m, n, o in JOINT_ANGLE_TRIPLETS:
        v1 = pos[m] - pos[n]
        v2 = pos[o] - pos[n]
        v1n = F.normalize(v1, dim=0)
        v2n = F.normalize(v2, dim=0)
        cos_a = torch.clamp(torch.dot(v1n, v2n), -1.0, 1.0)
        theta = torch.acos(cos_a)
        loss = loss + F.relu(torch.abs(theta) - theta_max) ** 2
    return loss


def kinematic_loss(instance: PPPRInstance, cfg: OptimConfig) -> torch.Tensor:
    """L_kine = L_bone + L_rigid + L_joint  (Eq. 15)."""
    theta_max = torch.deg2rad(torch.tensor(cfg.theta_max_deg, device=instance.positions.device))
    return (
        bone_length_loss(instance)
        + rigid_bone_loss(instance)
        + joint_angle_loss(instance, float(theta_max))
    )


def percentile_mask(heatmap: torch.Tensor, tau_pct: float) -> torch.Tensor:
    """Binary mask B = {H > percentile(H, 100 - τ_pct)} (footnote 4)."""
    flat = heatmap.reshape(-1)
    q = torch.quantile(flat, 1.0 - tau_pct / 100.0)
    return (heatmap > q).float()


def electromagnetic_loss(
    h_sim: torch.Tensor,
    h_ori: torch.Tensor,
    tau_pct: float = 10.0,
) -> torch.Tensor:
    """L_EM = 1 - IoU(B_sim, B_ori)  (Eq. 16)."""
    h_sim = match_heatmap_shape(h_sim, h_ori)
    # Ensure comparable dynamic range
    h_sim_n = h_sim / (h_sim.max().clamp(min=1e-8))
    h_ori_n = h_ori / (h_ori.max().clamp(min=1e-8))
    b_sim = percentile_mask(h_sim_n, tau_pct)
    b_ori = percentile_mask(h_ori_n, tau_pct)
    inter = (b_sim * b_ori).sum()
    union = ((b_sim + b_ori) > 0).float().sum().clamp(min=1.0)
    iou = inter / union
    return 1.0 - iou


def centroid_separation_loss(
    instances: Sequence[PPPRInstance],
    d_sep: float,
) -> torch.Tensor:
    """L_sep (Eq. 18)."""
    if len(instances) < 2:
        return instances[0].positions.new_zeros(())
    loss = instances[0].positions.new_zeros(())
    cents = [inst.centroid() for inst in instances]
    for s in range(len(cents)):
        for t in range(s + 1, len(cents)):
            dist = torch.norm(cents[s] - cents[t])
            loss = loss + F.relu(d_sep - dist)
    return loss


def joint_collision_loss(
    instances: Sequence[PPPRInstance],
    d_joint: float,
) -> torch.Tensor:
    """L_coll (Eq. 19)."""
    if len(instances) < 2:
        return instances[0].positions.new_zeros(())
    loss = instances[0].positions.new_zeros(())
    for s in range(len(instances)):
        for t in range(s + 1, len(instances)):
            ps = instances[s].positions
            pt = instances[t].positions
            # Pairwise distances [Nj, Nj]
            diff = ps[:, None, :] - pt[None, :, :]
            dists = torch.norm(diff, dim=-1)
            loss = loss + F.relu(d_joint - dists).sum()
    return loss


def kinematic_loss_multi(
    instances: Sequence[PPPRInstance],
    cfg: OptimConfig,
) -> torch.Tensor:
    """L_kine,multi (Eq. 20)."""
    loss = instances[0].positions.new_zeros(())
    for inst in instances:
        loss = loss + kinematic_loss(inst, cfg)
    loss = loss + centroid_separation_loss(instances, cfg.d_sep)
    loss = loss + joint_collision_loss(instances, cfg.d_joint)
    return loss


def total_loss(
    em: torch.Tensor,
    kine: torch.Tensor,
    w_em: float,
    w_kine: float,
) -> torch.Tensor:
    """L_total = w_EM L_EM + w_kine L_kine  (Eq. 17)."""
    return w_em * em + w_kine * kine
