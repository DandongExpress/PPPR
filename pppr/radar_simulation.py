"""Differentiable radar pipeline simulation (Sec. 4.2, Eqs. 6–11)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import RadarConfig
from .representation import PPPRInstance


class RadarSimulator:
    """Reconstruct H_sim from PPPR via electromagnetic operators.

    H_sim = Σ_j M_atten^{(j)} M_range^{(j)} M_Dopp^{(j)} M_angle^{(j)} R_j
    """

    def __init__(self, radar: RadarConfig, device: Optional[torch.device] = None):
        self.radar = radar
        self.device = device or torch.device("cpu")

    def _grid(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Range–azimuth (–elevation) coordinate grids in Cartesian metres."""
        r = self.radar
        range_vals = torch.linspace(0, r.max_range, r.range_bins, device=self.device)
        az_vals = torch.linspace(
            -0.5 * r.fov_h_deg, 0.5 * r.fov_h_deg, r.angle_bins, device=self.device
        )
        az = torch.deg2rad(az_vals)
        if r.elevation_bins > 1:
            el_vals = torch.linspace(
                -0.5 * r.fov_v_deg, 0.5 * r.fov_v_deg, r.elevation_bins, device=self.device
            )
            el = torch.deg2rad(el_vals)
            R, AZ, EL = torch.meshgrid(range_vals, az, el, indexing="ij")
            X = R * torch.cos(EL) * torch.cos(AZ)
            Y = R * torch.cos(EL) * torch.sin(AZ)
            Z = R * torch.sin(EL)
            return X, Y, Z
        R, AZ = torch.meshgrid(range_vals, az, indexing="ij")
        X = R * torch.cos(AZ)
        Y = R * torch.sin(AZ)
        Z = torch.zeros_like(X)
        return X, Y, Z

    def joint_contribution(
        self,
        position: torch.Tensor,
        scale: torch.Tensor,
        rotation_q: torch.Tensor,
        velocity: torch.Tensor,
        opacity: torch.Tensor,
        omega: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """Complex field contribution of one joint on the RA(E) grid."""
        rcfg = self.radar
        c = rcfg.speed_of_light
        p = position
        d = torch.norm(p).clamp(min=1e-4)

        # --- Spatial Gaussian G_j (Eq. 4), evaluated on grid ---
        from .representation import quaternion_to_rotation_matrix

        s = scale.abs().clamp(min=1e-4)
        Rmat = quaternion_to_rotation_matrix(rotation_q)
        S = torch.diag(s)
        cov = Rmat @ S @ S.T @ Rmat.T
        cov = cov + 1e-5 * torch.eye(3, device=p.device, dtype=p.dtype)

        pts = torch.stack([X, Y, Z], dim=-1)  # [..., 3]
        diff = pts - p
        # Mahalanobis via inverse
        inv = torch.linalg.inv(cov)
        quad = torch.einsum("...i,ij,...j->...", diff, inv, diff)
        # Soft spatial support (omit exact (2π)^{-3/2}|Σ|^{-1/2} for heatmap amp)
        G = torch.exp(-0.5 * quad)

        # Complex return phase from Eq. 5
        n = min(omega.numel(), 3)
        phase_feat = velocity[:n] * (pts - p)[..., :n]
        phase_doppler_spatial = (omega[:n] * phase_feat).sum(dim=-1)
        R_complex = opacity * G * torch.exp(1j * phase_doppler_spatial)

        # --- M_atten (Eq. 7): d^{-4} / d_max^{-4} ---
        M_atten = (d ** (-4)) / (rcfg.max_range ** (-4) + 1e-12)

        # --- M_range (Eq. 8): exp(i 2π f_beat τ), τ = 2d/c, f_beat = S τ ---
        tau = 2.0 * d / c
        f_beat = rcfg.chirp_slope * tau
        M_range = torch.exp(1j * 2.0 * torch.pi * f_beat * tau)

        # --- M_Dopp (Eq. 9): exp(i 2π Δf_Dopp T_frame) ---
        radial_dir = p / d
        v_r = torch.dot(velocity, radial_dir)
        delta_f = 2.0 * v_r / rcfg.wavelength
        M_dopp = torch.exp(1j * 2.0 * torch.pi * delta_f * rcfg.frame_duration)

        # --- M_angle (Eqs. 10–11) ---
        theta_az = torch.atan2(p[1], p[0])
        horiz = torch.sqrt(p[0] ** 2 + p[1] ** 2).clamp(min=1e-6)
        theta_el = torch.atan2(p[2], horiz)
        dphi_az = (2.0 * torch.pi / rcfg.wavelength) * rcfg.antenna_spacing_az_m * torch.sin(
            theta_az
        )
        dphi_el = (2.0 * torch.pi / rcfg.wavelength) * rcfg.antenna_spacing_el_m * torch.sin(
            theta_el
        )
        M_angle = torch.exp(1j * dphi_az) * torch.exp(1j * dphi_el)

        field = M_atten * M_range * M_dopp * M_angle * R_complex
        return field

    def simulate(self, instance: PPPRInstance) -> torch.Tensor:
        """Return real-valued Heatmap intensity |H_sim| matching radar bins."""
        X, Y, Z = self._grid()
        field = torch.zeros_like(X, dtype=torch.complex64)
        for joint in instance.joints:
            feat = joint()
            field = field + self.joint_contribution(
                position=feat["position"],
                scale=feat["scale"],
                rotation_q=feat["rotation"],
                velocity=feat["velocity"],
                opacity=feat["opacity"],
                omega=feat["omega"],
                X=X,
                Y=Y,
                Z=Z,
            )
        power = torch.abs(field)
        # Optional elevation collapse for 2D Heatmaps used by most HPE models
        if power.ndim == 3:
            power = power.sum(dim=-1)
        # Soft normalize for stable IoU comparison
        power = power / (power.max().clamp(min=1e-8))
        return power

    def simulate_multi(self, instances: list) -> torch.Tensor:
        """Superpose multiple persons' simulated Heatmaps."""
        acc = None
        for inst in instances:
            h = self.simulate(inst)
            acc = h if acc is None else acc + h
        assert acc is not None
        return acc / (acc.max().clamp(min=1e-8))


def match_heatmap_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Radar-agnostic size adaptation via bilinear resize (Sec. 4.2.2)."""
    if pred.shape == target.shape:
        return pred
    # Assume 2D Heatmaps
    if pred.ndim == 2 and target.ndim == 2:
        p = pred.unsqueeze(0).unsqueeze(0)
        out = F.interpolate(p, size=target.shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0)
    if pred.ndim == 3 and target.ndim == 3:
        p = pred.unsqueeze(0)
        out = F.interpolate(p, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return out.squeeze(0)
    # Fallback flatten elevation
    if pred.ndim == 3 and target.ndim == 2:
        pred = pred.sum(dim=-1)
        return match_heatmap_shape(pred, target)
    return pred
