"""Physics-informed initialization from Heatmaps (Sec. 4.1)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import AgglomerativeClustering

from .config import RadarConfig
from .skeleton import NUM_JOINTS, t_pose_positions


def handle_nans(heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Fill NaNs via low-pass filtering (Sec. 4.1.1)."""
    out = heatmap.astype(np.float64).copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out
    out[nan_mask] = 0.0
    filled = gaussian_filter(out, sigma=sigma)
    out[nan_mask] = filled[nan_mask]
    return out


def detect_peaks(
    heatmap: np.ndarray,
    percentile: float = 90.0,
    min_distance: int = 2,
) -> np.ndarray:
    """Local maxima above adaptive percentile threshold → P_cand."""
    h = handle_nans(heatmap)
    if h.ndim == 3:
        # Collapse elevation / Doppler by max projection for peak finding
        h2 = h.max(axis=-1)
    else:
        h2 = h
    thr = np.percentile(h2, percentile)
    neighborhood = maximum_filter(h2, size=min_distance * 2 + 1)
    mask = (h2 == neighborhood) & (h2 >= thr)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        # Fallback: global max
        idx = np.unravel_index(np.argmax(h2), h2.shape)
        coords = np.array([idx])
    # Sort by intensity descending
    intensities = h2[coords[:, 0], coords[:, 1]]
    order = np.argsort(-intensities)
    return coords[order]


def bin_to_cartesian(
    n_r: float,
    n_a: float,
    radar: RadarConfig,
    n_e: Optional[float] = None,
    n_r_max: Optional[int] = None,
    n_a_max: Optional[int] = None,
) -> np.ndarray:
    """Bin indices → Cartesian (Eq. 2 / elevation variant)."""
    Nr = n_r_max or radar.range_bins
    Na = n_a_max or radar.angle_bins
    r = (n_r / max(Nr - 1, 1)) * radar.max_range
    theta_az = ((n_a / max(Na - 1, 1)) - 0.5) * np.deg2rad(radar.fov_h_deg)
    if n_e is not None and radar.elevation_bins > 1:
        Ne = radar.elevation_bins
        theta_el = ((n_e / max(Ne - 1, 1)) - 0.5) * np.deg2rad(radar.fov_v_deg)
        x = r * np.cos(theta_el) * np.cos(theta_az)
        y = r * np.cos(theta_el) * np.sin(theta_az)
        z = r * np.sin(theta_el)
    else:
        # Planar init: z = 0 (Sec. 4.1.1)
        x = r * np.cos(theta_az)
        y = r * np.sin(theta_az)
        z = 0.0
    return np.array([x, y, z], dtype=np.float32)


def doppler_to_velocity(
    n_d: float,
    theta_az: float,
    theta_el: float,
    radar: RadarConfig,
) -> np.ndarray:
    """Eq. 3: v_r = λ n_d PRF / (2 N_d), then Cartesian decomposition."""
    Nd = max(radar.doppler_bins, 1)
    # Centre Doppler bins around zero
    n_d_c = n_d - (Nd - 1) / 2.0
    v_r = (radar.wavelength * n_d_c * radar.prf) / (2.0 * Nd)
    vx = v_r * np.cos(theta_el) * np.cos(theta_az)
    vy = v_r * np.cos(theta_el) * np.sin(theta_az)
    vz = v_r * np.sin(theta_el)
    return np.array([vx, vy, vz], dtype=np.float32)


def gradient_velocity(
    heatmap: np.ndarray,
    peak: np.ndarray,
    radar: RadarConfig,
) -> np.ndarray:
    """Appendix A.1: v̂_r = γ ||∇H|| for Doppler-deficient datasets."""
    h = handle_nans(heatmap)
    if h.ndim == 3:
        h = h.max(axis=-1)
    gy, gx = np.gradient(h.astype(np.float64))
    r_idx = int(np.clip(peak[0], 0, h.shape[0] - 1))
    a_idx = int(np.clip(peak[1], 0, h.shape[1] - 1))
    g = float(np.hypot(gy[r_idx, a_idx], gx[r_idx, a_idx]))
    v_r = radar.velocity_gamma * g
    # Cap to typical human walking speed
    v_r = float(np.clip(v_r, -2.0, 2.0))
    Nr, Na = h.shape
    theta_az = ((a_idx / max(Na - 1, 1)) - 0.5) * np.deg2rad(radar.fov_h_deg)
    return np.array(
        [v_r * np.cos(theta_az), v_r * np.sin(theta_az), 0.0],
        dtype=np.float32,
    )


def extract_peaks_with_velocity(
    heatmap: np.ndarray,
    radar: RadarConfig,
    doppler_volume: Optional[np.ndarray] = None,
    max_peaks: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions [M,3], velocities [M,3]) from Heatmap peaks."""
    peaks = detect_peaks(heatmap)[:max_peaks]
    positions = []
    velocities = []
    h = handle_nans(heatmap)
    for pk in peaks:
        n_e = None
        if h.ndim == 3 and h.shape[-1] == radar.elevation_bins:
            # Prefer strongest elevation bin at this RA location
            n_e = int(np.argmax(h[pk[0], pk[1]]))
        pos = bin_to_cartesian(float(pk[0]), float(pk[1]), radar, n_e=n_e)
        positions.append(pos)
        Nr = h.shape[0] if h.ndim >= 2 else radar.range_bins
        Na = h.shape[1] if h.ndim >= 2 else radar.angle_bins
        theta_az = ((pk[1] / max(Na - 1, 1)) - 0.5) * np.deg2rad(radar.fov_h_deg)
        theta_el = 0.0
        if n_e is not None and radar.elevation_bins > 1:
            theta_el = ((n_e / max(radar.elevation_bins - 1, 1)) - 0.5) * np.deg2rad(
                radar.fov_v_deg
            )
        if doppler_volume is not None:
            # Take argmax Doppler at this spatial location
            if doppler_volume.ndim == 3:
                n_d = int(np.argmax(doppler_volume[pk[0], pk[1]]))
            else:
                n_d = float(doppler_volume)
            vel = doppler_to_velocity(n_d, theta_az, theta_el, radar)
        elif radar.approximate_velocity:
            vel = gradient_velocity(h, pk, radar)
        else:
            vel = np.zeros(3, dtype=np.float32)
        velocities.append(vel)
    if not positions:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.stack(positions), np.stack(velocities)


def map_peaks_to_joints(
    positions: np.ndarray,
    velocities: np.ndarray,
    num_joints: int = NUM_JOINTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical clustering + skeletal anchoring (Sec. 4.1.2)."""
    template = t_pose_positions().numpy()
    if len(positions) == 0:
        return torch.from_numpy(template), torch.zeros_like(torch.from_numpy(template))

    # Estimate body centroid from peaks
    centroid = positions.mean(axis=0)
    template_c = template.copy()
    template_c[:, 0] = centroid[0]
    template_c[:, 1] += centroid[1]
    # Scale template depth to observed range
    if centroid[0] > 0.3:
        template_c[:, 0] = centroid[0] + (template[:, 0] - template[:, 0].mean()) * 0.15

    # Assign each joint to nearest peak (with soft fallback to template)
    joint_pos = template_c.copy()
    joint_vel = np.zeros_like(joint_pos)
    if len(positions) >= 2:
        n_clusters = min(num_joints, len(positions))
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(positions)
            cluster_centers = np.array(
                [positions[labels == k].mean(axis=0) for k in range(n_clusters)]
            )
            cluster_vels = np.array(
                [velocities[labels == k].mean(axis=0) for k in range(n_clusters)]
            )
        except Exception:
            cluster_centers = positions
            cluster_vels = velocities
    else:
        cluster_centers = positions
        cluster_vels = velocities

    used = set()
    for j in range(num_joints):
        dists = np.linalg.norm(cluster_centers - template_c[j], axis=1)
        order = np.argsort(dists)
        chosen = int(order[0])
        # Prefer unused clusters when possible
        for c in order:
            if int(c) not in used:
                chosen = int(c)
                break
        used.add(chosen)
        # Blend template prior with observed peak
        alpha = 0.6
        joint_pos[j] = alpha * cluster_centers[chosen] + (1 - alpha) * template_c[j]
        # Keep anatomical height prior stronger for vertical axis
        joint_pos[j, 2] = 0.3 * cluster_centers[chosen, 2] + 0.7 * template_c[j, 2]
        joint_vel[j] = cluster_vels[min(chosen, len(cluster_vels) - 1)]

    return torch.from_numpy(joint_pos.astype(np.float32)), torch.from_numpy(
        joint_vel.astype(np.float32)
    )


class Initializer:
    """Sec. 4.1: Extract P and V, then parameterize as PPPR seeds."""

    def __init__(self, radar: RadarConfig):
        self.radar = radar

    def __call__(
        self,
        heatmap: np.ndarray,
        doppler_volume: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos, vel = extract_peaks_with_velocity(heatmap, self.radar, doppler_volume)
        return map_peaks_to_joints(pos, vel)
