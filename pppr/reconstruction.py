"""Convert optimized PPPR back to Heatmap / Point Cloud (Fig. 1)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .config import RadarConfig
from .radar_simulation import RadarSimulator
from .representation import PPPRInstance


def pppr_to_heatmap(
    instance: Union[PPPRInstance, list],
    radar: RadarConfig,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reproject PPPR parameters to a synthetic Heatmap (PPPR-Heatmap)."""
    device = device or next(
        (j.position.device for inst in (instance if isinstance(instance, list) else [instance]) for j in inst.joints),
        torch.device("cpu"),
    )
    sim = RadarSimulator(radar, device=device)
    if isinstance(instance, list):
        return sim.simulate_multi(instance)
    return sim.simulate(instance)


def pppr_to_pointcloud(
    instance: Union[PPPRInstance, list],
    radar: RadarConfig,
    max_points: int = 512,
    tau_pct: float = 10.0,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """PPPR-Heatmap → Point Cloud via percentile thresholding (PPPR-PC).

    Returns:
        points: [N, 3] Cartesian coordinates
        intensities: [N]
    """
    heatmap = pppr_to_heatmap(instance, radar, device=device)
    h = heatmap.detach().cpu().numpy()
    flat = h.reshape(-1)
    thr = np.percentile(flat, 100.0 - tau_pct)
    mask = h >= thr
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)

    intensities = h[mask]
    order = np.argsort(-intensities)[:max_points]
    coords = coords[order]
    intensities = intensities[order]

    points = []
    for r_idx, a_idx in coords:
        r = (r_idx / max(radar.range_bins - 1, 1)) * radar.max_range
        theta = ((a_idx / max(radar.angle_bins - 1, 1)) - 0.5) * np.deg2rad(radar.fov_h_deg)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.0
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32), intensities.astype(np.float32)


def heatmap_to_pointcloud(
    heatmap: Union[np.ndarray, torch.Tensor],
    radar: RadarConfig,
    max_points: int = 512,
    tau_pct: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """CFAR-style percentile extraction from a raw Heatmap → PC."""
    if isinstance(heatmap, torch.Tensor):
        h = heatmap.detach().cpu().numpy()
    else:
        h = np.asarray(heatmap)
    if h.ndim == 3:
        h = h.max(axis=-1)
    flat = h.reshape(-1)
    thr = np.percentile(flat, 100.0 - tau_pct)
    mask = h >= thr
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    intensities = h[mask]
    order = np.argsort(-intensities)[:max_points]
    coords = coords[order]
    intensities = intensities[order]
    points = []
    for r_idx, a_idx in coords:
        r = (r_idx / max(h.shape[0] - 1, 1)) * radar.max_range
        theta = ((a_idx / max(h.shape[1] - 1, 1)) - 0.5) * np.deg2rad(radar.fov_h_deg)
        points.append([r * np.cos(theta), r * np.sin(theta), 0.0])
    return np.asarray(points, dtype=np.float32), intensities.astype(np.float32)
