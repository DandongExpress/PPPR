"""Shared utilities: metrics, synthetic data, I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def majpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Joint Position Error (mm). pred/gt: [J, 3] in metres."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)) * 1000.0)


def pa_majpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-Aligned MAJPE (mm)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    X = pred - mu_p
    Y = gt - mu_g
    # Kabsch
    C = X.T @ Y
    U, _, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = (X @ R) + mu_g
    return float(np.mean(np.linalg.norm(aligned - gt, axis=-1)) * 1000.0)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def synthesize_heatmap_from_joints(
    joints: np.ndarray,
    range_bins: int = 256,
    angle_bins: int = 128,
    max_range: float = 4.0,
    fov_h_deg: float = 90.0,
    sigma_r: float = 2.0,
    sigma_a: float = 2.0,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Render a synthetic range–azimuth Heatmap from 3D joints (for demos/tests)."""
    h = np.zeros((range_bins, angle_bins), dtype=np.float32)
    rr = np.arange(range_bins)[:, None]
    aa = np.arange(angle_bins)[None, :]
    for p in joints:
        r = float(np.linalg.norm(p[:2]))
        theta = float(np.arctan2(p[1], p[0]))
        r_idx = (r / max_range) * (range_bins - 1)
        a_idx = (theta / np.deg2rad(fov_h_deg) + 0.5) * (angle_bins - 1)
        if not (0 <= r_idx < range_bins and 0 <= a_idx < angle_bins):
            continue
        amp = 1.0 / (max(r, 0.3) ** 2)
        blob = amp * np.exp(
            -0.5 * (((rr - r_idx) / sigma_r) ** 2 + ((aa - a_idx) / sigma_a) ** 2)
        )
        h += blob.astype(np.float32)
    if noise_level > 0:
        h += noise_level * np.random.randn(*h.shape).astype(np.float32)
        # Sparse clutter peaks
        for _ in range(8):
            ri = np.random.randint(0, range_bins)
            ai = np.random.randint(0, angle_bins)
            h[ri, ai] += float(np.random.uniform(0.2, 0.6))
    h = np.clip(h, 0, None)
    h /= h.max() + 1e-8
    return h


def save_pppr_npz(
    path: Union[str, Path],
    packed: torch.Tensor,
    positions: np.ndarray,
    heatmap_pppr: Optional[np.ndarray] = None,
    pc_points: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
) -> None:
    payload = {
        "pppr": packed.cpu().numpy() if torch.is_tensor(packed) else packed,
        "positions": positions,
        "meta": meta or {},
    }
    if heatmap_pppr is not None:
        payload["pppr_heatmap"] = heatmap_pppr
    if pc_points is not None:
        payload["pppr_pc"] = pc_points
    ensure_dir(Path(path).parent)
    np.savez_compressed(str(path), **payload)


def load_pppr_npz(path: Union[str, Path]) -> Dict:
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}
