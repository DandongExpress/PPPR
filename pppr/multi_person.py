"""Multi-person counting: ETCM-CFAR + DBSCAN + MLP (Sec. 4.4, Appendix A.2)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from .config import RadarConfig
from .initialization import bin_to_cartesian, detect_peaks, handle_nans


def etcm_cfar(
    heatmap: np.ndarray,
    beta: float = 3.0,
    guard: int = 2,
    ref: int = 6,
) -> np.ndarray:
    """Simplified ETCM-CFAR peak mask (Eq. 23).

    threshold = μ_ref + β σ_ref over reference cells excluding guard cells.
    """
    h = handle_nans(heatmap)
    if h.ndim == 3:
        h2 = h.max(axis=-1)
    else:
        h2 = h
    H, W = h2.shape
    mask = np.zeros_like(h2, dtype=bool)
    for i in range(H):
        for j in range(W):
            i0 = max(0, i - ref)
            i1 = min(H, i + ref + 1)
            j0 = max(0, j - ref)
            j1 = min(W, j + ref + 1)
            window = h2[i0:i1, j0:j1].copy()
            # Zero-out guard region
            gi0 = max(0, i - guard) - i0
            gi1 = min(H, i + guard + 1) - i0
            gj0 = max(0, j - guard) - j0
            gj1 = min(W, j + guard + 1) - j0
            window[gi0:gi1, gj0:gj1] = np.nan
            refs = window[~np.isnan(window)]
            if refs.size < 4:
                continue
            thr = refs.mean() + beta * refs.std()
            if h2[i, j] > thr:
                mask[i, j] = True
    return mask


class PersonCountMLP(nn.Module):
    """Appendix A.2: FC(12→64)→ReLU→Dropout→FC(64→32)→ReLU→FC(32→5)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cluster_features(
    points: np.ndarray,
    intensities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Build 12-D cluster statistics for MLP (Appendix A.2)."""
    feats = []
    for k in sorted(set(labels)):
        if k < 0:
            continue
        pts = points[labels == k]
        ints = intensities[labels == k]
        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        mu_i = float(ints.mean()) if len(ints) else 0.0
        var_i = float(ints.var()) if len(ints) else 0.0
        n_pts = float(len(pts))
        # Temporal consistency placeholder (single-frame → 1.0)
        c_t = 1.0
        feat = np.array(
            [
                centroid[0],
                centroid[1],
                centroid[2],
                extent[0],
                extent[1],
                extent[2],
                mu_i,
                var_i,
                n_pts,
                c_t,
                float(pts[:, 0].std()),
                float(pts[:, 1].std()),
            ],
            dtype=np.float32,
        )
        feats.append(feat)
    if not feats:
        return np.zeros((0, 12), dtype=np.float32)
    return np.stack(feats)


class PersonCounter:
    """ETCM-CFAR → DBSCAN → MLP person counting module."""

    def __init__(
        self,
        radar: RadarConfig,
        eps: float = 0.3,
        min_pts: int = 3,
        cfar_beta: float = 3.0,
        mlp_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.radar = radar
        self.eps = eps
        self.min_pts = min_pts
        self.cfar_beta = cfar_beta
        self.device = device or torch.device("cpu")
        self.mlp = PersonCountMLP().to(self.device)
        self.mlp.eval()
        if mlp_checkpoint:
            state = torch.load(mlp_checkpoint, map_location=self.device)
            self.mlp.load_state_dict(state)

    def detect_points(
        self, heatmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = etcm_cfar(heatmap, beta=self.cfar_beta)
        # Also include local maxima for denser peaks
        peaks = detect_peaks(heatmap, percentile=85.0)
        h = handle_nans(heatmap)
        if h.ndim == 3:
            h2 = h.max(axis=-1)
        else:
            h2 = h
        coords = np.argwhere(mask)
        if len(peaks):
            coords = np.unique(np.vstack([coords, peaks]), axis=0) if len(coords) else peaks
        points = []
        intensities = []
        for pk in coords:
            points.append(bin_to_cartesian(float(pk[0]), float(pk[1]), self.radar))
            intensities.append(float(h2[pk[0], pk[1]]))
        if not points:
            return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
        return np.stack(points), np.asarray(intensities, dtype=np.float32)

    def count(self, heatmap: np.ndarray) -> Tuple[int, List[np.ndarray]]:
        """Return (N_person, list of per-person point arrays)."""
        points, intensities = self.detect_points(heatmap)
        if len(points) == 0:
            return 1, []  # default single person
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_pts)
        labels = clustering.fit_predict(points)
        valid = [k for k in set(labels) if k >= 0]
        n_dbscan = max(len(valid), 1)

        feats = cluster_features(points, intensities, labels)
        if len(feats) == 0:
            return n_dbscan, [points]

        with torch.no_grad():
            # Aggregate cluster features by mean for global count head
            x = torch.from_numpy(feats.mean(axis=0, keepdims=True)).to(self.device)
            logits = self.mlp(x)
            n_mlp = int(logits.argmax(dim=-1).item())
        # Blend: prefer DBSCAN cluster count when MLP is untrained (near-uniform)
        probs = torch.softmax(logits, dim=-1)
        if float(probs.max()) < 0.35:
            n_person = n_dbscan
        else:
            n_person = max(n_mlp, 1)

        clusters = []
        for k in sorted(valid):
            clusters.append(points[labels == k])
        if not clusters:
            clusters = [points]
        # If MLP says more people than clusters, keep DBSCAN count
        n_person = min(max(n_person, 1), max(len(clusters), 1))
        return n_person, clusters[:n_person]
