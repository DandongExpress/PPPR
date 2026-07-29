"""Dataset loaders for MMVR / HuPR / XRF55 with a flexible NPZ protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pppr.config import RadarConfig, load_config, radar_from_config
from pppr.skeleton import NUM_JOINTS, t_pose_positions
from pppr.utils import synthesize_heatmap_from_joints


DATASET_NAMES = ("MMVR", "HuPR", "XRF55")


def resolve_radar(dataset: str) -> RadarConfig:
    cfg = load_config(dataset=dataset)
    return radar_from_config(cfg)


class HeatmapPoseDataset(Dataset):
    """Generic mmWave HPE dataset.

    Expected layout under ``root`` (any of the following works):

    1. ``*.npz`` files each containing:
       - ``heatmap`` : [R, A] or [R, A, E]
       - ``joints``  : [J, 3]  (metres; optional for prepare-only)
       - optional ``doppler`` : [R, A, D]
       - optional ``n_person`` : int

    2. Subfolders ``heatmaps/`` + ``joints/`` with matching stems.

    3. If ``root`` is missing / empty and ``synthetic=True``, generate demo frames.
    """

    def __init__(
        self,
        root: str,
        dataset: str = "MMVR",
        split: str = "train",
        synthetic: bool = False,
        synthetic_size: int = 64,
        transform=None,
    ):
        self.root = Path(root)
        self.dataset = dataset.upper()
        self.split = split
        self.transform = transform
        self.radar = resolve_radar(self.dataset)
        self.samples: List[Path] = []
        self._synthetic = False
        self._synth_size = synthetic_size

        if synthetic or not self.root.exists():
            self._synthetic = True
        else:
            self.samples = self._discover()
            if len(self.samples) == 0:
                self._synthetic = True

    def _discover(self) -> List[Path]:
        root = self.root
        split_dir = root / self.split
        search_roots = [split_dir, root] if split_dir.exists() else [root]
        files: List[Path] = []
        for sr in search_roots:
            files.extend(sorted(sr.rglob("*.npz")))
        # Prefer files that look like frames
        return [f for f in files if "pppr" not in f.stem.lower()]

    def __len__(self) -> int:
        if self._synthetic:
            return self._synth_size
        return len(self.samples)

    def _load_npz(self, path: Path) -> Dict[str, Any]:
        data = np.load(str(path), allow_pickle=True)
        keys = set(data.files)
        heatmap = None
        for k in ("heatmap", "hori", "H", "radar"):
            if k in keys:
                heatmap = np.asarray(data[k], dtype=np.float32)
                break
        if heatmap is None:
            raise KeyError(f"No heatmap key in {path}")
        joints = None
        for k in ("joints", "pose", "keypoints", "gt"):
            if k in keys:
                joints = np.asarray(data[k], dtype=np.float32)
                break
        doppler = data["doppler"] if "doppler" in keys else None
        n_person = int(data["n_person"]) if "n_person" in keys else 1
        return {
            "heatmap": heatmap,
            "joints": joints,
            "doppler": None if doppler is None else np.asarray(doppler, dtype=np.float32),
            "n_person": n_person,
            "id": path.stem,
            "path": str(path),
        }

    def _make_synthetic(self, idx: int) -> Dict[str, Any]:
        rng = np.random.RandomState(idx + (0 if self.split == "train" else 10_000))
        base = t_pose_positions(distance=float(rng.uniform(1.2, 2.5))).numpy()
        # Random joint jitter
        joints = base + rng.randn(*base.shape).astype(np.float32) * 0.03
        joints[:, 1] += float(rng.uniform(-0.3, 0.3))
        h = synthesize_heatmap_from_joints(
            joints,
            range_bins=self.radar.range_bins,
            angle_bins=self.radar.angle_bins,
            max_range=self.radar.max_range,
            fov_h_deg=self.radar.fov_h_deg,
            noise_level=0.08,
        )
        return {
            "heatmap": h,
            "joints": joints.astype(np.float32),
            "doppler": None,
            "n_person": 1,
            "id": f"synth_{self.split}_{idx:05d}",
            "path": "",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._synthetic:
            sample = self._make_synthetic(idx)
        else:
            sample = self._load_npz(self.samples[idx])
        if sample["joints"] is None:
            sample["joints"] = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_heatmap_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    heatmaps = [torch.from_numpy(np.asarray(b["heatmap"], dtype=np.float32)) for b in batch]
    # Pad / resize to max shape in batch via simple crop/pad on last two dims
    max_r = max(h.shape[0] for h in heatmaps)
    max_a = max(h.shape[-1] if h.ndim == 2 else h.shape[1] for h in heatmaps)
    h_out = []
    for h in heatmaps:
        if h.ndim == 3:
            h = h.max(dim=-1).values
        r, a = h.shape
        canvas = torch.zeros(max_r, max_a)
        canvas[:r, :a] = h
        h_out.append(canvas)
    joints = torch.stack(
        [torch.from_numpy(np.asarray(b["joints"], dtype=np.float32)) for b in batch]
    )
    return {
        "heatmap": torch.stack(h_out),
        "joints": joints,
        "ids": [b["id"] for b in batch],
        "n_person": torch.tensor([b.get("n_person", 1) for b in batch]),
    }


def build_dataset(
    dataset: str,
    data_root: str,
    split: str = "train",
    synthetic: bool = False,
    **kwargs,
) -> HeatmapPoseDataset:
    return HeatmapPoseDataset(
        root=data_root,
        dataset=dataset,
        split=split,
        synthetic=synthetic,
        **kwargs,
    )
