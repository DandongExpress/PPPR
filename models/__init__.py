"""HPE model registry and lightweight plug-and-play backbones.

Full official RETR / HuPR / mmDiff / PoseformerV2 weights can be plugged in via
``register_model``. The implementations below provide end-to-end trainable
surrogates that accept heatmap / PC / PPPR inputs so the public pipeline runs
without external checkpoints.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from pppr.representation import mean_pool_pppr_features
from pppr.skeleton import NUM_JOINTS

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    def deco(cls):
        _REGISTRY[name.lower()] = cls
        return cls

    return deco


def list_models():
    return sorted(_REGISTRY.keys())


def build_model(name: str, input_type: str = "pppr", **kwargs) -> nn.Module:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[key](input_type=input_type, **kwargs)


class HeatmapEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.net(x)
        return self.fc(h.flatten(1))


class PointCloudEncoder(nn.Module):
    """Permutation-invariant PointNet-style encoder for variable-length PC."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: [B, N, 3] or [B, N, 4]
        if pts.shape[-1] == 3:
            ones = torch.ones(*pts.shape[:-1], 1, device=pts.device, dtype=pts.dtype)
            pts = torch.cat([pts, ones], dim=-1)
        feat = self.mlp(pts)
        return feat.max(dim=1).values


class PPPREncoder(nn.Module):
    def __init__(self, in_dim: int = 374, out_dim: int = 256):
        # 17 joints * 22 params = 374
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.flatten(1)
        return self.net(x)


class PoseHead(nn.Module):
    def __init__(self, in_dim: int = 256, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_joints * 3),
        )
        self.num_joints = num_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).view(-1, self.num_joints, 3)


class BaseHPE(nn.Module):
    """Shared encode → pose head interface for all input types."""

    def __init__(self, input_type: str = "pppr", feat_dim: int = 256, **kwargs):
        super().__init__()
        self.input_type = input_type.lower()
        self.heatmap_enc = HeatmapEncoder(feat_dim)
        self.pc_enc = PointCloudEncoder(feat_dim)
        self.pppr_enc = PPPREncoder(out_dim=feat_dim)
        self.head = PoseHead(feat_dim)

    def encode(self, batch: Dict) -> torch.Tensor:
        t = self.input_type
        if t in ("heatmap", "pppr_heatmap"):
            key = "pppr_heatmap" if t == "pppr_heatmap" and "pppr_heatmap" in batch else "heatmap"
            return self.heatmap_enc(batch[key])
        if t in ("pc", "pppr_pc"):
            key = "pppr_pc" if t == "pppr_pc" and "pppr_pc" in batch else "pc"
            return self.pc_enc(batch[key])
        # pppr
        return self.pppr_enc(batch["pppr"])

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.head(self.encode(batch))


@register_model("RETR")
class RETR(BaseHPE):
    """Transformer-style Heatmap/PPPR HPE surrogate (RETR-compatible I/O)."""

    def __init__(self, input_type: str = "pppr", feat_dim: int = 256, **kwargs):
        super().__init__(input_type=input_type, feat_dim=feat_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, batch: Dict) -> torch.Tensor:
        feat = self.encode(batch).unsqueeze(1)
        feat = self.transformer(feat).squeeze(1)
        return self.head(self.proj(feat))


@register_model("HuprModel")
class HuprModel(BaseHPE):
    """Multi-scale CNN surrogate for HuPR model."""

    def __init__(self, input_type: str = "pppr", feat_dim: int = 256, **kwargs):
        super().__init__(input_type=input_type, feat_dim=feat_dim)
        self.refine = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.head(self.refine(self.encode(batch)))


@register_model("mmDiff")
class mmDiff(BaseHPE):
    """Diffusion-style iterative refinement surrogate for PC / PPPR-PC."""

    def __init__(self, input_type: str = "pppr", feat_dim: int = 256, steps: int = 3, **kwargs):
        super().__init__(input_type=input_type, feat_dim=feat_dim)
        self.steps = steps
        self.denoise = nn.Sequential(
            nn.Linear(feat_dim + NUM_JOINTS * 3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, NUM_JOINTS * 3),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        feat = self.encode(batch)
        pose = self.head(feat)
        for _ in range(self.steps):
            inp = torch.cat([feat, pose.flatten(1)], dim=-1)
            pose = pose + 0.1 * self.denoise(inp).view_as(pose)
        return pose


@register_model("PoseformerV2")
class PoseformerV2(BaseHPE):
    """Temporal transformer surrogate; single-frame uses identity temporal axis."""

    def __init__(self, input_type: str = "pppr", feat_dim: int = 256, **kwargs):
        super().__init__(input_type=input_type, feat_dim=feat_dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=4, batch_first=True),
            num_layers=2,
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        feat = self.encode(batch).unsqueeze(1)  # [B, 1, D]
        feat = self.temporal(feat).squeeze(1)
        return self.head(feat)


@register_model("MLP")
class MLPPPPR(nn.Module):
    """Appendix A.6: MLP+PPPR baseline (48 → 2048 → 1792 → 3 N_j)."""

    def __init__(self, input_type: str = "pppr", **kwargs):
        super().__init__()
        self.input_type = "pppr"
        self.net = nn.Sequential(
            nn.Linear(48, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1792),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1792, NUM_JOINTS * 3),
        )

    def forward(self, batch: Dict) -> torch.Tensor:
        x = batch["pppr"]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        feats = []
        for i in range(x.shape[0]):
            feats.append(mean_pool_pppr_features(x[i]))
        feat = torch.stack(feats, dim=0)
        return self.net(feat).view(-1, NUM_JOINTS, 3)
