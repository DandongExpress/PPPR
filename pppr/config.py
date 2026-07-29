"""Radar configuration helpers and dataset-specific calibration (Sec. 4.2.2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RadarConfig:
    """Physical / FFT parameters for radar-agnostic calibration."""

    wavelength: float = 3.896e-3
    max_range: float = 4.0
    fov_h_deg: float = 90.0
    fov_v_deg: float = 90.0
    range_bins: int = 256
    angle_bins: int = 128
    elevation_bins: int = 1
    doppler_bins: int = 128
    prf: float = 10000.0
    bandwidth: float = 4.0e9
    chirp_duration: float = 40.0e-6
    frame_duration: float = 0.033
    antenna_spacing_az: float = 0.5  # wavelengths
    antenna_spacing_el: float = 0.5
    fft_mode: str = "3d_unified"
    approximate_velocity: bool = False
    velocity_gamma: float = 0.5
    speed_of_light: float = 2.99792458e8

    @property
    def chirp_slope(self) -> float:
        return self.bandwidth / self.chirp_duration

    @property
    def antenna_spacing_az_m(self) -> float:
        return self.antenna_spacing_az * self.wavelength

    @property
    def antenna_spacing_el_m(self) -> float:
        return self.antenna_spacing_el * self.wavelength

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    n_iter: int = 100
    w_em: float = 0.5
    w_kine: float = 0.5
    tau_pct: float = 10.0
    bone_tolerance: float = 0.05
    theta_max_deg: float = 170.0
    d_sep: float = 0.5
    d_joint: float = 0.1


def deep_update(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    dataset: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1] / "configs"
    cfg: Dict[str, Any] = {}
    default_path = root / "default.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if dataset:
        ds_path = root / f"{dataset.lower()}.yaml"
        if ds_path.exists():
            with open(ds_path, "r", encoding="utf-8") as f:
                cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def radar_from_config(cfg: Dict[str, Any]) -> RadarConfig:
    r = cfg.get("radar", {})
    known = {f.name for f in RadarConfig.__dataclass_fields__.values()}
    return RadarConfig(**{k: v for k, v in r.items() if k in known})


def optim_from_config(cfg: Dict[str, Any]) -> OptimConfig:
    o = cfg.get("optimization", {})
    mp = cfg.get("multi_person", {})
    return OptimConfig(
        lr=float(o.get("lr", 1e-3)),
        betas=tuple(o.get("betas", [0.9, 0.999])),
        n_iter=int(o.get("n_iter", 100)),
        w_em=float(o.get("w_em", 0.5)),
        w_kine=float(o.get("w_kine", 0.5)),
        tau_pct=float(o.get("tau_pct", 10.0)),
        bone_tolerance=float(o.get("bone_tolerance", 0.05)),
        theta_max_deg=float(o.get("theta_max_deg", 170.0)),
        d_sep=float(mp.get("d_sep", 0.5)),
        d_joint=float(mp.get("d_joint", 0.1)),
    )
