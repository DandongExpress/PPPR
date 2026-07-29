"""Unit / smoke tests for the MHP pipeline (no external datasets required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pppr.config import OptimConfig, RadarConfig, load_config, radar_from_config
from pppr.initialization import Initializer
from pppr.losses import electromagnetic_loss, kinematic_loss, total_loss
from pppr.mhp import MHPOptimizer
from pppr.radar_simulation import RadarSimulator
from pppr.reconstruction import pppr_to_heatmap, pppr_to_pointcloud
from pppr.representation import PPPRInstance, pack_pppr, unpack_pppr
from pppr.skeleton import NUM_JOINTS, t_pose_positions
from pppr.utils import majpe, synthesize_heatmap_from_joints


def test_pack_unpack():
    inst = PPPRInstance(t_pose_positions())
    vec = pack_pppr(inst)
    assert vec.numel() == NUM_JOINTS * 22
    restored = unpack_pppr(vec)
    assert torch.allclose(restored.positions, inst.positions, atol=1e-5)


def test_radar_simulate_shape():
    radar = RadarConfig(range_bins=64, angle_bins=32, max_range=4.0)
    sim = RadarSimulator(radar, device=torch.device("cpu"))
    inst = PPPRInstance(t_pose_positions(distance=1.5))
    h = sim.simulate(inst)
    assert h.shape == (64, 32)
    assert torch.isfinite(h).all()


def test_losses_finite():
    radar = RadarConfig(range_bins=64, angle_bins=32)
    inst = PPPRInstance(t_pose_positions())
    cfg = OptimConfig()
    lk = kinematic_loss(inst, cfg)
    sim = RadarSimulator(radar)
    h_sim = sim.simulate(inst)
    h_ori = h_sim.detach() + 0.01 * torch.randn_like(h_sim)
    lem = electromagnetic_loss(h_sim, h_ori, tau_pct=10.0)
    lt = total_loss(lem, lk, 0.5, 0.5)
    assert torch.isfinite(lk) and torch.isfinite(lem) and torch.isfinite(lt)


def test_mhp_optimize_smoke():
    cfg = load_config(dataset="MMVR")
    radar = radar_from_config(cfg)
    # Smaller grids for speed
    radar.range_bins = 64
    radar.angle_bins = 32
    optim = OptimConfig(n_iter=5, lr=1e-2, w_em=0.5, w_kine=0.5)
    gt = t_pose_positions(distance=1.6).numpy()
    heatmap = synthesize_heatmap_from_joints(
        gt,
        range_bins=radar.range_bins,
        angle_bins=radar.angle_bins,
        max_range=radar.max_range,
        noise_level=0.02,
    )
    mhp = MHPOptimizer(radar=radar, optim=optim, device=torch.device("cpu"))
    result = mhp.optimize(heatmap)
    assert result.n_person == 1
    assert len(result.instances) == 1
    assert result.h_sim.shape[0] == radar.range_bins
    pred = result.positions()[0]
    err = majpe(pred, gt)
    print(f"  smoke MAJPE={err:.1f} mm losses={result.losses}")
    assert np.isfinite(err)


def test_reconstruction():
    radar = RadarConfig(range_bins=64, angle_bins=32)
    inst = PPPRInstance(t_pose_positions())
    hm = pppr_to_heatmap(inst, radar)
    pts, ints = pppr_to_pointcloud(inst, radar)
    assert hm.ndim == 2
    assert pts.ndim == 2 and pts.shape[1] == 3


def test_initializer():
    radar = RadarConfig(range_bins=64, angle_bins=32)
    gt = t_pose_positions(distance=1.5).numpy()
    h = synthesize_heatmap_from_joints(
        gt, range_bins=64, angle_bins=32, max_range=radar.max_range, noise_level=0.0
    )
    init = Initializer(radar)
    pos, vel = init(h)
    assert pos.shape == (NUM_JOINTS, 3)
    assert vel.shape == (NUM_JOINTS, 3)


def test_model_forward():
    from models import build_model

    for name in ("RETR", "HuprModel", "mmDiff", "PoseformerV2", "MLP"):
        model = build_model(name, input_type="pppr")
        packed = pack_pppr(PPPRInstance(t_pose_positions())).unsqueeze(0)
        out = model({"pppr": packed})
        assert out.shape == (1, NUM_JOINTS, 3)


if __name__ == "__main__":
    tests = [
        test_pack_unpack,
        test_radar_simulate_shape,
        test_losses_finite,
        test_initializer,
        test_reconstruction,
        test_model_forward,
        test_mhp_optimize_smoke,
    ]
    for fn in tests:
        print(f"RUN {fn.__name__}")
        fn()
        print(f"  OK")
    print("All tests passed.")
