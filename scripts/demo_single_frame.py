#!/usr/bin/env python3
"""Single-frame MHP demo without external datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pppr.config import load_config, optim_from_config, radar_from_config
from pppr.mhp import MHPOptimizer
from pppr.reconstruction import pppr_to_heatmap, pppr_to_pointcloud
from pppr.skeleton import SKELETON_CONNECTIONS, t_pose_positions
from pppr.utils import ensure_dir, majpe, set_seed, synthesize_heatmap_from_joints


def parse_args():
    p = argparse.ArgumentParser(description="Demo: optimize PPPR on a synthetic Heatmap")
    p.add_argument("--dataset", type=str, default="MMVR")
    p.add_argument("--n_iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="outputs/demo")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cfg = load_config(dataset=args.dataset)
    radar = radar_from_config(cfg)
    optim = optim_from_config(cfg)
    optim.n_iter = args.n_iter

    gt = t_pose_positions(distance=1.8).numpy()
    gt = gt + np.random.randn(*gt.shape).astype(np.float32) * 0.02
    heatmap = synthesize_heatmap_from_joints(
        gt,
        range_bins=radar.range_bins,
        angle_bins=radar.angle_bins,
        max_range=radar.max_range,
        fov_h_deg=radar.fov_h_deg,
    )

    mhp = MHPOptimizer(radar=radar, optim=optim, device=device)
    result = mhp.optimize(heatmap)
    pred = result.positions()[0]
    err = majpe(pred, gt)

    h_sim = result.h_sim.cpu().numpy()
    h_pppr = pppr_to_heatmap(result.instances, radar, device=device).cpu().numpy()
    pc, _ = pppr_to_pointcloud(result.instances, radar, device=device)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(heatmap, aspect="auto", origin="lower", cmap="hot")
    axes[0].set_title("H_ori")
    axes[1].imshow(h_sim, aspect="auto", origin="lower", cmap="hot")
    axes[1].set_title("H_sim")
    axes[2].imshow(h_pppr, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title("PPPR-Heatmap")
    fig.suptitle(f"MHP demo | MAJPE={err:.1f} mm | converged={result.converged}")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmaps.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="g", label="GT")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="r", label="PPPR")
    for a, b in SKELETON_CONNECTIONS:
        ax.plot(*zip(pred[a], pred[b]), c="r", alpha=0.6)
        ax.plot(*zip(gt[a], gt[b]), c="g", alpha=0.4)
    if len(pc):
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="b", s=5, alpha=0.3, label="PPPR-PC")
    ax.legend()
    ax.set_title("Optimized PPPR skeleton")
    fig.savefig(out_dir / "skeleton.png", dpi=150)
    plt.close(fig)

    np.savez(
        out_dir / "demo_result.npz",
        heatmap=heatmap,
        h_sim=h_sim,
        positions=pred,
        gt=gt,
        losses=result.losses,
    )
    print(f"[demo] MAJPE={err:.2f} mm  losses={result.losses}")
    print(f"[demo] saved outputs → {out_dir}")


if __name__ == "__main__":
    main()
