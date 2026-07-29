#!/usr/bin/env python3
"""Generate PPPR representations from mmWave Heatmaps (MHP pipeline).

Usage (matches GitHub README)::

    python prepare_pppr.py \\
        --dataset MMVR \\
        --data_root data/MMVR \\
        --output_root data/MMVR_pppr \\
        --w_em 0.5 \\
        --w_kine 0.5 \\
        --n_iter 100

    python prepare_pppr.py --dataset MMVR --multi_person --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Allow running from repo root without installation
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataset
from pppr.config import load_config, optim_from_config, radar_from_config
from pppr.mhp import MHPOptimizer
from pppr.reconstruction import heatmap_to_pointcloud, pppr_to_heatmap, pppr_to_pointcloud
from pppr.representation import pack_pppr
from pppr.utils import ensure_dir, save_pppr_npz, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Prepare PPPR via MmWave Human Parameterization (MHP)")
    p.add_argument("--dataset", type=str, default="MMVR", choices=["MMVR", "HuPR", "XRF55"])
    p.add_argument("--data_root", type=str, default=None, help="Raw dataset root (default: data/<DATASET>)")
    p.add_argument("--output_root", type=str, default=None, help="Output PPPR root (default: data/<DATASET>_pppr)")
    p.add_argument("--config", type=str, default=None, help="Optional YAML override")
    p.add_argument("--w_em", type=float, default=None)
    p.add_argument("--w_kine", type=float, default=None)
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--tau_pct", type=float, default=None)
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--multi_person", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic Heatmaps if data missing")
    p.add_argument("--synthetic_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--save_heatmap", action="store_true", default=True)
    p.add_argument("--save_pc", action="store_true", default=True)
    p.add_argument("--no_save_heatmap", action="store_true")
    p.add_argument("--no_save_pc", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root or str(ROOT / "data" / args.dataset)
    output_root = args.output_root or str(ROOT / "data" / f"{args.dataset}_pppr")
    ensure_dir(output_root)

    overrides = {"optimization": {}, "multi_person": {}}
    if args.w_em is not None:
        overrides["optimization"]["w_em"] = args.w_em
    if args.w_kine is not None:
        overrides["optimization"]["w_kine"] = args.w_kine
    if args.n_iter is not None:
        overrides["optimization"]["n_iter"] = args.n_iter
    if args.tau_pct is not None:
        overrides["optimization"]["tau_pct"] = args.tau_pct

    cfg = load_config(dataset=args.dataset, config_path=args.config, overrides=overrides)
    radar = radar_from_config(cfg)
    optim = optim_from_config(cfg)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[prepare_pppr] dataset={args.dataset} device={device}")
    print(f"[prepare_pppr] data_root={data_root}")
    print(f"[prepare_pppr] output_root={output_root}")
    print(f"[prepare_pppr] w_em={optim.w_em} w_kine={optim.w_kine} n_iter={optim.n_iter}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    mhp = MHPOptimizer(
        radar=radar,
        optim=optim,
        device=device,
        multi_person=args.multi_person,
    )

    meta = {
        "dataset": args.dataset,
        "radar": radar.to_dict(),
        "optim": {
            "w_em": optim.w_em,
            "w_kine": optim.w_kine,
            "n_iter": optim.n_iter,
            "tau_pct": optim.tau_pct,
        },
        "multi_person": args.multi_person,
    }
    with open(Path(output_root) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    save_hm = args.save_heatmap and not args.no_save_heatmap
    save_pc = args.save_pc and not args.no_save_pc

    total = 0
    for split in splits:
        ds = build_dataset(
            dataset=args.dataset,
            data_root=data_root,
            split=split,
            synthetic=args.synthetic,
            synthetic_size=args.synthetic_size,
        )
        out_split = ensure_dir(Path(output_root) / split)
        n = len(ds) if args.max_frames is None else min(len(ds), args.max_frames)
        print(f"[prepare_pppr] split={split} frames={n} synthetic={ds._synthetic}")

        warm = None
        for i in tqdm(range(n), desc=f"PPPR/{split}"):
            sample = ds[i]
            heatmap = sample["heatmap"]
            result = mhp.optimize(
                heatmap,
                doppler_volume=sample.get("doppler"),
                n_person=int(sample.get("n_person", 1)) if args.multi_person else 1,
                warm_start=warm,
            )
            warm = result.instances  # online warm-start for consecutive frames

            packed = pack_pppr(result.instances[0]).detach().cpu()
            positions = result.instances[0].positions.detach().cpu().numpy()

            # Multi-person: stack packed vectors
            if result.n_person > 1:
                packed_list = [pack_pppr(inst).detach().cpu().numpy() for inst in result.instances]
                packed_np = np.stack(packed_list, axis=0)
                positions = np.stack(
                    [inst.positions.detach().cpu().numpy() for inst in result.instances], axis=0
                )
            else:
                packed_np = packed.numpy()

            hm_pppr = None
            pc_pts = None
            if save_hm:
                hm = pppr_to_heatmap(result.instances, radar, device=device)
                hm_pppr = hm.detach().cpu().numpy()
            if save_pc:
                pc_pts, _ = pppr_to_pointcloud(result.instances, radar, device=device)

            # Also store original heatmap for training baselines
            out_path = out_split / f"{sample['id']}.npz"
            payload_meta = {
                "id": sample["id"],
                "n_person": result.n_person,
                "converged": bool(result.converged),
                "losses": result.losses,
                "has_gt": sample.get("joints") is not None,
            }
            save_pppr_npz(
                out_path,
                packed=torch.from_numpy(packed_np) if isinstance(packed_np, np.ndarray) else packed,
                positions=positions,
                heatmap_pppr=hm_pppr,
                pc_points=pc_pts,
                meta=payload_meta,
            )
            # Append original heatmap + GT joints into the same file
            existing = dict(np.load(out_path, allow_pickle=True))
            existing["heatmap"] = np.asarray(heatmap, dtype=np.float32)
            existing["joints"] = np.asarray(sample["joints"], dtype=np.float32)
            # Raw PC from original heatmap for PC baselines
            raw_pc, _ = heatmap_to_pointcloud(heatmap, radar)
            existing["pc"] = raw_pc
            np.savez_compressed(out_path, **existing)
            total += 1

    print(f"[prepare_pppr] done. wrote {total} frames → {output_root}")


if __name__ == "__main__":
    main()
