#!/usr/bin/env python3
"""Evaluate an HPE checkpoint (within-dataset or cross-dataset).

Usage (matches GitHub README)::

    python evaluate.py \\
        --model RETR \\
        --dataset MMVR \\
        --input_type pppr \\
        --checkpoint checkpoints/retr_mmvr_pppr/best.pth

    python evaluate.py \\
        --model RETR \\
        --train_dataset MMVR \\
        --test_dataset XRF55 \\
        --input_type pppr \\
        --checkpoint checkpoints/retr_mmvr_pppr/best.pth \\
        --cross_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model, list_models
from pppr.utils import majpe, pa_majpe, set_seed
from train import PPPRTrainDataset, collate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HPE model")
    p.add_argument("--model", type=str, default="RETR")
    p.add_argument("--dataset", type=str, default="MMVR", help="Test dataset (within-dataset mode)")
    p.add_argument("--train_dataset", type=str, default=None)
    p.add_argument("--test_dataset", type=str, default=None)
    p.add_argument(
        "--input_type",
        type=str,
        default="pppr",
        choices=["heatmap", "pc", "pppr", "pppr_heatmap", "pppr_pc"],
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cross_dataset", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None, help="Optional JSON metrics path")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_dataset = args.test_dataset or args.dataset
    if args.cross_dataset:
        if not args.test_dataset:
            raise ValueError("--cross_dataset requires --test_dataset")
        test_dataset = args.test_dataset
        print(
            f"[evaluate] cross-dataset: train={args.train_dataset or 'CKPT'} → test={test_dataset}"
        )

    data_root = Path(args.data_root or ROOT / "data" / f"{test_dataset}_pppr")
    if not data_root.exists() or not any(data_root.rglob("*.npz")):
        raise FileNotFoundError(
            f"No prepared PPPR data at {data_root}. "
            f"Run: python prepare_pppr.py --dataset {test_dataset} --synthetic"
        )

    # Prefer requested split; fall back to val/train
    split = args.split
    try:
        ds = PPPRTrainDataset(str(data_root), split=split, input_type=args.input_type)
    except FileNotFoundError:
        for alt in ("val", "train"):
            try:
                ds = PPPRTrainDataset(str(data_root), split=alt, input_type=args.input_type)
                split = alt
                break
            except FileNotFoundError:
                continue
        else:
            raise

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model", args.model)
    input_type = ckpt.get("input_type", args.input_type)
    model = build_model(model_name, input_type=input_type).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    errs, pa_errs = [], []
    with torch.no_grad():
        for batch in loader:
            batch_d = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            pred = model(batch_d)
            gt = batch_d["joints"]
            for i in range(pred.shape[0]):
                errs.append(majpe(pred[i].cpu().numpy(), gt[i].cpu().numpy()))
                pa_errs.append(pa_majpe(pred[i].cpu().numpy(), gt[i].cpu().numpy()))

    metrics = {
        "model": model_name,
        "input_type": input_type,
        "test_dataset": test_dataset,
        "split": split,
        "n_frames": len(errs),
        "MAJPE": float(np.mean(errs)),
        "PA-MAJPE": float(np.mean(pa_errs)),
        "checkpoint": str(args.checkpoint),
        "cross_dataset": bool(args.cross_dataset),
    }
    print(
        f"[evaluate] {model_name} / {input_type} on {test_dataset} ({split}, n={metrics['n_frames']})"
    )
    print(f"  MAJPE    = {metrics['MAJPE']:.2f} mm")
    print(f"  PA-MAJPE = {metrics['PA-MAJPE']:.2f} mm")

    out = args.output or str(Path(args.checkpoint).with_suffix(".eval.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] wrote {out}")


if __name__ == "__main__":
    main()
