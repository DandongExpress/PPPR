#!/usr/bin/env python3
"""Train an HPE model with Heatmap / PC / PPPR inputs.

Usage (matches GitHub README)::

    python train.py \\
        --model RETR \\
        --dataset MMVR \\
        --input_type pppr \\
        --data_root data/MMVR_pppr \\
        --output_dir checkpoints/retr_mmvr_pppr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model, list_models
from pppr.skeleton import NUM_JOINTS
from pppr.utils import ensure_dir, majpe, pa_majpe, set_seed


class PPPRTrainDataset(Dataset):
    """Load prepared PPPR NPZ frames written by ``prepare_pppr.py``."""

    def __init__(self, root: str, split: str = "train", input_type: str = "pppr"):
        self.root = Path(root)
        self.split = split
        self.input_type = input_type.lower()
        split_dir = self.root / split
        search = split_dir if split_dir.exists() else self.root
        self.files = sorted(search.rglob("*.npz"))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No NPZ files under {search}. Run prepare_pppr.py first "
                f"(or pass --synthetic to train.py)."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = np.load(self.files[idx], allow_pickle=True)
        joints = np.asarray(data["joints"], dtype=np.float32)
        if joints.ndim == 3:
            joints = joints[0]  # take first person for single-person training
        sample: Dict[str, Any] = {
            "joints": torch.from_numpy(joints),
            "id": self.files[idx].stem,
        }
        t = self.input_type
        if t == "pppr":
            pppr = np.asarray(data["pppr"], dtype=np.float32)
            if pppr.ndim == 2:
                pppr = pppr[0]
            sample["pppr"] = torch.from_numpy(pppr.reshape(-1))
        elif t in ("heatmap", "pppr_heatmap"):
            key = "pppr_heatmap" if t == "pppr_heatmap" and "pppr_heatmap" in data.files else "heatmap"
            hm = np.asarray(data[key], dtype=np.float32)
            if hm.ndim == 3:
                hm = hm.max(axis=-1)
            sample["heatmap"] = torch.from_numpy(hm)
            if "pppr_heatmap" in data.files:
                ph = np.asarray(data["pppr_heatmap"], dtype=np.float32)
                if ph.ndim == 3:
                    ph = ph.max(axis=-1)
                sample["pppr_heatmap"] = torch.from_numpy(ph)
        elif t in ("pc", "pppr_pc"):
            key = "pppr_pc" if t == "pppr_pc" and "pppr_pc" in data.files else "pc"
            pts = np.asarray(data[key], dtype=np.float32)
            sample["pc"] = torch.from_numpy(self._pad_pc(pts))
            if "pppr_pc" in data.files:
                sample["pppr_pc"] = torch.from_numpy(self._pad_pc(np.asarray(data["pppr_pc"], dtype=np.float32)))
        else:
            raise ValueError(f"Unsupported input_type: {t}")
        return sample

    @staticmethod
    def _pad_pc(pts: np.ndarray, n: int = 256) -> np.ndarray:
        if pts.ndim != 2 or pts.shape[-1] < 3:
            return np.zeros((n, 3), dtype=np.float32)
        pts = pts[:, :3]
        if len(pts) >= n:
            return pts[:n]
        pad = np.zeros((n - len(pts), 3), dtype=np.float32)
        return np.concatenate([pts, pad], axis=0)


def collate(batch):
    out: Dict[str, Any] = {"ids": [b["id"] for b in batch]}
    out["joints"] = torch.stack([b["joints"] for b in batch])
    for key in ("pppr", "heatmap", "pppr_heatmap", "pc", "pppr_pc"):
        if key in batch[0]:
            if key in ("heatmap", "pppr_heatmap"):
                # Pad to common spatial size
                max_r = max(b[key].shape[0] for b in batch)
                max_a = max(b[key].shape[1] for b in batch)
                canvases = []
                for b in batch:
                    h = b[key]
                    c = torch.zeros(max_r, max_a)
                    c[: h.shape[0], : h.shape[1]] = h
                    canvases.append(c)
                out[key] = torch.stack(canvases)
            else:
                out[key] = torch.stack([b[key] for b in batch])
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train HPE model on PPPR / Heatmap / PC")
    p.add_argument("--model", type=str, default="RETR", choices=list_models() + [m.upper() for m in list_models()])
    p.add_argument("--dataset", type=str, default="MMVR")
    p.add_argument(
        "--input_type",
        type=str,
        default="pppr",
        choices=["heatmap", "pc", "pppr", "pppr_heatmap", "pppr_pc"],
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="Auto-run prepare_pppr if data missing")
    p.add_argument("--max_train", type=int, default=None)
    return p.parse_args()


def maybe_prepare_synthetic(args, data_root: Path):
    if data_root.exists() and any(data_root.rglob("*.npz")):
        return
    if not args.synthetic:
        print(
            f"[train] No data at {data_root}. Re-run with --synthetic "
            f"or run prepare_pppr.py first."
        )
        sys.exit(1)
    print("[train] Preparing synthetic PPPR data …")
    import subprocess

    cmd = [
        sys.executable,
        str(ROOT / "prepare_pppr.py"),
        "--dataset",
        args.dataset,
        "--output_root",
        str(data_root),
        "--synthetic",
        "--synthetic_size",
        "48",
        "--n_iter",
        "30",
        "--split",
        "all",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))


def evaluate_loader(model, loader, device):
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
                p = pred[i].cpu().numpy()
                g = gt[i].cpu().numpy()
                errs.append(majpe(p, g))
                pa_errs.append(pa_majpe(p, g))
    return float(np.mean(errs)), float(np.mean(pa_errs))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_root = Path(args.data_root or ROOT / "data" / f"{args.dataset}_pppr")
    output_dir = ensure_dir(
        args.output_dir or ROOT / "checkpoints" / f"{args.model.lower()}_{args.dataset.lower()}_{args.input_type}"
    )
    maybe_prepare_synthetic(args, data_root)

    train_set = PPPRTrainDataset(str(data_root), split="train", input_type=args.input_type)
    # Fall back: use train as val if val missing
    try:
        val_set = PPPRTrainDataset(str(data_root), split="val", input_type=args.input_type)
    except FileNotFoundError:
        val_set = train_set

    if args.max_train:
        train_set.files = train_set.files[: args.max_train]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = build_model(args.model, input_type=args.input_type).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best = float("inf")
    patience_left = args.patience
    history = []

    print(f"[train] model={args.model} input={args.input_type} device={device}")
    print(f"[train] train={len(train_set)} val={len(val_set)} → {output_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch_d = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            pred = model(batch_d)
            loss = F.mse_loss(pred, batch_d["joints"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        maj, pa = evaluate_loader(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "majpe": maj, "pa_majpe": pa}
        history.append(row)
        print(f"  epoch {epoch}: loss={row['loss']:.4f} MAJPE={maj:.2f} PA-MAJPE={pa:.2f}")

        ckpt = {
            "model": args.model,
            "input_type": args.input_type,
            "dataset": args.dataset,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": row,
        }
        torch.save(ckpt, output_dir / "last.pth")
        if maj < best:
            best = maj
            patience_left = args.patience
            torch.save(ckpt, output_dir / "best.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[train] early stopping")
                break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] best MAJPE={best:.2f} mm  checkpoint={output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
