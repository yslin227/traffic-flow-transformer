

"""
train_multihop.py  —  Training script for Multi-Hop ST-Transformer
===================================================================
Fixed issues vs Transformer_model.py:
  1. No hardcoded Colab path
  2. Loads real adjacency matrix for multi-hop graph bias
  3. Saves checkpoint to local dir (works on HPC)

Run on HPC:
    python train_multihop.py --dataset METR-LA --output_dir ./processed
    python train_multihop.py --dataset PEMS-BAY --output_dir ./processed
"""

import argparse
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ST_multihop import STTransformerMultiHop


# Dataset
class TrafficDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.Y = torch.tensor(np.load(y_path), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def make_loader(base, split, batch_size, shuffle):
    ds = TrafficDataset(f"{base}_{split}_X.npy", f"{base}_{split}_Y.npy")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def masked_mae(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    return (mask * (pred - target).abs()).sum() / mask.sum().clamp(min=1)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best = float("inf")
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


def load_adj(base, tag):
    """
    Try to load adjacency matrix from standard locations.
    Returns a torch.FloatTensor [N, N] or None.
    """
    candidates = [
        f"{base}_adj.pkl",
        f"./processed/{tag}_adj.pkl",
        f"./data/adj_mx.pkl",
        f"./adj_mx.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
            # Handle different formats
            if isinstance(obj, tuple):         # DCRNN format: (ids, ids, matrix)
                adj = torch.tensor(obj[2], dtype=torch.float32)
            elif isinstance(obj, np.ndarray):
                adj = torch.tensor(obj, dtype=torch.float32)
            elif isinstance(obj, torch.Tensor):
                adj = obj.float()
            else:
                continue
            print(f"  Loaded adjacency matrix from {path}  shape={tuple(adj.shape)}")
            return adj
    print("  WARNING: No adjacency matrix found — graph bias will be learnable only (no topology)")
    return None


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag    = cfg["dataset"].replace("-", "_").lower()
    base   = f"{cfg['output_dir']}/{tag}"

    # ── checkpoint path (local, no Colab) ────────────────────────────────────
    ckpt_path = f"best_{tag}_multihop.pt"

    # ── loaders ──────────────────────────────────────────────────────────────
    train_loader = make_loader(base, "train", cfg["batch_size"], shuffle=True)
    val_loader   = make_loader(base, "val",   cfg["batch_size"], shuffle=False)
    num_sensors  = train_loader.dataset.X.shape[2]

    # ── adjacency matrix ─────────────────────────────────────────────────────
    adj = load_adj(base, tag)
    if adj is not None:
        adj = adj.to(device)

    # ── model ────────────────────────────────────────────────────────────────
    model = STTransformerMultiHop(
        num_sensors=num_sensors,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        adj=adj,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*60}")
    print(f"  Model   : ST-Transformer Multi-Hop")
    print(f"  Dataset : {cfg['dataset']}  |  Sensors: {num_sensors}")
    print(f"  Device  : {device}  |  Params: {params:,}")
    print(f"  Adj     : {'3-hop loaded' if adj is not None else 'learnable only'}")
    print(f"{'─'*60}")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])
    stopper   = EarlyStopping(patience=cfg["patience"])

    start_epoch = 1
    # Resume if checkpoint exists (safe — no hardcoded path)
    if os.path.exists(ckpt_path):
        print(f"\n  Resuming from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        start_epoch = ck["epoch"] + 1
        stopper.best = ck["val_loss"]
        print(f"  Resumed from epoch {ck['epoch']}  val_MAE={ck['val_loss']:.4f}")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = masked_mae(model(X), Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                val_loss += masked_mae(model(X.to(device)), Y.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4}  train={tr_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss <= stopper.best:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss, "cfg": cfg}, ckpt_path)

        if stopper.step(val_loss):
            print(f"\n  Early stop at epoch {epoch}.  Best val MAE: {stopper.best:.4f}")
            break

    print(f"\n  Saved → {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="METR-LA")
    p.add_argument("--output_dir",   default="./processed")
    p.add_argument("--seq_len",      type=int,   default=12)
    p.add_argument("--pred_len",     type=int,   default=12)
    p.add_argument("--d_model",      type=int,   default=64)
    p.add_argument("--num_heads",    type=int,   default=4)
    p.add_argument("--num_layers",   type=int,   default=3)
    p.add_argument("--ff_dim",       type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--clip_grad",    type=float, default=5.0)
    p.add_argument("--patience",     type=int,   default=15)
    args = p.parse_args()
    train(vars(args))
