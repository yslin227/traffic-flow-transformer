"""
Training Loop
========================
PyTorch training pipeline with data loading,
validation, early stopping, checkpointing, and LR scheduling.

Run:
    python part3_train.py --dataset METR-LA
    python part3_train.py --dataset PEMS-BAY --epochs 150 --lr 5e-4
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from STTmodel import STTransformer


# Dataset 
class TrafficDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.Y = torch.tensor(np.load(y_path), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def make_loader(base: str, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TrafficDataset(f"{base}_{split}_X.npy", f"{base}_{split}_Y.npy")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


# Loss (masked MAE) 
def masked_mae(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0):
    mask = (target != null_val).float()
    return (mask * (pred - target).abs()).sum() / mask.sum().clamp(min=1)


# Early stopping
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False          # keep going
        self.counter += 1
        return self.counter >= self.patience


# Training
def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag    = cfg["dataset"].replace("-", "_").lower()
    base   = f"{cfg['output_dir']}/{tag}"
    ckpt   = f"best_{tag}.pt"

    # Loaders
    train_loader = make_loader(base, "train", cfg["batch_size"], shuffle=True)
    val_loader   = make_loader(base, "val",   cfg["batch_size"], shuffle=False)

    # Infer num_sensors from data
    num_sensors = train_loader.dataset.X.shape[2]

    # Model
    model = STTransformer(
        num_sensors=num_sensors,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*55}")
    print(f"  Dataset : {cfg['dataset']}  |  Sensors: {num_sensors}")
    print(f"  Device  : {device}  |  Params: {params:,}")
    print(f"{'─'*55}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    stopper   = EarlyStopping(patience=cfg["patience"])

    for epoch in range(1, cfg["epochs"] + 1):

        # Train
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

        # Validate 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                val_loss += masked_mae(model(X.to(device)), Y.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4}  train={tr_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss <= stopper.best:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss, "cfg": cfg}, ckpt)

        if stopper.step(val_loss):
            print(f"\n  Early stop at epoch {epoch}. Best val MAE: {stopper.best:.4f}")
            break

    print(f"\n  Checkpoint saved → {ckpt}")
    print(f"  Next step: python part4_eval.py --dataset {cfg['dataset']}")


# CLI
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
