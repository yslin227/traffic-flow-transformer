"""
Evaluation & Metrics
Responsibilities: Load best checkpoint, evaluate MAE / RMSE / MAPE
on the test set, compare with LSTM baseline, print per-horizon breakdown.

Run:
    python eval_STTransformer.py --dataset METR-LA
    python eval_STTransformer.py --dataset PEMS-BAY
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from part2_model import STTransformer
from part3_train import TrafficDataset, make_loader


# Metrics 
def mae(pred, target, null=0.0):
    mask = (target != null)
    return np.abs(pred[mask] - target[mask]).mean()

def rmse(pred, target, null=0.0):
    mask = (target != null)
    return np.sqrt(((pred[mask] - target[mask]) ** 2).mean())

def mape(pred, target, null=0.0, eps=1e-8):
    mask = (np.abs(target) > eps) & (target != null)
    return (np.abs((pred[mask] - target[mask]) / (np.abs(target[mask]) + eps))).mean() * 100


# LSTM baseline 
class LSTMBaseline(nn.Module):
    def __init__(self, num_sensors: int, hidden: int, pred_len: int):
        super().__init__()
        self.lstm    = nn.LSTM(num_sensors, hidden, batch_first=True, num_layers=2,
                               dropout=0.1)
        self.out     = nn.Linear(hidden, num_sensors * pred_len)
        self.pred_len = pred_len
        self.N = num_sensors

    def forward(self, x):      # x: [B, T, N]
        h, _ = self.lstm(x)
        return self.out(h[:, -1]).view(x.size(0), self.pred_len, self.N)


def train_lstm_baseline(train_loader, val_loader, num_sensors, pred_len, device,
                         epochs=30, lr=1e-3):
    model = LSTMBaseline(num_sensors, hidden=64, pred_len=pred_len).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            loss = nn.L1Loss()(model(X), Y)
            loss.backward(); opt.step()
        model.eval()
        vl = sum(nn.L1Loss()(model(X.to(device)), Y.to(device)).item()
                 for X, Y in val_loader) / len(val_loader)
        if vl < best_val:
            best_val = vl
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"    LSTM epoch {ep:>3}  val MAE={vl:.4f}")
    model.load_state_dict(best_sd)
    return model


# Collect predictions 
def collect(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, Y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(Y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# Inverse transform 
def inverse(arr, scaler):
    B, M, N = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, N)).reshape(B, M, N)


# Per-horizon table ─────────────────────────────────────────────────────────
def horizon_table(pred, target, steps=(3, 6, 12)):
    print(f"\n  {'Horizon':>10}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>8}")
    print(f"  {'─'*44}")
    for s in steps:
        if s <= pred.shape[1]:
            m = mae(pred[:, :s],  target[:, :s])
            r = rmse(pred[:, :s], target[:, :s])
            p = mape(pred[:, :s], target[:, :s])
            mins = s * 5
            print(f"  {mins:>8} min  {m:>8.4f}  {r:>8.4f}  {p:>7.2f}%")


# Main evaluation 
def evaluate(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag    = cfg["dataset"].replace("-", "_").lower()
    base   = f"{cfg['output_dir']}/{tag}"
    ckpt   = f"best_{tag}.pt"

    # Load scaler
    with open(f"{base}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load checkpoint
    checkpoint  = torch.load(ckpt, map_location=device)
    saved_cfg   = checkpoint["cfg"]
    num_sensors = np.load(f"{base}_test_X.npy").shape[2]

    model = STTransformer(
        num_sensors=num_sensors,
        seq_len=saved_cfg["seq_len"],
        pred_len=saved_cfg["pred_len"],
        d_model=saved_cfg["d_model"],
        num_heads=saved_cfg["num_heads"],
        num_layers=saved_cfg["num_layers"],
        ff_dim=saved_cfg["ff_dim"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"\n  Loaded checkpoint (epoch {checkpoint['epoch']}, "
          f"val MAE={checkpoint['val_loss']:.4f})")

    # Data loaders
    test_loader  = make_loader(base, "test",  cfg["batch_size"], shuffle=False)
    train_loader = make_loader(base, "train", cfg["batch_size"], shuffle=True)
    val_loader   = make_loader(base, "val",   cfg["batch_size"], shuffle=False)

    # Transformer results 
    pred_n, tgt_n = collect(model, test_loader, device)
    pred  = inverse(pred_n, scaler)
    tgt   = inverse(tgt_n,  scaler)

    print(f"\n{'─'*55}")
    print(f"  ST-Transformer  [{cfg['dataset']}]")
    print(f"  Overall  MAE={mae(pred,tgt):.4f}  RMSE={rmse(pred,tgt):.4f}  "
          f"MAPE={mape(pred,tgt):.2f}%")
    horizon_table(pred, tgt)

    # LSTM baseline results
    print(f"\n{'─'*55}")
    print(f"  Training LSTM baseline (30 epochs)...")
    lstm  = train_lstm_baseline(train_loader, val_loader, num_sensors,
                                 saved_cfg["pred_len"], device)
    lp_n, lt_n = collect(lstm, test_loader, device)
    lp   = inverse(lp_n, scaler)
    lt   = inverse(lt_n, scaler)

    print(f"\n  LSTM baseline  [{cfg['dataset']}]")
    print(f"  Overall  MAE={mae(lp,lt):.4f}  RMSE={rmse(lp,lt):.4f}  "
          f"MAPE={mape(lp,lt):.2f}%")
    horizon_table(lp, lt)

    # Improvement summary
    print(f"\n{'─'*55}")
    imp_mae  = (mae(lp,lt)  - mae(pred,tgt))  / mae(lp,lt)  * 100
    imp_rmse = (rmse(lp,lt) - rmse(pred,tgt)) / rmse(lp,lt) * 100
    print(f"  Improvement over LSTM — MAE: {imp_mae:+.1f}%  RMSE: {imp_rmse:+.1f}%")
    print(f"{'─'*55}\n")


# CLI 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="METR-LA")
    p.add_argument("--output_dir", default="./processed")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    evaluate(vars(args))
