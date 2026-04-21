

"""
eval_multihop.py  —  Evaluation for ST-Transformer Multi-Hop
=============================================================
Run:
    python eval_multihop.py

Files needed in same folder:
    - best_metr_la_multihop.pt
    - metr_la_test_X.npy
    - metr_la_test_Y.npy
    - metr_la_scaler.pkl
    - ST_multihop.py
"""

import math, pickle, numpy as np, torch, torch.nn as nn
from typing import Optional


# ── Copy of MultiHopGraphBias + model from ST_multihop.py ────────────────────

class MultiHopGraphBias(nn.Module):
    def __init__(self, num_sensors, adj=None):
        super().__init__()
        if adj is not None:
            A1 = (adj > 0).float()
            A1.fill_diagonal_(1)
            A2 = torch.clamp(A1 @ A1, 0, 1)
            A3 = torch.clamp(A2 @ A1, 0, 1)
            self.register_buffer("A1", A1)
            self.register_buffer("A2", A2)
            self.register_buffer("A3", A3)
            self.use_multihop = True
            self.w1 = nn.Parameter(torch.tensor(1.0))
            self.w2 = nn.Parameter(torch.tensor(0.5))
            self.w3 = nn.Parameter(torch.tensor(0.25))
        else:
            self.bias = nn.Parameter(torch.zeros(num_sensors, num_sensors))
            self.use_multihop = False

    def forward(self):
        if self.use_multihop:
            return self.w1 * self.A1 + self.w2 * self.A2 + self.w3 * self.A3
        return self.bias


class LearnablePE(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        with torch.no_grad():
            self.pe.weight.copy_(pe)

    def forward(self, x):
        return self.dropout(x + self.pe(torch.arange(x.size(1), device=x.device)))


class STBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_sensors, ff_dim, dropout, adj=None):
        super().__init__()
        self.t_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_t  = nn.LayerNorm(d_model)
        self.s_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_s  = nn.LayerNorm(d_model)
        self.g_bias  = MultiHopGraphBias(num_sensors, adj)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt, _ = self.t_attn(xt, xt, xt)
        x = self.norm_t(x + self.drop(xt).reshape(B, N, T, D).permute(0, 2, 1, 3))
        xs = x.reshape(B * T, N, D)
        xs, _ = self.s_attn(xs, xs, xs, attn_mask=self.g_bias())
        x = self.norm_s(x + self.drop(xs).reshape(B, T, N, D))
        x = self.norm_ff(x + self.drop(self.ff(x)))
        return x


class STTransformerMultiHop(nn.Module):
    def __init__(self, num_sensors, seq_len, pred_len,
                 d_model=64, num_heads=4, num_layers=3,
                 ff_dim=256, dropout=0.1, adj=None):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.node_emb   = nn.Embedding(num_sensors, d_model)
        self.pos_enc    = LearnablePE(seq_len, d_model, dropout)
        self.encoder    = nn.ModuleList([
            STBlock(d_model, num_heads, num_sensors, ff_dim, dropout, adj)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        B, T, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))
        h = h + self.node_emb(torch.arange(N, device=x.device))
        h_pe = self.pos_enc(h.mean(2))
        h = h + h_pe.unsqueeze(2)
        for block in self.encoder:
            h = block(h)
        out = self.out_proj(h.mean(1))
        return out.permute(0, 2, 1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def masked_mae(p, t, eps=1e-4):
    m = np.abs(t) > eps
    return float(np.mean(np.abs(p[m] - t[m])))

def masked_rmse(p, t, eps=1e-4):
    m = np.abs(t) > eps
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))

def masked_mape(p, t, eps=1e-4):
    m = np.abs(t) > eps
    return float(np.mean(np.abs((p[m] - t[m]) / t[m])) * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load scaler
    with open("metr_la_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    mean_arr = scaler.mean_    # (207,)
    std_arr  = scaler.scale_   # (207,)

    # Load test data
    X_test = np.load("metr_la_test_X.npy")
    Y_test = np.load("metr_la_test_Y.npy")
    print(f"Test X: {X_test.shape}   Test Y: {Y_test.shape}")

    # Load adjacency
    with open("metr_la_adj.pkl", "rb") as f:
        adj_obj = pickle.load(f)
    if isinstance(adj_obj, list):
        adj = torch.tensor(adj_obj[2], dtype=torch.float32).to(device)
    elif isinstance(adj_obj, np.ndarray):
        adj = torch.tensor(adj_obj, dtype=torch.float32).to(device)
    else:
        adj = adj_obj.float().to(device)
    print(f"Adj shape: {adj.shape}")

    # Load checkpoint
    ck  = torch.load("best_metr_la_multihop.pt", map_location=device)
    cfg = ck["cfg"]
    print(f"Checkpoint — epoch {ck['epoch']}  val_MAE(norm)={ck['val_loss']:.4f}")

    # Build model
    model = STTransformerMultiHop(
        num_sensors=207,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=0.0,
        adj=adj,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("Model loaded OK")

    # Inference
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 64):
            xb = X_tensor[i:i+64].to(device)
            all_preds.append(model(xb).cpu().numpy())

    preds_norm   = np.concatenate(all_preds, axis=0)   # (N, 12, 207)
    targets_norm = Y_test                               # (N, 12, 207)

    # Inverse normalise (per-sensor)
    preds   = preds_norm   * std_arr[None, None, :] + mean_arr[None, None, :]
    targets = targets_norm * std_arr[None, None, :] + mean_arr[None, None, :]
    print(f"Targets: {targets.min():.1f} ~ {targets.max():.1f} mph")

    # Print results
    print(f"\n{'═'*58}")
    print(f"  ST-Transformer Multi-Hop (3-hop)  [METR-LA]")
    print(f"{'═'*58}")
    print(f"  {'Horizon':>10}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>8}")
    print(f"  {'─'*44}")
    for step, label in [(2, "15 min"), (5, "30 min"), (11, "60 min")]:
        p, t = preds[:, step, :], targets[:, step, :]
        print(f"  {label:>10}  {masked_mae(p,t):>8.4f}  "
              f"{masked_rmse(p,t):>8.4f}  {masked_mape(p,t):>7.2f}%")
    print(f"  {'─'*44}")
    print(f"  {'Overall':>10}  {masked_mae(preds,targets):>8.4f}  "
          f"{masked_rmse(preds,targets):>8.4f}  {masked_mape(preds,targets):>7.2f}%")
    print(f"{'═'*58}")

    print(f"\n  Compare with ST-Transformer (1-hop, Mohan):")
    print(f"  15min MAE=3.3208  30min MAE=4.0521  60min MAE=5.0573")


if __name__ == "__main__":
    main()
