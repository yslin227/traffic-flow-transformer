"""
Model Architecture
Responsibilities: Define the Spatial-Temporal Transformer with
decoupled attention heads, graph bias, and learnable positional encoding.

Run (smoke test, no data needed):
    python part2_model.py
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# Learnable positional encoding
class LearnablePE(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe      = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self._sinusoidal_init(seq_len, d_model)

    def _sinusoidal_init(self, L: int, D: int):
        pos      = torch.arange(L).unsqueeze(1).float()
        div      = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe       = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:D // 2])
        with torch.no_grad():
            self.pe.weight.copy_(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.dropout(x + self.pe(torch.arange(x.size(1), device=x.device)))


# ── Graph-aware spatial bias ──────────────────────────────────────────────────
class GraphBias(nn.Module):
    """Learnable [N, N] additive bias on spatial attention logits."""
    def __init__(self, num_sensors: int, adj: Optional[torch.Tensor] = None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_sensors, num_sensors))
        if adj is not None:
            with torch.no_grad():
                self.bias.copy_((adj.float() * 2 - 1) * 0.5)

    def forward(self) -> torch.Tensor:
        return self.bias


# Core encoder block 
class STBlock(nn.Module):
    """
    Decoupled Spatial-Temporal Attention block.
      Temporal: each sensor attends over time  → [B*N, T, D]
      Spatial : each timestep attends over sensors (+ graph bias) → [B*T, N, D]
      FFN     : shared feed-forward + LayerNorm
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_sensors: int,
        ff_dim: int,
        dropout: float,
        adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.t_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_t  = nn.LayerNorm(d_model)
        self.s_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_s  = nn.LayerNorm(d_model)
        self.g_bias  = GraphBias(num_sensors, adj)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        B, T, N, D = x.shape

        # Temporal attention
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt, _ = self.t_attn(xt, xt, xt)
        x = self.norm_t(x + self.drop(xt).reshape(B, N, T, D).permute(0, 2, 1, 3))

        # Spatial attention with graph bias
        xs = x.reshape(B * T, N, D)
        xs, _ = self.s_attn(xs, xs, xs, attn_mask=self.g_bias())
        x = self.norm_s(x + self.drop(xs).reshape(B, T, N, D))

        # Feed-forward
        x = self.norm_ff(x + self.drop(self.ff(x)))
        return x


# Full model 
class STTransformer(nn.Module):
    """
    Spatial-Temporal Transformer for traffic flow forecasting.

    Input  : [B, T, N]  (normalised speed / flow)
    Output : [B, M, N]  (predicted M future steps)
    """
    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        pred_len: int,
        d_model: int      = 64,
        num_heads: int    = 4,
        num_layers: int   = 3,
        ff_dim: int       = 256,
        dropout: float    = 0.1,
        adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.node_emb   = nn.Embedding(num_sensors, d_model)
        self.pos_enc    = LearnablePE(seq_len, d_model, dropout)
        self.encoder    = nn.ModuleList([
            STBlock(d_model, num_heads, num_sensors, ff_dim, dropout, adj)
            for _ in range(num_layers)
        ])
        self.out_proj   = nn.Linear(d_model, pred_len)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))           # [B, T, N, D]
        h = h + self.node_emb(torch.arange(N, device=x.device))
        h_pe = self.pos_enc(h.mean(2))                 # [B, T, D]
        h = h + h_pe.unsqueeze(2)
        for block in self.encoder:
            h = block(h)
        out = self.out_proj(h.mean(1))                 # [B, N, M]
        return out.permute(0, 2, 1)                    # [B, M, N]


# Smoke test
if __name__ == "__main__":
    torch.manual_seed(42)
    model = STTransformer(num_sensors=207, seq_len=12, pred_len=12)
    x     = torch.randn(8, 12, 207)
    y     = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Params: {params:,}")
    assert y.shape == (8, 12, 207), "Shape mismatch!"
    print("Part 2 OK — model architecture verified.")
