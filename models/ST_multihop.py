


"""
ST_multihop.py  —  Multi-Hop Spatial-Temporal Transformer
==========================================================
This model is identical to STTransformer.py except GraphBias
now uses 1-hop + 2-hop + 3-hop adjacency matrices with learnable
per-hop weights.

Usage (smoke test):
    python ST_multihop.py

Usage (training):
    python train_multihop.py --dataset METR-LA
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# CHANGED: Multi-Hop Graph Bias

class MultiHopGraphBias(nn.Module):
    """
    Additive bias on spatial attention logits using multi-hop adjacency.

    Given a binary adjacency matrix A (1-hop), computes:
        A^2  = clamp(A @ A, 0, 1)   — 2-hop neighbors
        A^3  = clamp(A^2 @ A, 0, 1) — 3-hop neighbors

    Final bias = w1*A + w2*A^2 + w3*A^3
    where w1, w2, w3 are learnable scalars (initialized so 1-hop dominates).

    If no adj is provided, falls back to a plain learnable [N,N] matrix
    (same behavior as Mohan's original GraphBias).
    """
    def __init__(self, num_sensors: int, adj: Optional[torch.Tensor] = None):
        super().__init__()

        if adj is not None:
            # Binarize adjacency (distance matrix → 0/1)
            A1 = (adj > 0).float()
            A1.fill_diagonal_(1)                          # self-loops

            A2 = torch.clamp(A1 @ A1, 0, 1)              # 2-hop
            A3 = torch.clamp(A2 @ A1, 0, 1)              # 3-hop

            # Register as buffers (not parameters — fixed topology)
            self.register_buffer("A1", A1)
            self.register_buffer("A2", A2)
            self.register_buffer("A3", A3)
            self.use_multihop = True

            # Learnable per-hop weights, init so 1-hop dominates
            self.w1 = nn.Parameter(torch.tensor(1.0))
            self.w2 = nn.Parameter(torch.tensor(0.5))
            self.w3 = nn.Parameter(torch.tensor(0.25))

        else:
            # Fallback: plain learnable bias (same as original)
            self.bias = nn.Parameter(torch.zeros(num_sensors, num_sensors))
            self.use_multihop = False

    def forward(self) -> torch.Tensor:
        if self.use_multihop:
            return self.w1 * self.A1 + self.w2 * self.A2 + self.w3 * self.A3
        else:
            return self.bias



# UNCHANGED here down


class LearnablePE(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe      = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self._sinusoidal_init(seq_len, d_model)

    def _sinusoidal_init(self, L: int, D: int):
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe  = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:D // 2])
        with torch.no_grad():
            self.pe.weight.copy_(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe(torch.arange(x.size(1), device=x.device)))


class STBlock(nn.Module):
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

        # ← Only this line changed: MultiHopGraphBias instead of GraphBias
        self.g_bias  = MultiHopGraphBias(num_sensors, adj)

        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape

        # Temporal attention (unchanged)
        xt = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt, _ = self.t_attn(xt, xt, xt)
        x = self.norm_t(x + self.drop(xt).reshape(B, N, T, D).permute(0, 2, 1, 3))

        # Spatial attention with multi-hop graph bias (unchanged interface)
        xs = x.reshape(B * T, N, D)
        xs, _ = self.s_attn(xs, xs, xs, attn_mask=self.g_bias())
        x = self.norm_s(x + self.drop(xs).reshape(B, T, N, D))

        # Feed-forward (unchanged)
        x = self.norm_ff(x + self.drop(self.ff(x)))
        return x


class STTransformerMultiHop(nn.Module):
    """
    Multi-Hop ST-Transformer.
    API is identical to Mohan's STTransformer — just pass adj to enable multi-hop.
    """
    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        pred_len: int,
        d_model: int   = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int    = 256,
        dropout: float = 0.1,
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
        h = self.input_proj(x.unsqueeze(-1))
        h = h + self.node_emb(torch.arange(N, device=x.device))
        h_pe = self.pos_enc(h.mean(2))
        h = h + h_pe.unsqueeze(2)
        for block in self.encoder:
            h = block(h)
        out = self.out_proj(h.mean(1))        # [B, N, M]
        return out.permute(0, 2, 1)           # [B, M, N]


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    N = 207

    # Simulate a random adjacency matrix
    adj = torch.zeros(N, N)
    for i in range(N):
        neighbors = torch.randperm(N)[:5]
        adj[i, neighbors] = 1
    adj = ((adj + adj.T) > 0).float()   # symmetrize

    model = STTransformerMultiHop(
        num_sensors=N, seq_len=12, pred_len=12, adj=adj
    )
    x   = torch.randn(8, 12, N)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {params:,}")

    # Check hop weights
    bias_module = model.encoder[0].g_bias
    print(f"\nHop weights:")
    print(f"  w1 (1-hop): {bias_module.w1.item():.4f}")
    print(f"  w2 (2-hop): {bias_module.w2.item():.4f}")
    print(f"  w3 (3-hop): {bias_module.w3.item():.4f}")

    A1_nnz = bias_module.A1.sum().item()
    A2_nnz = bias_module.A2.sum().item()
    A3_nnz = bias_module.A3.sum().item()
    print(f"\nAdjacency coverage:")
    print(f"  1-hop non-zeros: {A1_nnz:.0f}  ({A1_nnz/N**2*100:.1f}% of N²)")
    print(f"  2-hop non-zeros: {A2_nnz:.0f}  ({A2_nnz/N**2*100:.1f}% of N²)")
    print(f"  3-hop non-zeros: {A3_nnz:.0f}  ({A3_nnz/N**2*100:.1f}% of N²)")

    assert out.shape == (8, 12, N)
    print("\n✓ Multi-hop model OK")
