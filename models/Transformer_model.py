"""
STTransformer (part2_model.py)
==============================
Spatial-Temporal Transformer for traffic flow forecasting.
Inspired by MTESformer (Dong et al., 2024, IEEE Access).

Architecture:
  1. Input projection
  2. Multi-Scale Convolution Unit (MSCU) – temporal feature extraction
  3. Stacked ST-Blocks (Temporal Self-Attention + Spatial Attention + FFN)
  4. Output projection → pred_len steps

Expected tensor shapes:
  Input  X : (B, T, N, F)   B=batch, T=seq_len, N=num_sensors, F=input_features
  Output   : (B, T_out, N, 1)   T_out=pred_len
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. Multi-Scale Convolution Unit (MSCU)
# ─────────────────────────────────────────────
class MSCU(nn.Module):
    """
    Extracts multi-scale temporal features with parallel dilated convolutions.
    Kernels at three scales are concatenated then projected back to d_model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        branch_dim = d_model // 4           # output dim per branch

        # Three dilated conv branches (kernel=3, dilations 1,2,4)
        self.conv1 = nn.Conv1d(d_model, branch_dim, kernel_size=3,
                               padding=1,  dilation=1)
        self.conv2 = nn.Conv1d(d_model, branch_dim, kernel_size=3,
                               padding=2,  dilation=2)
        self.conv3 = nn.Conv1d(d_model, branch_dim, kernel_size=3,
                               padding=4,  dilation=4)
        # Identity branch to preserve residual info
        self.conv_id = nn.Conv1d(d_model, branch_dim, kernel_size=1)

        self.proj    = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B*N, T, d_model)
        returns: same shape
        """
        residual = x
        # Conv1d expects (B, C, L)
        h = x.transpose(1, 2)
        h = torch.cat([
            self.act(self.conv1(h)),
            self.act(self.conv2(h)),
            self.act(self.conv3(h)),
            self.act(self.conv_id(h)),
        ], dim=1)                           # (B*N, d_model, T)
        h = h.transpose(1, 2)              # (B*N, T, d_model)
        h = self.dropout(self.proj(h))
        return self.norm(h + residual)


# ─────────────────────────────────────────────
# 2. Temporal Self-Attention
# ─────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """Standard multi-head self-attention over the time dimension."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B*N, T, d_model)"""
        h, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(h))


# ─────────────────────────────────────────────
# 3. Spatial Attention
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Multi-head self-attention over the sensor (node) dimension.
    Captures inter-sensor correlations at each time step.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, N, d_model)  →  same shape"""
        B, T, N, D = x.shape
        # Attend over N for each time step independently
        h = x.reshape(B * T, N, D)
        h, _ = self.attn(h, h, h)
        h = h.reshape(B, T, N, D)
        return self.norm(x + self.dropout(h))


# ─────────────────────────────────────────────
# 4. Feed-Forward Network
# ─────────────────────────────────────────────
class FFN(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ─────────────────────────────────────────────
# 5. ST-Block
# ─────────────────────────────────────────────
class STBlock(nn.Module):
    """
    One Spatial-Temporal block:
      MSCU  →  Temporal Attention  →  Spatial Attention  →  FFN
    """
    def __init__(self, d_model: int, num_heads: int,
                 ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mscu       = MSCU(d_model, dropout)
        self.temp_attn  = TemporalAttention(d_model, num_heads, dropout)
        self.spat_attn  = SpatialAttention(d_model, num_heads, dropout)
        self.ffn        = FFN(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, N, d_model)"""
        B, T, N, D = x.shape

        # MSCU + Temporal attention: operate on (B*N, T, D)
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h = self.mscu(h)
        h = self.temp_attn(h)
        x = h.reshape(B, N, T, D).permute(0, 2, 1, 3)  # back to (B,T,N,D)

        # Spatial attention
        x = self.spat_attn(x)

        # FFN
        x = self.ffn(x)
        return x


# ─────────────────────────────────────────────
# 6. Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)   # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, N, d_model)"""
        x = x + self.pe[:x.size(1)].unsqueeze(0).unsqueeze(2)
        return self.dropout(x)


# ─────────────────────────────────────────────
# 7. STTransformer (top-level)
# ─────────────────────────────────────────────
class STTransformer(nn.Module):
    """
    Full Spatial-Temporal Transformer.

    Args:
        num_sensors : number of road sensors / graph nodes  (N)
        seq_len     : input sequence length                 (T)
        pred_len    : prediction horizon                    (T_out)
        d_model     : internal embedding dimension
        num_heads   : attention heads (must divide d_model)
        num_layers  : number of ST-Blocks
        ff_dim      : feed-forward hidden size
        dropout     : dropout probability
        in_features : raw input features per sensor (default 1)
    """
    def __init__(
        self,
        num_sensors : int,
        seq_len     : int,
        pred_len    : int,
        d_model     : int  = 64,
        num_heads   : int  = 4,
        num_layers  : int  = 3,
        ff_dim      : int  = 256,
        dropout     : float = 0.1,
        in_features : int  = 1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.d_model  = d_model

        # Input projection: map raw features → d_model
        self.input_proj = nn.Linear(in_features, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max(seq_len, pred_len) + 1,
                                          dropout=dropout)

        # Stack of ST-Blocks
        self.blocks = nn.ModuleList([
            STBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output head: (T, d_model) → pred_len speed values
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, N) or (B, T, N, F)
        Returns:
            out : (B, T_out, N, 1)
        """
        # Handle both 3-D and 4-D inputs
        if x.dim() == 3:
            x = x.unsqueeze(-1)            # (B, T, N, 1)

        B, T, N, F = x.shape

        # Project to d_model
        h = self.input_proj(x)             # (B, T, N, d_model)
        h = self.pos_enc(h)

        # ST-Blocks
        for block in self.blocks:
            h = block(h)                   # (B, T, N, d_model)

        h = self.out_norm(h)

        # Flatten time & d_model, then project to pred_len
        h = h.permute(0, 2, 1, 3)         # (B, N, T, d_model)
        h = h.reshape(B, N, T * self.d_model)
        out = self.out_proj(h)             # (B, N, pred_len)
        out = out.permute(0, 2, 1)         # (B, pred_len, N)
        out = out.unsqueeze(-1)            # (B, pred_len, N, 1)
        return out


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    B, T, N = 4, 12, 207          # METR-LA: 207 sensors
    model = STTransformer(num_sensors=N, seq_len=T, pred_len=12,
                          d_model=64, num_heads=4, num_layers=3, ff_dim=256)
    x   = torch.randn(B, T, N)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {params:,}")
    assert out.shape == (B, 12, N, 1), f"Unexpected shape: {out.shape}"
    print("✓ Shape check passed")
