import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        input_dim=209,
        output_dim=207,
        seq_len=12,
        output_steps=12,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(SimpleTransformer, self).__init__()

        self.seq_len = seq_len
        self.output_steps = output_steps
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Forecast head: use the whole encoded sequence, not per-step direct mapping
        self.head = nn.Sequential(
            nn.Linear(seq_len * d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_steps * output_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)              # (B, T, d_model)
        x = self.pos_encoder(x)             # (B, T, d_model)
        x = self.transformer(x)             # (B, T, d_model)

        x = x.reshape(x.size(0), -1)        # (B, T*d_model)
        x = self.head(x)                    # (B, output_steps*output_dim)
        x = x.view(-1, self.output_steps, self.output_dim)

        return x