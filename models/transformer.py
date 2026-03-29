import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        input_dim=209,
        output_dim=207,
        d_model=64,
        nhead=4,
        num_layers=2
    ):
        super(SimpleTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, T, input_dim)

        x = self.input_proj(x)   # (batch, T, d_model)
        x = self.transformer(x)  # (batch, T, d_model)
        x = self.fc(x)           # (batch, T, output_dim)

        return x