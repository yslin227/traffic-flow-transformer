import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=209,
        output_dim=207,
        hidden_dim=64,
        num_layers=2,
        output_steps=12
    ):
        super(LSTMModel, self).__init__()

        # LSTM input
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim * output_steps)

        self.output_steps = output_steps
        self.output_dim = output_dim

    def forward(self, x):
        # x: (batch, T, input_dim)

        out, _ = self.lstm(x)  # (batch, T, hidden)

        out = out[:, -1, :]  # (batch, hidden)

        out = self.fc(out)  # (batch, output_dim * output_steps)

        out = out.view(-1, self.output_steps, self.output_dim)

        return out