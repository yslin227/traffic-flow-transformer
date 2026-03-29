import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
sys.path.append(PROJECT_ROOT)

from datasets.load_data import (
    load_metr_la,
    df_to_numpy,
    load_adj,
    create_windows,
    train_val_test_split,
    normalize_data,
    create_time_features,
)
from datasets.traffic_dataset import TrafficDataset
from models.lstm import LSTMModel
from evaluation.metrics import mae, rmse, mape


def train():
    # Load & preprocess data
    df = load_metr_la()
    data = df_to_numpy(df)
    adj = load_adj()
    time_features = create_time_features(df)
    X, Y = create_windows(data, time_features)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)

    train_dataset = TrafficDataset(X_train, Y_train)
    val_dataset = TrafficDataset(X_val, Y_val)
    test_dataset = TrafficDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = LSTMModel()

    # Loss & optimizer
    criterion = nn.L1Loss()  # MAE loss for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Test evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_mae = mae(all_targets, all_preds)
    test_rmse = rmse(all_targets, all_preds)
    test_mape = mape(all_targets, all_preds)

    result_str = (
        f"Test MAE: {test_mae:.4f}\n"
        f"Test RMSE: {test_rmse:.4f}\n"
        f"Test MAPE: {test_mape:.4f}"
    )

    print("\n" + result_str)

    with open("results/lstm_results.txt", "w") as f:
        f.write(result_str + "\n\n")


if __name__ == "__main__":
    train()