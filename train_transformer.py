import os
import sys
import copy
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
    create_time_features,
    create_windows,
    train_val_test_split,
    normalize_data,
)
from datasets.traffic_dataset import TrafficDataset
from models.transformer import SimpleTransformer
from evaluation.metrics import mae, rmse, mape


def train():
    # Create output folders
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = load_metr_la()
    data = df_to_numpy(df)
    _ = load_adj()

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
    model = SimpleTransformer().to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Training config
    epochs = 100
    patience = 10
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, "checkpoints/best_transformer.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()

    # Test evaluation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_mae = mae(all_targets, all_preds)
    test_rmse = rmse(all_targets, all_preds)
    test_mape = mape(all_targets, all_preds)

    result_str = (
        f"Best Epoch: {best_epoch}\n"
        f"Best Val Loss: {best_val_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n"
        f"Test RMSE: {test_rmse:.4f}\n"
        f"Test MAPE: {test_mape:.4f}"
    )

    print("\n" + result_str)

    with open("results/transformer_results.txt", "w") as f:
        f.write(result_str + "\n")


if __name__ == "__main__":
    train()