import os
import sys
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
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
from evaluation.metrics import (
    mae,
    rmse,
    mape,
    masked_mape,
    wmape,
    smape,
    r2_score,
    median_ae,
    max_ae,
    analyze_data_distribution,
    analyze_error_distribution,
    compute_horizon_metrics,
    compute_sensor_statistics,
    count_parameters,
    format_results,
)


def train():
    # Create output directories under project root
    results_dir = os.path.join(PROJECT_ROOT, "results")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
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
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = LSTMModel().to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Training settings
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                best_model_state,
                os.path.join(checkpoints_dir, "best_lstm.pt")
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Load best model before testing
    model.load_state_dict(best_model_state)
    model.eval()

    # Test evaluation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Overall metrics
    test_mae = mae(all_targets, all_preds)
    test_rmse = rmse(all_targets, all_preds)
    test_mape = mape(all_targets, all_preds)
    test_mape_masked, test_mape_coverage = masked_mape(all_targets, all_preds)
    test_wmape = wmape(all_targets, all_preds)
    test_smape = smape(all_targets, all_preds)
    test_r2 = r2_score(all_targets, all_preds)
    test_median_ae = median_ae(all_targets, all_preds)
    test_max_ae = max_ae(all_targets, all_preds)

    # Analysis blocks
    data_distribution = analyze_data_distribution(all_targets)
    error_distribution = analyze_error_distribution(all_targets, all_preds)
    horizon_metrics = compute_horizon_metrics(all_targets, all_preds)
    sensor_stats = compute_sensor_statistics(all_targets, all_preds)
    param_info = count_parameters(model)

    # Build full report dict
    results_dict = {
        "model_name": "LSTM",
        "dataset": "METR-LA",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_sensors": all_targets.shape[2],
        "seq_len": X_test.shape[1],
        "pred_len": Y_test.shape[1],
        "test_samples": all_targets.shape[0],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
        "test_mape_masked": test_mape_masked,
        "test_mape_coverage": test_mape_coverage,
        "test_wmape": test_wmape,
        "test_smape": test_smape,
        "test_r2": test_r2,
        "test_median_ae": test_median_ae,
        "test_max_ae": test_max_ae,
        "data_distribution": data_distribution,
        "error_distribution": error_distribution,
        "horizon_metrics": horizon_metrics,
        "sensor_stats": sensor_stats,
        "total_params": param_info["total"],
        "trainable_params": param_info["trainable"],
        "input_dim": 209,
        "output_dim": 207,
        "hidden_dim": 64,
        "num_layers": 2,
    }

    # Short result file
    short_result_str = (
        f"Best Epoch: {best_epoch}\n"
        f"Best Val Loss: {best_val_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n"
        f"Test RMSE: {test_rmse:.4f}\n"
        f"Test MAPE: {test_mape:.4f}"
    )

    print("\n" + short_result_str)

    with open(os.path.join(results_dir, "lstm_results.txt"), "w", encoding="utf-8") as f:
        f.write(short_result_str + "\n")

    # Comprehensive result file
    comprehensive_result_str = format_results(results_dict)

    print("\n" + comprehensive_result_str)

    with open(
        os.path.join(results_dir, "lstm_results_comprehensive.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(comprehensive_result_str + "\n")


if __name__ == "__main__":
    train()