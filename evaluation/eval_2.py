"""
Evaluation & Metrics:
Run:
    python eval_2.py --dataset METR-LA
    python eval_2.py --dataset PEMS-BAY
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import time

from part2_model import STTransformer
from part3_train import TrafficDataset, make_loader


# Metrics with robust handling
def mae(pred, target, null=0.0):
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    return np.abs(pred[mask] - target[mask]).mean()

def rmse(pred, target, null=0.0):
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    return np.sqrt(((pred[mask] - target[mask]) ** 2).mean())

def mape(pred, target, null=0.0, threshold=0.1):
    """
    MAPE with threshold to exclude near-zero values
    For traffic data, speeds below threshold (e.g., 0.1 mph) are excluded
    """
    mask = (target > threshold) & (target != null)
    if mask.sum() == 0:
        return 0.0
    return (np.abs((pred[mask] - target[mask]) / target[mask])).mean() * 100

def masked_mape(pred, target, null=0.0):
    """
    Alternative MAPE: only compute on non-null, positive values
    Reports what % of data was used
    """
    mask = (target > 0) & (target != null) & (~np.isnan(target)) & (~np.isnan(pred))
    if mask.sum() == 0:
        return 0.0, 0.0
    mape_val = (np.abs((pred[mask] - target[mask]) / target[mask])).mean() * 100
    coverage = mask.sum() / target.size * 100
    return mape_val, coverage

def wmape(pred, target, null=0.0):
    """
    Weighted MAPE - more stable for values near zero
    WMAPE = sum(|pred - target|) / sum(|target|) * 100
    """
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    return (np.abs(pred[mask] - target[mask]).sum() / np.abs(target[mask]).sum()) * 100

def smape(pred, target, null=0.0, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    pred_masked = pred[mask]
    target_masked = target[mask]
    numerator = np.abs(pred_masked - target_masked)
    denominator = (np.abs(pred_masked) + np.abs(target_masked)) / 2 + eps
    return (numerator / denominator).mean() * 100

def r2_score(pred, target, null=0.0):
    """Coefficient of determination"""
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    pred_masked = pred[mask]
    target_masked = target[mask]
    ss_res = np.sum((target_masked - pred_masked) ** 2)
    ss_tot = np.sum((target_masked - np.mean(target_masked)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def median_ae(pred, target, null=0.0):
    """Median Absolute Error"""
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    return np.median(np.abs(pred[mask] - target[mask]))

def max_ae(pred, target, null=0.0):
    """Maximum Absolute Error"""
    mask = (target != null)
    if mask.sum() == 0:
        return 0.0
    return np.max(np.abs(pred[mask] - target[mask]))


# LSTM baseline 
class LSTMBaseline(nn.Module):
    def __init__(self, num_sensors: int, hidden: int, pred_len: int):
        super().__init__()
        self.lstm    = nn.LSTM(num_sensors, hidden, batch_first=True, num_layers=2,
                               dropout=0.1)
        self.out     = nn.Linear(hidden, num_sensors * pred_len)
        self.pred_len = pred_len
        self.N = num_sensors

    def forward(self, x):      # x: [B, T, N]
        h, _ = self.lstm(x)
        return self.out(h[:, -1]).view(x.size(0), self.pred_len, self.N)


def train_lstm_baseline(train_loader, val_loader, num_sensors, pred_len, device,
                         epochs=30, lr=1e-3):
    model = LSTMBaseline(num_sensors, hidden=64, pred_len=pred_len).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            loss = nn.L1Loss()(model(X), Y)
            loss.backward(); opt.step()
        model.eval()
        vl = sum(nn.L1Loss()(model(X.to(device)), Y.to(device)).item()
                 for X, Y in val_loader) / len(val_loader)
        if vl < best_val:
            best_val = vl
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"    LSTM epoch {ep:>3}  val MAE={vl:.4f}")
    model.load_state_dict(best_sd)
    return model


# Collect predictions 
def collect(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, Y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(Y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# Inverse transform 
def inverse(arr, scaler):
    B, M, N = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, N)).reshape(B, M, N)


# Data statistics
def analyze_data_distribution(target):
    """Analyze the distribution of target values"""
    flat_target = target.flatten()
    flat_target = flat_target[~np.isnan(flat_target)]
    
    zero_count = np.sum(flat_target == 0)
    near_zero_count = np.sum(np.abs(flat_target) < 0.1)
    low_speed_count = np.sum((flat_target > 0) & (flat_target < 5))
    
    return {
        'min': np.min(flat_target),
        'max': np.max(flat_target),
        'mean': np.mean(flat_target),
        'median': np.median(flat_target),
        'std': np.std(flat_target),
        'zero_count': int(zero_count),
        'zero_percent': (zero_count / len(flat_target)) * 100,
        'near_zero_count': int(near_zero_count),
        'near_zero_percent': (near_zero_count / len(flat_target)) * 100,
        'low_speed_count': int(low_speed_count),
        'low_speed_percent': (low_speed_count / len(flat_target)) * 100,
    }


# Per-horizon metrics with multiple MAPE variants
def compute_horizon_metrics(pred, target, horizons=None, mape_threshold=0.1):
    """Compute metrics for each prediction horizon"""
    if horizons is None:
        horizons = list(range(1, pred.shape[1] + 1))
    
    results = []
    for h in horizons:
        if h <= pred.shape[1]:
            mape_val, coverage = masked_mape(pred[:, :h], target[:, :h])
            
            metrics = {
                'horizon_step': h,
                'horizon_minutes': h * 5,
                'mae': mae(pred[:, :h], target[:, :h]),
                'rmse': rmse(pred[:, :h], target[:, :h]),
                'mape': mape(pred[:, :h], target[:, :h], threshold=mape_threshold),
                'mape_masked': mape_val,
                'mape_coverage': coverage,
                'wmape': wmape(pred[:, :h], target[:, :h]),
                'smape': smape(pred[:, :h], target[:, :h]),
                'r2': r2_score(pred[:, :h], target[:, :h]),
            }
            results.append(metrics)
    return results


# Per-sensor statistics
def compute_sensor_statistics(pred, target, num_top_sensors=10, mape_threshold=0.1):
    """Compute per-sensor error statistics"""
    num_sensors = pred.shape[2]
    sensor_errors = []
    
    for i in range(num_sensors):
        sensor_pred = pred[:, :, i]
        sensor_target = target[:, :, i]
        
        sensor_mae = mae(sensor_pred, sensor_target)
        sensor_rmse = rmse(sensor_pred, sensor_target)
        sensor_mape = mape(sensor_pred, sensor_target, threshold=mape_threshold)
        sensor_wmape = wmape(sensor_pred, sensor_target)
        
        sensor_errors.append({
            'sensor_id': i,
            'mae': sensor_mae,
            'rmse': sensor_rmse,
            'mape': sensor_mape,
            'wmape': sensor_wmape,
        })
    
    # Sort by MAE
    sensor_errors.sort(key=lambda x: x['mae'])
    
    return {
        'best_sensors': sensor_errors[:num_top_sensors],
        'worst_sensors': sensor_errors[-num_top_sensors:],
        'avg_mae': np.mean([s['mae'] for s in sensor_errors]),
        'avg_rmse': np.mean([s['rmse'] for s in sensor_errors]),
        'avg_mape': np.mean([s['mape'] for s in sensor_errors]),
        'avg_wmape': np.mean([s['wmape'] for s in sensor_errors]),
    }


# Error distribution analysis
def analyze_error_distribution(pred, target):
    """Analyze the distribution of prediction errors"""
    errors = np.abs(pred - target)
    errors = errors[~np.isnan(errors)]
    
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'median_error': np.median(errors),
        'q25_error': np.percentile(errors, 25),
        'q75_error': np.percentile(errors, 75),
        'q95_error': np.percentile(errors, 95),
        'q99_error': np.percentile(errors, 99),
    }


# Count model parameters
def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


# Format results to text
def format_results(results_dict):
    """Format comprehensive results as text"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"COMPREHENSIVE EVALUATION REPORT - {results_dict['dataset']}")
    lines.append("=" * 80)
    lines.append(f"Generated: {results_dict['timestamp']}")
    lines.append("")
    
    # Dataset Info
    lines.append("-" * 80)
    lines.append("DATASET INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Dataset: {results_dict['dataset']}")
    lines.append(f"Number of Sensors: {results_dict['num_sensors']}")
    lines.append(f"Sequence Length: {results_dict['seq_len']} steps ({results_dict['seq_len'] * 5} minutes)")
    lines.append(f"Prediction Length: {results_dict['pred_len']} steps ({results_dict['pred_len'] * 5} minutes)")
    lines.append(f"Test Samples: {results_dict['test_samples']}")
    lines.append("")
    
    # Data distribution
    if 'data_distribution' in results_dict:
        dist = results_dict['data_distribution']
        lines.append("-" * 80)
        lines.append("TARGET DATA DISTRIBUTION (Test Set)")
        lines.append("-" * 80)
        lines.append(f"Min Value:      {dist['min']:.4f}")
        lines.append(f"Max Value:      {dist['max']:.4f}")
        lines.append(f"Mean Value:     {dist['mean']:.4f}")
        lines.append(f"Median Value:   {dist['median']:.4f}")
        lines.append(f"Std Dev:        {dist['std']:.4f}")
        lines.append(f"Zero Values:    {dist['zero_count']:,} ({dist['zero_percent']:.2f}%)")
        lines.append(f"Near-Zero (<0.1): {dist['near_zero_count']:,} ({dist['near_zero_percent']:.2f}%)")
        lines.append(f"Low Speed (<5):  {dist['low_speed_count']:,} ({dist['low_speed_percent']:.2f}%)")
        lines.append("")
        lines.append("NOTE: High percentage of zero/low values can inflate MAPE.")
        lines.append("      WMAPE and MAE are more reliable for such distributions.")
        lines.append("")
    
    # Model Configuration
    lines.append("-" * 80)
    lines.append("MODEL CONFIGURATION")
    lines.append("-" * 80)
    lines.append(f"Model Type: ST-Transformer")
    lines.append(f"Embedding Dimension (d_model): {results_dict['d_model']}")
    lines.append(f"Number of Attention Heads: {results_dict['num_heads']}")
    lines.append(f"Number of Layers: {results_dict['num_layers']}")
    lines.append(f"Feed-forward Dimension: {results_dict['ff_dim']}")
    lines.append(f"Total Parameters: {results_dict['total_params']:,}")
    lines.append(f"Trainable Parameters: {results_dict['trainable_params']:,}")
    lines.append("")
    
    # Training Info
    lines.append("-" * 80)
    lines.append("TRAINING INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Best Epoch: {results_dict['best_epoch']}")
    lines.append(f"Best Validation MAE: {results_dict['best_val_loss']:.4f}")
    lines.append(f"Inference Time (total): {results_dict['inference_time']:.2f} seconds")
    lines.append(f"Inference Time (per sample): {results_dict['inference_time_per_sample']:.4f} seconds")
    lines.append("")
    
    # Overall Test Performance
    lines.append("-" * 80)
    lines.append("OVERALL TEST PERFORMANCE")
    lines.append("-" * 80)
    lines.append(f"MAE:            {results_dict['test_mae']:.4f}")
    lines.append(f"RMSE:           {results_dict['test_rmse']:.4f}")
    lines.append(f"MAPE (>0.1):    {results_dict['test_mape']:.2f}%  (excludes near-zero values)")
    lines.append(f"MAPE (masked):  {results_dict['test_mape_masked']:.2f}%  (coverage: {results_dict['test_mape_coverage']:.1f}%)")
    lines.append(f"WMAPE:          {results_dict['test_wmape']:.2f}%  (recommended for low-speed data)")
    lines.append(f"SMAPE:          {results_dict['test_smape']:.2f}%")
    lines.append(f"R² Score:       {results_dict['test_r2']:.4f}")
    lines.append(f"Median AE:      {results_dict['test_median_ae']:.4f}")
    lines.append(f"Max AE:         {results_dict['test_max_ae']:.4f}")
    lines.append("")
    lines.append("METRIC GUIDE:")
    lines.append("  - Use MAE/RMSE as primary metrics (not affected by division)")
    lines.append("  - WMAPE is recommended for traffic data with low speeds")
    lines.append("  - MAPE variants shown for reference (can be unreliable near zero)")
    lines.append("")
    
    # Error Distribution
    lines.append("-" * 80)
    lines.append("ERROR DISTRIBUTION ANALYSIS")
    lines.append("-" * 80)
    err_dist = results_dict['error_distribution']
    lines.append(f"Mean Error:      {err_dist['mean_error']:.4f}")
    lines.append(f"Std Dev Error:   {err_dist['std_error']:.4f}")
    lines.append(f"Min Error:       {err_dist['min_error']:.4f}")
    lines.append(f"Max Error:       {err_dist['max_error']:.4f}")
    lines.append(f"Median Error:    {err_dist['median_error']:.4f}")
    lines.append(f"25th Percentile: {err_dist['q25_error']:.4f}")
    lines.append(f"75th Percentile: {err_dist['q75_error']:.4f}")
    lines.append(f"95th Percentile: {err_dist['q95_error']:.4f}")
    lines.append(f"99th Percentile: {err_dist['q99_error']:.4f}")
    lines.append("")
    
    # Per Horizon Performance
    lines.append("-" * 80)
    lines.append("PER-HORIZON PERFORMANCE BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"{'Horizon':>8} {'Minutes':>8} {'MAE':>10} {'RMSE':>10} {'WMAPE':>10} {'R²':>10}")
    lines.append("-" * 80)
    for h_metrics in results_dict['horizon_metrics']:
        lines.append(
            f"{h_metrics['horizon_step']:>8} "
            f"{h_metrics['horizon_minutes']:>8} "
            f"{h_metrics['mae']:>10.4f} "
            f"{h_metrics['rmse']:>10.4f} "
            f"{h_metrics['wmape']:>9.2f}% "
            f"{h_metrics['r2']:>10.4f}"
        )
    lines.append("")
    
    # Sensor Statistics
    if 'sensor_stats' in results_dict:
        sensor_stats = results_dict['sensor_stats']
        lines.append("-" * 80)
        lines.append("SENSOR-LEVEL STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Average MAE across sensors:   {sensor_stats['avg_mae']:.4f}")
        lines.append(f"Average RMSE across sensors:  {sensor_stats['avg_rmse']:.4f}")
        lines.append(f"Average WMAPE across sensors: {sensor_stats['avg_wmape']:.2f}%")
        lines.append("")
        
        lines.append("Top 10 Best Performing Sensors (by MAE):")
        lines.append(f"{'Sensor ID':>12} {'MAE':>10} {'RMSE':>10} {'WMAPE':>10}")
        lines.append("-" * 45)
        for sensor in sensor_stats['best_sensors']:
            lines.append(
                f"{sensor['sensor_id']:>12} "
                f"{sensor['mae']:>10.4f} "
                f"{sensor['rmse']:>10.4f} "
                f"{sensor['wmape']:>9.2f}%"
            )
        lines.append("")
        
        lines.append("Top 10 Worst Performing Sensors (by MAE):")
        lines.append(f"{'Sensor ID':>12} {'MAE':>10} {'RMSE':>10} {'WMAPE':>10}")
        lines.append("-" * 45)
        for sensor in sensor_stats['worst_sensors']:
            lines.append(
                f"{sensor['sensor_id']:>12} "
                f"{sensor['mae']:>10.4f} "
                f"{sensor['rmse']:>10.4f} "
                f"{sensor['wmape']:>9.2f}%"
            )
        lines.append("")
    
    # Baseline Comparision
    if 'lstm_baseline' in results_dict:
        lstm = results_dict['lstm_baseline']
        lines.append("-" * 80)
        lines.append("BASELINE COMPARISON (LSTM)")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<15} {'ST-Transformer':>15} {'LSTM Baseline':>15} {'Improvement':>15}")
        lines.append("-" * 80)
        
        st_mae = results_dict['test_mae']
        lstm_mae = lstm['test_mae']
        improvement_mae = ((lstm_mae - st_mae) / lstm_mae) * 100
        lines.append(f"{'MAE':<15} {st_mae:>15.4f} {lstm_mae:>15.4f} {improvement_mae:>14.2f}%")
        
        st_rmse = results_dict['test_rmse']
        lstm_rmse = lstm['test_rmse']
        improvement_rmse = ((lstm_rmse - st_rmse) / lstm_rmse) * 100
        lines.append(f"{'RMSE':<15} {st_rmse:>15.4f} {lstm_rmse:>15.4f} {improvement_rmse:>14.2f}%")
        
        st_wmape = results_dict['test_wmape']
        lstm_wmape = lstm['test_wmape']
        improvement_wmape = ((lstm_wmape - st_wmape) / lstm_wmape) * 100
        lines.append(f"{'WMAPE':<15} {st_wmape:>14.2f}% {lstm_wmape:>14.2f}% {improvement_wmape:>14.2f}%")
        lines.append("")
        lines.append(f"LSTM Parameters: {lstm['total_params']:,}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# Main evaluation 
def evaluate(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag    = cfg["dataset"].replace("-", "_").lower()
    base   = f"{cfg['output_dir']}/{tag}"
    ckpt   = f"best_{tag}.pt"

    # Load scaler
    with open(f"{base}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load checkpoint
    checkpoint  = torch.load(ckpt, map_location=device)
    saved_cfg   = checkpoint["cfg"]
    num_sensors = np.load(f"{base}_test_X.npy").shape[2]

    model = STTransformer(
        num_sensors=num_sensors,
        seq_len=saved_cfg["seq_len"],
        pred_len=saved_cfg["pred_len"],
        d_model=saved_cfg["d_model"],
        num_heads=saved_cfg["num_heads"],
        num_layers=saved_cfg["num_layers"],
        ff_dim=saved_cfg["ff_dim"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    
    # Count parameters
    params = count_parameters(model)
    
    print(f"\n  Loaded checkpoint (epoch {checkpoint['epoch']}, "
          f"val MAE={checkpoint['val_loss']:.4f})")
    print(f"  Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Data loaders
    test_loader  = make_loader(base, "test",  cfg["batch_size"], shuffle=False)
    train_loader = make_loader(base, "train", cfg["batch_size"], shuffle=True)
    val_loader   = make_loader(base, "val",   cfg["batch_size"], shuffle=False)

    # Transformer results with timing
    print("\n  Running inference on test set...")
    start_time = time.time()
    pred_n, tgt_n = collect(model, test_loader, device)
    inference_time = time.time() - start_time
    
    pred  = inverse(pred_n, scaler)
    tgt   = inverse(tgt_n,  scaler)
    
    test_samples = pred.shape[0]
    
    # Analyze data distribution
    print("  Analyzing data distribution...")
    data_dist = analyze_data_distribution(tgt)

    print(f"\n{'─'*80}")
    print(f"  ST-Transformer  [{cfg['dataset']}]")
    print(f"  Data: min={data_dist['min']:.2f}, max={data_dist['max']:.2f}, "
          f"mean={data_dist['mean']:.2f}")
    print(f"  Zero/Near-zero values: {data_dist['near_zero_percent']:.2f}%")

    # Compute all metrics with robust MAPE
    test_mae = mae(pred, tgt)
    test_rmse = rmse(pred, tgt)
    test_mape = mape(pred, tgt, threshold=cfg.get('mape_threshold', 0.1))
    test_mape_masked, test_mape_coverage = masked_mape(pred, tgt)
    test_wmape = wmape(pred, tgt)
    test_smape = smape(pred, tgt)
    test_r2 = r2_score(pred, tgt)
    test_median_ae = median_ae(pred, tgt)
    test_max_ae = max_ae(pred, tgt)
    
    print(f"  Overall  MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  WMAPE={test_wmape:.2f}%")
    
    # Horizon metrics
    print("\n  Computing per-horizon metrics...")
    horizon_steps = list(range(1, saved_cfg["pred_len"] + 1))
    horizon_metrics = compute_horizon_metrics(pred, tgt, horizons=horizon_steps,
                                             mape_threshold=cfg.get('mape_threshold', 0.1))
    
    # Sensor statistics
    print("  Computing per-sensor statistics...")
    sensor_stats = compute_sensor_statistics(pred, tgt, num_top_sensors=10,
                                            mape_threshold=cfg.get('mape_threshold', 0.1))
    
    # Error distribution
    print("  Analyzing error distribution...")
    error_dist = analyze_error_distribution(pred, tgt)
    
    # Compile results
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': cfg['dataset'],
        'num_sensors': num_sensors,
        'seq_len': saved_cfg['seq_len'],
        'pred_len': saved_cfg['pred_len'],
        'test_samples': test_samples,
        'd_model': saved_cfg['d_model'],
        'num_heads': saved_cfg['num_heads'],
        'num_layers': saved_cfg['num_layers'],
        'ff_dim': saved_cfg['ff_dim'],
        'total_params': params['total'],
        'trainable_params': params['trainable'],
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / test_samples,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'test_mape_masked': test_mape_masked,
        'test_mape_coverage': test_mape_coverage,
        'test_wmape': test_wmape,
        'test_smape': test_smape,
        'test_r2': test_r2,
        'test_median_ae': test_median_ae,
        'test_max_ae': test_max_ae,
        'data_distribution': data_dist,
        'error_distribution': error_dist,
        'horizon_metrics': horizon_metrics,
        'sensor_stats': sensor_stats,
    }
    
    # LSTM Baseline (optional)
    if cfg.get('run_baseline', False):
        print("\n  Training LSTM baseline...")
        lstm_model = train_lstm_baseline(
            train_loader, val_loader, num_sensors, 
            saved_cfg['pred_len'], device, epochs=30
        )
        lstm_params = count_parameters(lstm_model)
        
        print("  Evaluating LSTM baseline on test set...")
        lstm_pred_n, _ = collect(lstm_model, test_loader, device)
        lstm_pred = inverse(lstm_pred_n, scaler)
        
        results['lstm_baseline'] = {
            'test_mae': mae(lstm_pred, tgt),
            'test_rmse': rmse(lstm_pred, tgt),
            'test_wmape': wmape(lstm_pred, tgt),
            'total_params': lstm_params['total'],
        }
        
        print(f"  LSTM  MAE={results['lstm_baseline']['test_mae']:.4f}  "
              f"RMSE={results['lstm_baseline']['test_rmse']:.4f}  "
              f"WMAPE={results['lstm_baseline']['test_wmape']:.2f}%")
    
    # Format and save
    output_text = format_results(results)
    print("\n" + output_text)
    
    output_file = f"results_{cfg['dataset'].replace('-', '_').lower()}_comprehensive.txt"
    with open(output_file, "w") as f:
        f.write(output_text)
    
    print(f"\n  Comprehensive results saved to: {output_file}")
    
    # Save as pickle
    pickle_file = f"results_{cfg['dataset'].replace('-', '_').lower()}_comprehensive.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"  Results pickle saved to: {pickle_file}")


# CLI 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="METR-LA")
    p.add_argument("--output_dir",   default="./processed")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--mape_threshold", type=float, default=0.1,
                   help="Minimum target value for MAPE calculation (excludes near-zero)")
    p.add_argument("--run_baseline", action="store_true", 
                   help="Train and evaluate LSTM baseline for comparison")
    args = p.parse_args()
    evaluate(vars(args))
