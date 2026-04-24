import numpy as np


# Basic Metrics

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, null_val=0.0):
    mask = np.not_equal(y_true, null_val)
    mask = mask.astype(np.float32)

    if mask.mean() == 0:
        return 0.0

    mask /= mask.mean()

    mape_value = np.abs((y_pred - y_true) / np.where(mask > 0, y_true, 1.0))
    mape_value = np.nan_to_num(mask * mape_value)

    return np.mean(mape_value) * 100


# Improved Metrics

def masked_mape(y_true, y_pred, threshold=0.1):
    mask = (y_true > threshold) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return 0.0, 0.0

    mape_val = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    coverage = mask.sum() / y_true.size * 100
    return mape_val, coverage


def wmape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true)) + 1e-8
    return np.sum(np.abs(y_true - y_pred)) / denominator * 100


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def median_ae(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))


def max_ae(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


# Data Analysis

def analyze_data_distribution(y_true):
    flat = y_true.flatten()
    flat = flat[~np.isnan(flat)]

    total_count = len(flat)
    zero_count = int(np.sum(flat == 0))
    near_zero_count = int(np.sum(np.abs(flat) < 0.1))
    low_speed_count = int(np.sum((flat > 0) & (flat < 5)))

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "zero_count": zero_count,
        "zero_percent": float(zero_count / total_count * 100),
        "near_zero_count": near_zero_count,
        "near_zero_percent": float(near_zero_count / total_count * 100),
        "low_speed_count": low_speed_count,
        "low_speed_percent": float(low_speed_count / total_count * 100),
    }


def analyze_error_distribution(y_true, y_pred):
    errors = np.abs(y_true - y_pred).flatten()
    errors = errors[~np.isnan(errors)]

    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
        "q25_error": float(np.percentile(errors, 25)),
        "q75_error": float(np.percentile(errors, 75)),
        "q95_error": float(np.percentile(errors, 95)),
        "q99_error": float(np.percentile(errors, 99)),
    }


# Horizon Metrics

def compute_horizon_metrics(y_true, y_pred, threshold=0.1):
    horizons = y_true.shape[1]
    results = []

    for h in range(1, horizons + 1):
        yt = y_true[:, :h]
        yp = y_pred[:, :h]

        masked_mape_val, coverage = masked_mape(yt, yp, threshold=threshold)

        results.append({
            "horizon_step": h,
            "horizon_minutes": h * 5,
            "mae": float(mae(yt, yp)),
            "rmse": float(rmse(yt, yp)),
            "mape": float(mape(yt, yp)),
            "mape_masked": float(masked_mape_val),
            "mape_coverage": float(coverage),
            "wmape": float(wmape(yt, yp)),
            "smape": float(smape(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        })

    return results


# Sensor-level Metrics

def compute_sensor_statistics(y_true, y_pred):
    num_sensors = y_true.shape[2]
    sensor_stats = []

    for i in range(num_sensors):
        yt = y_true[:, :, i]
        yp = y_pred[:, :, i]

        sensor_stats.append({
            "sensor_id": i,
            "mae": float(mae(yt, yp)),
            "rmse": float(rmse(yt, yp)),
            "wmape": float(wmape(yt, yp)),
        })

    sensor_stats.sort(key=lambda x: x["mae"])

    return {
        "best_sensors": sensor_stats[:10],
        "worst_sensors": sensor_stats[-10:],
        "avg_mae": float(np.mean([s["mae"] for s in sensor_stats])),
        "avg_rmse": float(np.mean([s["rmse"] for s in sensor_stats])),
        "avg_wmape": float(np.mean([s["wmape"] for s in sensor_stats])),
    }


# Model Info

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": int(total),
        "trainable": int(trainable),
    }


# Format Output

def format_results(results):
    lines = []

    lines.append("=" * 80)
    lines.append(f"COMPREHENSIVE EVALUATION REPORT - {results['dataset']}")
    lines.append("=" * 80)

    if "timestamp" in results:
        lines.append(f"Generated: {results['timestamp']}")
        lines.append("")

    # Dataset Information
    lines.append("-" * 80)
    lines.append("DATASET INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Dataset: {results['dataset']}")
    lines.append(f"Model: {results['model_name']}")
    lines.append(f"Number of Sensors: {results['num_sensors']}")
    lines.append(f"Sequence Length: {results['seq_len']} steps ({results['seq_len'] * 5} minutes)")
    lines.append(f"Prediction Length: {results['pred_len']} steps ({results['pred_len'] * 5} minutes)")
    if "test_samples" in results:
        lines.append(f"Test Samples: {results['test_samples']}")
    lines.append("")

    # Target Data Distribution
    if "data_distribution" in results:
        dist = results["data_distribution"]
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
    lines.append(f"Model Type: {results['model_name']}")

    optional_keys = [
        ("input_dim", "Input Dimension"),
        ("output_dim", "Output Dimension"),
        ("hidden_dim", "Hidden Dimension"),
        ("num_layers", "Number of Layers"),
        ("d_model", "Embedding Dimension (d_model)"),
        ("nhead", "Number of Attention Heads"),
        ("num_heads", "Number of Attention Heads"),
        ("ff_dim", "Feed-forward Dimension"),
    ]

    for key, label in optional_keys:
        if key in results:
            lines.append(f"{label}: {results[key]}")

    if "total_params" in results:
        lines.append(f"Total Parameters: {results['total_params']:,}")
    if "trainable_params" in results:
        lines.append(f"Trainable Parameters: {results['trainable_params']:,}")
    lines.append("")

    # Training Information
    lines.append("-" * 80)
    lines.append("TRAINING INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Best Epoch: {results['best_epoch']}")
    lines.append(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    if "inference_time" in results:
        lines.append(f"Inference Time (total): {results['inference_time']:.2f} seconds")
    if "inference_time_per_sample" in results:
        lines.append(f"Inference Time (per sample): {results['inference_time_per_sample']:.4f} seconds")
    lines.append("")

    # Overall Test Performance
    lines.append("-" * 80)
    lines.append("OVERALL TEST PERFORMANCE")
    lines.append("-" * 80)
    lines.append(f"MAE:            {results['test_mae']:.4f}")
    lines.append(f"RMSE:           {results['test_rmse']:.4f}")
    if "test_mape" in results:
        lines.append(f"MAPE:           {results['test_mape']:.2f}%")
    if "test_mape_masked" in results and "test_mape_coverage" in results:
        lines.append(
            f"MAPE (>0.1):    {results['test_mape_masked']:.2f}%  "
            f"(coverage: {results['test_mape_coverage']:.1f}%)"
        )
    if "test_wmape" in results:
        lines.append(f"WMAPE:          {results['test_wmape']:.2f}%")
    if "test_smape" in results:
        lines.append(f"SMAPE:          {results['test_smape']:.2f}%")
    if "test_r2" in results:
        lines.append(f"R² Score:       {results['test_r2']:.4f}")
    if "test_median_ae" in results:
        lines.append(f"Median AE:      {results['test_median_ae']:.4f}")
    if "test_max_ae" in results:
        lines.append(f"Max AE:         {results['test_max_ae']:.4f}")
    lines.append("")
    lines.append("METRIC GUIDE:")
    lines.append("  - Use MAE/RMSE as primary metrics")
    lines.append("  - WMAPE is more stable when values are near zero")
    lines.append("  - MAPE can be misleading when true values are very small")
    lines.append("")

    # Error Distribution
    if "error_distribution" in results:
        err = results["error_distribution"]
        lines.append("-" * 80)
        lines.append("ERROR DISTRIBUTION ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Mean Error:      {err['mean_error']:.4f}")
        lines.append(f"Std Dev Error:   {err['std_error']:.4f}")
        lines.append(f"Min Error:       {err['min_error']:.4f}")
        lines.append(f"Max Error:       {err['max_error']:.4f}")
        lines.append(f"Median Error:    {err['median_error']:.4f}")
        lines.append(f"25th Percentile: {err['q25_error']:.4f}")
        lines.append(f"75th Percentile: {err['q75_error']:.4f}")
        lines.append(f"95th Percentile: {err['q95_error']:.4f}")
        lines.append(f"99th Percentile: {err['q99_error']:.4f}")
        lines.append("")

    # Per-Horizon Performance
    if "horizon_metrics" in results:
        lines.append("-" * 80)
        lines.append("PER-HORIZON PERFORMANCE BREAKDOWN")
        lines.append("-" * 80)
        lines.append(
            f"{'Horizon':>8} {'Minutes':>8} {'MAE':>10} {'RMSE':>10} "
            f"{'MAPE':>10} {'MAPE>0.1':>10} {'WMAPE':>10} {'R²':>10}"
        )
        lines.append("-" * 100)
        for h in results["horizon_metrics"]:
            lines.append(
                f"{h['horizon_step']:>8} "
                f"{h['horizon_minutes']:>8} "
                f"{h['mae']:>10.4f} "
                f"{h['rmse']:>10.4f} "
                f"{h['mape']:>9.2f}% "
                f"{h['mape_masked']:>9.2f}% "
                f"{h['wmape']:>9.2f}% "
                f"{h['r2']:>10.4f}"
            )
        lines.append("")

    # Sensor-level Statistics
    if "sensor_stats" in results:
        s = results["sensor_stats"]
        lines.append("-" * 80)
        lines.append("SENSOR-LEVEL STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Average MAE across sensors:   {s['avg_mae']:.4f}")
        lines.append(f"Average RMSE across sensors:  {s['avg_rmse']:.4f}")
        if "avg_wmape" in s:
            lines.append(f"Average WMAPE across sensors: {s['avg_wmape']:.2f}%")
        lines.append("")

        lines.append("Top 10 Best Performing Sensors (by MAE):")
        lines.append(f"{'Sensor ID':>12} {'MAE':>10} {'RMSE':>10} {'WMAPE':>10}")
        lines.append("-" * 45)
        for sensor in s["best_sensors"]:
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
        for sensor in reversed(s["worst_sensors"]):
            lines.append(
                f"{sensor['sensor_id']:>12} "
                f"{sensor['mae']:>10.4f} "
                f"{sensor['rmse']:>10.4f} "
                f"{sensor['wmape']:>9.2f}%"
            )
        lines.append("")

    return "\n".join(lines)