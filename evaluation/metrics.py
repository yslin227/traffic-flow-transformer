import numpy as np


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