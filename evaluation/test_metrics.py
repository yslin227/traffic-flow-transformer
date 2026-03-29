import numpy as np
from metrics import mae, rmse, mape


if __name__ == "__main__":
    y_true = np.array([10, 20, 30, 40], dtype=float)
    y_pred = np.array([12, 18, 33, 36], dtype=float)

    print("MAE:", mae(y_true, y_pred))
    print("RMSE:", rmse(y_true, y_pred))
    print("MAPE:", mape(y_true, y_pred))