import os
import sys
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
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


if __name__ == "__main__":
    df = load_metr_la()
    data = df_to_numpy(df)
    adj = load_adj()

    time_features = create_time_features(df)
    X, Y = create_windows(data, time_features)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)

    train_dataset = TrafficDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for x, y in train_loader:
        print("X batch:", x.shape)
        print("Y batch:", y.shape)
        break