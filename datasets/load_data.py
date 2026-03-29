import pandas as pd
import numpy as np
import pickle


# Load METR-LA dataset
def load_metr_la(path="data/metr-la.h5"):
    df = pd.read_hdf(path)

    print("=== METR-LA Data ===")
    print("Shape:", df.shape)
    print("Time range:", df.index.min(), "->", df.index.max())
    print("First 5 sensor IDs:", df.columns[:5])

    return df


# Convert to numpy
def df_to_numpy(df):
    data = df.values  # (time_steps, num_sensors)

    print("\n=== Numpy Data ===")
    print("Shape:", data.shape)
    print("Min:", data.min())
    print("Max:", data.max())
    print("Mean:", data.mean())

    return data


def create_time_features(df):
    # Use datetime index to build temporal features
    time_index = df.index

    # Time of day: normalize to [0, 1]
    time_of_day = (
        time_index.hour * 60 + time_index.minute
    ) / (24 * 60)

    # Day of week: normalize to [0, 1]
    day_of_week = time_index.dayofweek / 7.0

    time_of_day = time_of_day.values
    day_of_week = day_of_week.values

    # Shape: (time_steps, 2)
    time_features = np.stack([time_of_day, day_of_week], axis=1)

    print("\n=== Time Features ===")
    print("Shape:", time_features.shape)

    return time_features


# Load adjacency matrix
def load_adj(path="data/adj_mx.pkl"):
    with open(path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    print("\n=== Adjacency Matrix ===")
    print("Shape:", adj_mx.shape)
    print("Number of sensors:", len(sensor_ids))

    return adj_mx


# Create sliding windows
def create_windows(data, time_features, T=12, M=12):
    X = []
    Y = []

    total_steps = data.shape[0]

    for i in range(total_steps - T - M + 1):
        x_data = data[i : i + T]              # (T, 207)
        x_time = time_features[i : i + T]     # (T, 2)

        # Concatenate traffic data with time features
        x = np.concatenate([x_data, x_time], axis=1)   # (T, 209)

        y = data[i + T : i + T + M]           # (M, 207)

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    print("\n=== Sliding Window (with time features) ===")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y


# Train / Val / Test split
def train_val_test_split(X, Y, train_ratio=0.7, val_ratio=0.1):
    total = X.shape[0]

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    X_train = X[:train_end]
    Y_train = Y[:train_end]

    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]

    X_test = X[val_end:]
    Y_test = Y[val_end:]

    print("\n=== Data Split ===")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# Normalization (NO DATA LEAKAGE)
def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("\n=== Normalization ===")
    print("Mean:", mean)
    print("Std:", std)

    return X_train, X_val, X_test, mean, std


# Main
if __name__ == "__main__":
    # 1. Load dataframe
    df = load_metr_la()

    # 2. Convert to numpy
    data = df_to_numpy(df)

    # 3. Load graph
    adj = load_adj()

    # 4. Create sliding windows
    time_features = create_time_features(df)
    X, Y = create_windows(data, time_features)

    # 5. Split dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)

    # 6. Normalize (only X)
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)