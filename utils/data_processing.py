import h5py
import numpy as np
import os

LOOKBACK = 12  # 1 hour
HORIZON = 12   # 1 hour
VAL_RATIO = 0.1
TEST_RATIO = 0.2
DATA_PATH = "../data/METR-LA.h5"

with h5py.File(DATA_PATH, "r") as f:
    data = f['df']['block0_values'][:]  # shape (T, 207)

# Normalization
mean = data.mean()
std = data.std()
data_norm = (data - mean) / std

# Generate samples with sliding window
X, y = [], []
for t in range(LOOKBACK, len(data_norm) - HORIZON):
    x_t = data_norm[t - LOOKBACK:t]     # shape (LOOKBACK, 207)
    y_t = data_norm[t:t + HORIZON]      # shape (HORIZON, 207)
    if np.isnan(x_t).any() or np.isnan(y_t).any():
        continue
    X.append(x_t)
    y.append(y_t)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split dataset
num_samples = len(X)
num_test = int(TEST_RATIO * num_samples)
num_val = int(VAL_RATIO * num_samples)
num_train = num_samples - num_val - num_test

X_train, y_train = X[:num_train], y[:num_train]
X_val, y_val = X[num_train:num_train + num_val], y[num_train:num_train + num_val]
X_test, y_test = X[-num_test:], y[-num_test:]

# Save
os.makedirs("../processed_data", exist_ok=True)
np.savez_compressed("../processed_data/metr_la_preprocessed.npz",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    mean=mean, std=std)

print("✅ Data processed and saved to 'processed/metr_la_preprocessed.npz'")
print(f"Train shape: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")