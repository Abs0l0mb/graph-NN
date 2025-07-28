import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pickle
import matplotlib.pyplot as plt  # <-- for plotting


from models.A3TGCN import A3TGCN  # your model class

# Load data
data = np.load("./processed_data/metr_la_preprocessed.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)
X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32)
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.float32)

print("Data loaded")

# Load adjacency matrix and build edge_index
with open("./data/adj_METR-LA.pkl", "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

edge_index, _ = dense_to_sparse(torch.tensor(adj_mx, dtype=torch.float32))

print("Adjacency matrix loaded")

# Dataloaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

print("Dataloaders done")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

num_nodes = X_train.shape[2]
print("Num_nodes : ", num_nodes)

model = A3TGCN(node_features=2, periods=12, batch_size=batch_size).to(device)
print("Model : ", model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
val_losses   = []

for epoch in range(20):
    model.train()
    epoch_train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch, edge_index.to(device))
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch, edge_index.to(device))
            epoch_val_loss += criterion(y_pred, y_batch).item()
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1:02d}: Train {epoch_train_loss:.4f} | Val {epoch_val_loss:.4f}")

# Testing (unchanged)
model.eval()
test_loss = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch, edge_index.to(device))
        test_loss += criterion(y_pred, y_batch).item()
test_loss /= len(test_loader)
print(f"\n🧪 Test MSE: {test_loss:.4f}")

# --- New: save the model weights ---
torch.save(model.state_dict(), "a3tgnn_metrla.pth")
print("Model weights saved to a3tgnn_metrla.pth")

# --- New: plot loss curves ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val   Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()