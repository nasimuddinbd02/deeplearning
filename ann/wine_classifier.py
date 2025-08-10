# PyTorch implementation
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 0) reproducibility + device
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) load and split
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 2) scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3) to tensors + dataloaders
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# 4) model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, out_dim=3, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, out_dim)   # final layer: raw logits
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1], out_dim=len(np.unique(y))).to(device)

# 5) loss + optimizer
criterion = nn.CrossEntropyLoss()   # expects raw logits and integer class labels
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6) training loop
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)                 # shape (batch, classes)
        loss = criterion(out, yb)       # CrossEntropy combines LogSoftmax + NLLLoss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # validation on test set (quick check)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# 7) final test metrics (detailed)
model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.cpu().numpy())

print(classification_report(all_true, all_preds, target_names=data.target_names))
print("Test accuracy:", accuracy_score(all_true, all_preds))
print("Confusion matrix:\n", confusion_matrix(all_true, all_preds))
