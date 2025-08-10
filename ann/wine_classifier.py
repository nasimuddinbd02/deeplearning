import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class WineClassifier:
    def __init__(self, hidden1=64, hidden2=32, p_drop=0.2, lr=1e-3, batch_size=16, epochs=100, seed=42):
        # Reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.p_drop = p_drop
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = None
        self.criterion = None
        self.optimizer = None
        
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden1, hidden2, out_dim, p_drop):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden2, out_dim)
            )
        def forward(self, x):
            return self.net(x)
        
    def load_data(self):
        """Load and split dataset"""
        data = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
        )
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        # Data loaders
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_train))
        self.target_names = data.target_names
        
    def build_model(self):
        """Initialize model, loss, and optimizer"""
        self.model = self.MLP(self.input_dim, self.hidden1, self.hidden2, self.output_dim, self.p_drop).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self):
        """Train the model"""
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss, correct, total = 0, 0, 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation each epoch
            val_loss, val_acc = self.evaluate(return_metrics=False)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    def evaluate(self, return_metrics=True):
        """Evaluate model on test set"""
        self.model.eval()
        loss_total, correct, total = 0, 0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        avg_loss = loss_total / total
        acc = correct / total
        if return_metrics:
            print("Test Accuracy:", acc)
            print("Classification Report:\n", classification_report(all_true, all_preds, target_names=self.target_names))
            print("Confusion Matrix:\n", confusion_matrix(all_true, all_preds))
        else:
            return avg_loss, acc
        
    def run(self):
        """Full pipeline"""
        self.load_data()
        self.build_model()
        self.train()
        self.evaluate()

if __name__ == "__main__":
    classifier = WineClassifier(epochs=50)  # fewer epochs for quick test
    classifier.run()
