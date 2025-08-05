import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Step 1: Custom Dataset for XOR
class XORDataset(Dataset):
    def __init__(self):
        self.inputs = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.labels = torch.tensor([
            [0.],
            [1.],
            [1.],
            [0.]
        ])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# Step 2: Neural Network Model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Step 3: Trainer Class that Ties It All Together
class XORTrainer:
    def __init__(self, batch_size=2, learning_rate=0.1, epochs=10000):
        # Dataset & DataLoader
        self.dataset = XORDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Model, loss, optimizer
        self.model = XORNet()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self):
        print("Training started...\n")
        for epoch in range(self.epochs):
            for inputs, labels in self.dataloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")
        print("\nTraining finished!")

    def evaluate(self):
        print("\nPredictions:")
        with torch.no_grad():
            for inputs, label in self.dataset:
                prediction = self.model(inputs)
                pred_class = int(prediction.round().item())
                print(f"Input: {inputs.tolist()} => Predicted: {pred_class}, Actual: {int(label.item())}")


# Step 4: Run the Whole Pipeline
if __name__ == "__main__":
    trainer = XORTrainer()
    trainer.train()
    trainer.evaluate()
