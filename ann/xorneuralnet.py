import torch
import torch.nn as nn
import torch.optim as optim

class XORNeuralNet(nn.Module):
    def __init__(self):
        super(XORNeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class XORTrainer:
    def __init__(self, learning_rate=0.1, epochs=10000):
        # Dataset
        self.X = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.Y = torch.tensor([
            [0.],
            [1.],
            [1.],
            [0.]
        ])

        # Model, loss, optimizer
        self.model = XORNeuralNet()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self):
        print("Training started...")
        for epoch in range(self.epochs):
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        print("Training finished!")

    def evaluate(self):
        print("\nPredictions:")
        with torch.no_grad():
            predictions = self.model(self.X)
            predicted_classes = predictions.round()

            for input_val, prediction in zip(self.X, predicted_classes):
                print(f"Input: {input_val.tolist()} => Predicted: {int(prediction.item())}")


if __name__ == "__main__":
    xor_trainer = XORTrainer()
    xor_trainer.train()
    xor_trainer.evaluate()
