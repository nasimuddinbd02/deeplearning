import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------
# 1. Data Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                     # Convert image to Tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))       # Normalize to [-1, 1]
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 2. Define CNN Model using OOP
# ---------------------------
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Conv Layer 1: 1 input channel (grayscale), 8 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        
        # Conv Layer 2: 8 input channels, 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # after pooling, feature map size is 5x5
        
        # Fully Connected Layer 2 (Output Layer)
        self.fc2 = nn.Linear(120, 10)          # 10 classes in FashionMNIST

    def forward(self, x):
        # First conv + activation + pooling
        x = F.relu(self.conv1(x))          # Shape: (batch, 8, 26, 26)
        x = F.max_pool2d(x, 2)              # Shape: (batch, 8, 13, 13)

        # Second conv + activation + pooling
        x = F.relu(self.conv2(x))          # Shape: (batch, 16, 11, 11)
        x = F.max_pool2d(x, 2)              # Shape: (batch, 16, 5, 5)

        # Flatten before Fully Connected Layer
        x = x.view(x.size(0), -1)           # Shape: (batch, 400)

        # Fully connected layers
        x = F.relu(self.fc1(x))             # Shape: (batch, 120)
        x = self.fc2(x)                     # Shape: (batch, 10) → raw scores (logits)
        return x

# ---------------------------
# 3. Initialize model, loss, optimizer
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 4. Training Loop
# ---------------------------
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)              # Forward pass
        loss = criterion(outputs, labels)    # Compute loss
        loss.backward()                       # Backpropagation
        optimizer.step()                      # Update weights

        running_loss += loss.item()
    return running_loss / len(loader)

# ---------------------------
# 5. Evaluation Loop
# ---------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Predictions
            predicted = outputs.argmax(dim=1)   # Choose highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# ---------------------------
# 6. Run Training
# ---------------------------
epochs = 5
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Acc: {test_acc:.2f}%")
