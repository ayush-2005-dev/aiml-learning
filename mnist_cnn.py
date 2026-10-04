import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 16 filters scan the image
    nn.ReLU(),
    nn.MaxPool2d(2),                              # 28x28 -> 14x14
    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 filters look for bigger patterns
    nn.ReLU(),
    nn.MaxPool2d(2),                              # 14x14 -> 7x7
    nn.Flatten(),                                 # 32 x 7 x 7 = 1568 numbers
    nn.Linear(1568, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

print("Parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    model.train()
    return correct / len(test_data)

for epoch in range(3):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch + 1}  avg loss {avg_loss:.4f}  test accuracy {evaluate():.4f}")