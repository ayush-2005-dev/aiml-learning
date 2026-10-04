import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# Same iris data as before
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Neural networks train better when inputs are on a similar scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 4 inputs -> 16 hidden neurons -> 3 outputs (one per species)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()                    # clear old gradients
    loss = loss_fn(model(X_train), y_train)  # forward pass + loss
    loss.backward()                          # compute gradients
    optimizer.step()                         # update the weights
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
    accuracy = (predictions == y_test).float().mean().item()

print("Test accuracy:", accuracy)