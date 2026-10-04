import matplotlib.pyplot as plt
from torchvision import datasets, transforms

train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=transforms.ToTensor()
)

print("Training images:", len(train_data))
print("Test images:", len(test_data))

image, label = train_data[0]
print("One image shape:", image.shape)
print("Pixel range:", image.min().item(), "to", image.max().item())
print("First label:", label)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, lbl = train_data[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Label: {lbl}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()