import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transform (normalize like MNIST)
from torchvision.transforms import functional as TF

transform = transforms.Compose([
    transforms.Lambda(lambda img: TF.rotate(img, -90)),
    transforms.Lambda(lambda img: TF.hflip(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load EMNIST Balanced (47 classes)
train_dataset = datasets.EMNIST(
    root='../data',
    split='balanced',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root='../data',
    split='balanced',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check class count
print("Number of classes:", len(train_dataset.classes))  # 47 classes

import matplotlib.pyplot as plt
import os
import numpy as np

# Output directory
os.makedirs("emnist_class_grid", exist_ok=True)

# Unnormalize for visualization
def unnormalize(img):
    img = img * 0.3081 + 0.1307
    return img.squeeze().numpy()

# Create dict to collect 1 sample per class
samples = {}
for img, label in train_dataset:
    label_idx = label
    if label_idx not in samples:
        samples[label_idx] = unnormalize(img)
    if len(samples) == len(train_dataset.classes):
        break

# Sort by class index
sorted_samples = [samples[i] for i in sorted(samples.keys())]
sorted_labels = [train_dataset.classes[i] for i in sorted(samples.keys())]

# Plot in a grid
fig, axes = plt.subplots(6, 8, figsize=(10, 8))  # 6x8 grid = 48 (1 extra)
axes = axes.flatten()

for i, (img, label) in enumerate(zip(sorted_samples, sorted_labels)):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(label)
    axes[i].axis('off')

# Hide the unused subplot (48th)
if len(axes) > len(sorted_samples):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig("emnist_class_grid/emnist_balanced_labels_grid.png", dpi=300)
plt.show()
