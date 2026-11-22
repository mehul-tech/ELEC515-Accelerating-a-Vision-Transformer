"""MNIST data loaders and test accuracy. The dataset is downloaded to data/mnist when missing."""

from pathlib import Path

import torch
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "mnist"


# Pixels are normalized to [-1, 1]; training images are randomly shifted by up to 2 pixels.
def get_loaders(batch_size=256, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2, padding_mode="edge"),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=train_transform)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total
