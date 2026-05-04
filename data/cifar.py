import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os

def get_transforms():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

def get_dataloaders(data_dir, batch_size, num_workers=2, val_size=5000, seed=42):
    transform = get_transforms()
    train_full = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
        )
    
    #fixed train/val split — same every run
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_full))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(train_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_test_loader(data_dir, batch_size, num_workers=2):
    transform = get_transforms()
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader