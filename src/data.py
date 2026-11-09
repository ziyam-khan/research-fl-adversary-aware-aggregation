import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def dirichlet_split(dataset, num_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[i].extend(chunk.tolist())

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def get_mnist(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def get_cifar10(data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    return train, test


def make_client_loaders(dataset, client_indices, batch_size=32):
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return loaders


def get_client_class_distribution(dataset, client_indices):
    targets = np.array(dataset.targets)
    distributions = []
    for indices in client_indices:
        classes, counts = np.unique(targets[indices], return_counts=True)
        dist = np.zeros(10)
        dist[classes] = counts / counts.sum()
        distributions.append(dist)
    return np.array(distributions)
