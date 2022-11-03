import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os


def get_mnist(config):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    batch_size = config["batch_size"]
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_loader, test_loader


def make_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else:
        print(path_name, "already exists.")
