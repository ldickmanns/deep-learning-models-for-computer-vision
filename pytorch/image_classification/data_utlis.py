from os.path import join
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

DATA_PATH = join(Path(__file__).parent.parent.parent, 'data')


# def load_normalized_imagenet() -> tuple[...]:
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     datasets = ImageNet(root='../data/imagenet', split='val', transform=transform)
#     return datasets

def load_normalized_cifar10() -> tuple[CIFAR10, CIFAR10, CIFAR10]:
    pass


def load_normalized_cifar100() -> tuple[CIFAR10, CIFAR10, CIFAR10]:
    pass


def load_normalized_mnist() -> tuple[MNIST, MNIST, MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root=DATA_PATH, train=True, transform=transform, download=True)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    test_dataset = MNIST(root=DATA_PATH, train=False, transform=transform, download=True)

    return train_dataset, valid_dataset, test_dataset
