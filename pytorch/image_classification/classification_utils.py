from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, VisionDataset
from tqdm import tqdm

from pytorch.utils import get_device

Data = Union[DataLoader, Dataset]


def _n_correct(outputs: Tensor, y: Tensor) -> int:
    _, y_pred = torch.max(outputs, 1)
    n_correct = (y_pred == y).sum().item()
    return n_correct


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    criterion: Loss,
    optimizer: Optimizer,
    device: str
) -> tuple[float, float]:
    model.train().to(device)
    epoch_loss = 0.0
    n_correct = 0

    for X, y in tqdm(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        epoch_loss += loss.item() * X.size(0)
        n_correct += _n_correct(outputs, y)

        # Backward pass
        loss.backward()
        optimizer.step()

    dataset_size = len(train_loader.dataset)
    epoch_loss /= dataset_size
    epoch_acc = n_correct / dataset_size
    return epoch_loss, epoch_acc


def valid_epoch(
    model: Module,
    valid_loader: DataLoader,
    criterion: Loss,
    device: str
) -> tuple[float, float]:
    model.eval().to(device)
    epoch_loss = 0.0
    n_correct = 0

    for X, y in valid_loader:
        X = X.to(device)
        y = y.to(device)

        # Forward
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            epoch_loss += loss.item() * X.size(0)
            n_correct += _n_correct(outputs, y)

    dataset_size = len(valid_loader.dataset)
    epoch_loss /= dataset_size
    epoch_acc = n_correct / dataset_size
    return epoch_loss, epoch_acc


def train(
    model: Module,
    train_dataset: VisionDataset,
    epochs: int,
    optimizer: Optimizer = None,
    criterion: Loss = None,
    valid_dataset: Optional[VisionDataset] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    verbose: bool = True,
):
    device = get_device()
    model.to(device)

    if optimizer is None:
        optimizer = Adam(model.parameters())

    if criterion is None:
        criterion = CrossEntropyLoss()

    validate = valid_dataset is not None

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    if validate:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        train_losses.append(train_loss)

        if validate:
            valid_loss, valid_acc = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                device=device
            )
            valid_losses.append(valid_loss)

        if verbose:
            print(
                f'Epoch: {epoch + 1} - ' +
                f'loss: {train_loss} - ' +
                f'accuracy: {train_acc} - ' +
                (f'val_loss: {valid_loss} - ' +
                 f'val_accuracy: {valid_acc}' if validate else '')
            )


def load_normalized_mnist() -> tuple[MNIST, MNIST, MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root='../data/mnist', train=True, transform=transform, download=True)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    test_dataset = MNIST(root='../data/mnist', train=False, transform=transform, download=True)

    return train_dataset, valid_dataset, test_dataset
