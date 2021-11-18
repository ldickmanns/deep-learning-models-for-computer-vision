from typing import Optional, Union

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from pytorch.utils import get_device

Data = Union[DataLoader, Dataset]


def get_accuracy(model, data: Data, device: str) -> float:
    data_loader = data if isinstance(data, DataLoader) else DataLoader(data)
    n_correct = 0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        _, y_pred = torch.max(outputs, 1)
        n_correct += (y_pred == y).sum().item()

    return n_correct / len(data_loader.dataset)


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    criterion: Loss,
    optimizer: Optimizer,
    device: str
) -> float:
    model.train().to(device)
    epoch_loss = 0.0

    for X, y in tqdm(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        epoch_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader.dataset)
    return epoch_loss


def valid_epoch(
    model: Module,
    valid_loader: DataLoader,
    criterion: Loss,
    device: str
) -> float:
    model.eval().to(device)
    epoch_loss = 0.0

    for X, y in valid_loader:
        X = X.to(device)
        y = y.to(device)

        # Forward
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            epoch_loss += loss.item() * X.size(0)

    epoch_loss /= len(valid_loader.dataset)
    return epoch_loss


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
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        train_losses.append(train_loss)

        if validate:
            valid_loss = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                device=device
            )
            valid_losses.append(valid_loss)

        if verbose:
            train_acc = get_accuracy(model, train_loader, device)
            valid_acc = get_accuracy(model, valid_loader, device) if validate else 0.0
            print(
                f'Epoch: {epoch + 1} - ' +
                f'loss: {train_loss} - ' +
                f'accuracy: {train_acc} - ' +
                (f'val_loss: {valid_loss} - ' +
                 f'val_accuracy: {valid_acc}' if validate else '')
            )
