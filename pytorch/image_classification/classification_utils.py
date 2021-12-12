from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from ..utils import get_device

Data = Union[DataLoader, VisionDataset]


def _n_correct(outputs: Tensor, y: Tensor) -> int:
    _, y_pred = torch.max(outputs, 1)
    n_correct = (y_pred == y).sum().item()
    return n_correct


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Loss,
    device: str,
    verbose: bool = True,
) -> tuple[float, float]:
    model.train().to(device)
    epoch_loss = 0.0
    n_correct = 0

    _iterator = tqdm(train_loader, ncols=69) if verbose else train_loader
    for X, y in _iterator:
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
    train_data: Data,
    epochs: int,
    optimizer: Optimizer = None,
    criterion: Loss = None,
    valid_data: Optional[Data] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    verbose: int = 2,
    device: str = None
):
    device = device if device is not None else get_device()
    model.to(device)

    optimizer = optimizer if optimizer is not None else Adam(model.parameters())
    criterion = criterion if criterion is not None else CrossEntropyLoss()

    validate = valid_data is not None

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=shuffle
    )
    if validate:
        valid_loader = valid_data if isinstance(valid_data, DataLoader) else DataLoader(
            dataset=valid_data, batch_size=batch_size, shuffle=shuffle
        )

    history: dict[str, list[float]] = {'loss': [], 'accuracy': []}

    if validate:
        history['val_loss'] = []
        history['val_accuracy'] = []

    _epoch_range = range(epochs)
    _iterator = tqdm(_epoch_range, ncols=69) if verbose == 1 else _epoch_range
    for epoch in _iterator:
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            verbose=verbose == 2,
        )
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)

        if validate:
            valid_loss, valid_acc = valid_epoch(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                device=device,
            )
            history['val_loss'].append(valid_loss)
            history['val_accuracy'].append(valid_acc)

        if verbose == 2:
            print(
                f'Epoch: {epoch + 1} - ' +
                f'loss: {train_loss:.5f} - ' +
                f'accuracy: {train_acc:.5f} - ' +
                (f'val_loss: {valid_loss:.5f} - ' +
                 f'val_accuracy: {valid_acc:.5f}' if validate else '')
            )

    return history
