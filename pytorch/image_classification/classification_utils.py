from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


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
