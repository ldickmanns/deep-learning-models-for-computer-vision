from typing import Optional

from torch.nn import AvgPool2d, Conv2d, Flatten, Linear, Module, Sequential, Sigmoid, Softmax


class LeNet5(Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        activation: Optional[Module],
        head_activation: Optional[Module] = None,
        pooling: Optional[Module] = None,
    ):
        super(LeNet5, self).__init__()

        if activation is None:
            activation = Sigmoid()

        if head_activation is None:
            head_activation = activation

        if pooling is None:
            pooling = AvgPool2d(2)

        self.feature_extractor = Sequential(
            Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5),
            activation,
            pooling,
            Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            activation,
            pooling,
            Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            activation,
        )

        self.head = Sequential(
            Flatten(),
            Linear(in_features=120, out_features=84),
            head_activation,
            Linear(in_features=84, out_features=n_classes),
            Softmax(dim=n_classes)
        )

    def forward(self, X):
        X = self.feature_extractor(X)
        X = self.head(X)
        return X
