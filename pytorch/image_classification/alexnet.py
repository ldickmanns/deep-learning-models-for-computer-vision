from torch import Tensor
from torch.nn import Conv2d, Dropout, Flatten, Linear, LocalResponseNorm, MaxPool2d, Module, ReLU, Sequential, init


class AlexNet(Module):
    def __init__(self, n_classes: int = 1000):
        super(AlexNet, self).__init__()

        self.feature_extractor = Sequential(
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            ReLU(),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding='same'),
            ReLU(),
            LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding='same'),
            ReLU(),
            Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding='same'),
            ReLU(),
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding='same'),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2)
        )

        self.head = Sequential(
            Flatten(),
            Dropout(p=0.5),
            Linear(in_features=(256 * 6 * 6), out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=4096),
            ReLU(),
            Linear(in_features=4096, out_features=n_classes)
        )

        self.initialize()

    def initialize(self):
        # Feature extractor
        for layer in self.feature_extractor:
            if isinstance(layer, Conv2d):
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)
        init.constant_(self.feature_extractor[4].bias, 1)
        init.constant_(self.feature_extractor[10].bias, 1)
        init.constant_(self.feature_extractor[12].bias, 1)

        # Head
        for layer in self.head:
            if isinstance(layer, Linear):
                init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

    def forward(self, X: Tensor):
        X = self.feature_extractor(X)
        X = self.head(X)
        return X
