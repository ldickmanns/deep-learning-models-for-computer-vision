from __future__ import annotations

from typing import Callable

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import AvgPool2D, Conv2D, Flatten, Dense, Softmax, Layer
from tensorflow.python.keras.layers.pooling import Pooling2D
from tensorflow.keras.activations import tanh


class LeNet5(Model):
    def __init__(
        self,
        input_shape: tuple[int, int] | tuple[int, int, int],
        n_classes: int,
        activation: Callable | Layer | None = tanh,
        head_activation: Callable | Layer | None = None,
        pooling: Pooling2D = AvgPool2D(),
    ):
        super(LeNet5, self).__init__()

        if head_activation is None:
            head_activation = activation

        self.feature_extractor = Sequential(
            Conv2D(filters=6, kernel_size=5, padding='same', input_shape=input_shape, activation=activation),
            pooling,
            Conv2D(filters=16, kernel_size=5, activation=activation),
            pooling,
            Conv2D(filters=120, kernel_size=5, activation=activation),
        )

        self.head = Sequential(
            Flatten(),
            Dense(84, activation=head_activation),
            Dense(n_classes),
            Softmax(),
        )

    def call(self, inputs, training=None, mask=None):
        X = self.feature_extractor(inputs)
        X = self.head(X)
        return X

    def get_config(self):
        pass
