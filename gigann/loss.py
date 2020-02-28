from abc import ABC, abstractmethod

import numpy as np

from gigann import Tensor


class Loss(ABC):
    def __init__(self, input_shape):
        self.loss = Tensor([input_shape[0], 1])
        self.loss.dx = np.ones([input_shape[0], 1])

    @abstractmethod
    def forward(self, prediction: Tensor, truth: Tensor) -> None:
        pass

    @abstractmethod
    def backward(self, prediction: Tensor, truth: Tensor) -> None:
        pass

    def get_loss(self):
        return self.loss.x.sum()


class CrossEntropy(Loss, ABC):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def forward(self, prediction: Tensor, truth: Tensor) -> None:
        np.negative(np.sum(truth.x * np.log(prediction.x), axis=-1, keepdims=True), out=self.loss.x)

    def backward(self, prediction: Tensor, truth: Tensor) -> None:
        np.negative(np.divide(truth.x, prediction.x), out=prediction.dx)


class MeanSquaredError(Loss, ABC):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def forward(self, prediction: Tensor, truth: Tensor) -> None:
        np.divide(np.square(prediction.x - truth.x), 2).sum(axis=-1, keepdims=True, out=self.loss.x)

    def backward(self, prediction: Tensor, truth: Tensor) -> None:
        np.subtract(truth.x, prediction.x, out=prediction.dx)