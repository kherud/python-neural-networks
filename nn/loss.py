from abc import ABC, abstractmethod
import numpy as np
from nn.base import Tensor


class Loss(ABC):
    def __init__(self):
        self.loss = Tensor(1)
        self.loss.dx = np.array([1])

    @abstractmethod
    def forward(self, prediction: Tensor, truth: Tensor) -> None:
        pass

    @abstractmethod
    def backward(self, prediction: Tensor, truth: Tensor) -> None:
        pass


class CrossEntropy(Loss, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, prediction: Tensor, truth: Tensor) -> None:
        np.negative(np.sum(truth.x * np.log(prediction.x)), out=self.loss.x)

    def backward(self, prediction: Tensor, truth: Tensor) -> None:
        np.negative(np.divide(truth.x, prediction.x), out=prediction.dx)