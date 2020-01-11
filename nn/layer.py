from abc import ABC, abstractmethod
from nn.base import Tensor
import numpy as np
from typing import List


class Layer(ABC):
    def __init__(self, input_shape: List, output_shape: List) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    @abstractmethod
    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass


class TrainableLayer(Layer, ABC):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape)
        self.initializer = initializer

    @abstractmethod
    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass


class Dense(TrainableLayer):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape, initializer)
        self.W = Tensor(self.input_shape + self.output_shape)
        self.b = Tensor(self.output_shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.add(in_tensor.x @ self.W.x, self.b.x, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.dot(in_tensor.dx, self.W.x.T, out=out_tensor.dx)

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.copyto(self.b.dx, in_tensor.dx)
        np.dot(out_tensor.x.T, in_tensor.dx, out=self.W.dx)

