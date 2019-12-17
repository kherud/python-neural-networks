from abc import ABC
from nn.base import Tensor
import numpy as np
from typing import List


class Layer:
    def __init__(self, input_shape: List, output_shape: List) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        raise NotImplementedError("called abstract method")

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        raise NotImplementedError("called abstract method")


class TrainableLayer(Layer, ABC):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape)
        self.initializer = initializer

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        raise NotImplementedError("called abstract method")


class Dense(TrainableLayer, ABC):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape, initializer)
        self.W = Tensor(self.output_shape + self.input_shape)
        self.b = Tensor(self.output_shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.add(self.W.x @ in_tensor.x, self.b.x, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.dot(self.W.x.T, in_tensor.dx, out=out_tensor.dx)

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.copyto(self.b.x, in_tensor.dx)
        np.dot(in_tensor.dx.reshape(-1, 1), np.expand_dims(out_tensor.x.T, 0), out=self.W.dx)

