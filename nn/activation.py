from abc import ABC
import numpy as np
from nn.layer import Layer
from nn.base import Tensor
from typing import List


class Sigmoid(Layer, ABC):
    def __init__(self, shape) -> None:
        super().__init__(shape, shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.divide(1, (1 + np.exp(-in_tensor.x)), out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.x * (1 - in_tensor.x), in_tensor.dx, out=out_tensor.dx)


class Softmax(Layer, ABC):
    def __init__(self, shape: List) -> None:
        super().__init__(shape, shape)
        self._exponents = np.empty(shape=shape).astype(np.float32)
        self._jacobian = np.empty(shape=shape + shape).astype(np.float32)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.exp(in_tensor.x - np.max(in_tensor.x), out=self._exponents)
        np.divide(self._exponents, self._exponents.sum(axis=0), out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        s = in_tensor.x.reshape(-1, 1)
        np.subtract(np.diagflat(s), np.dot(s, s.T), out=self._jacobian)
        np.dot(in_tensor.dx, self._jacobian, out=out_tensor.dx)


class ReLU(Layer, ABC):
    def __init__(self, shape) -> None:
        super().__init__(shape, shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.x, in_tensor.x > 0, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.dx, in_tensor.x > 0, out=out_tensor.dx)
