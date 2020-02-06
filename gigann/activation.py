import numpy as np
from .layer import Layer
from .base import Tensor
from typing import List


class Sigmoid(Layer):
    def __init__(self, shape: List) -> None:
        super().__init__(shape, shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.divide(1, (1 + np.exp(-in_tensor.x)), out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.x * (1 - in_tensor.x), in_tensor.dx, out=out_tensor.dx)


class Softmax(Layer):
    def __init__(self, shape: List) -> None:
        super().__init__(shape, shape)
        self._exponents = np.empty(shape=shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.exp(in_tensor.x - np.max(in_tensor.x, axis=-1, keepdims=True), out=self._exponents)
        np.divide(self._exponents, self._exponents.sum(axis=-1, keepdims=True), out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.subtract(in_tensor.dx * in_tensor.x, np.einsum("ij,ij,ik->ik", in_tensor.dx, in_tensor.x, in_tensor.x),
                    out=out_tensor.dx)


class Softmax2D(Softmax):
    def __init__(self, shape: List) -> None:
        super().__init__(shape)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.subtract(in_tensor.dx * in_tensor.x, np.einsum("ilj,ilj,ilk->ilk", in_tensor.dx, in_tensor.x, in_tensor.x),
                    out=out_tensor.dx)


class ReLU(Layer):
    def __init__(self, shape: List) -> None:
        super().__init__(shape, shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.x, in_tensor.x > 0, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.multiply(in_tensor.dx, in_tensor.x > 0, out=out_tensor.dx)
