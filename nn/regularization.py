import numpy as np
from typing import List

from nn.layer import Layer
from nn.base import Tensor


class Dropout(Layer):
    def __init__(self,
                 input_shape: List,
                 rate: float = 0.5):
        super().__init__(input_shape, input_shape)
        self.rate = rate
        self._mask = np.zeros(shape=input_shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        self._mask = np.random.binomial(1, p=self.rate, size=self.input_shape)
        out_tensor.x = np.where(self._mask == 0, in_tensor.x, 0)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        out_tensor.dx = np.where(self._mask == 0, in_tensor.dx, 0)


class LayerNormalization(Layer):
    def __init__(self, input_shape: List):
        super().__init__(input_shape, input_shape)
