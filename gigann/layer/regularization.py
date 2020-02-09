from typing import List

import numpy as np

from gigann.initializer import ones, zeros
from gigann.layer import Layer
from gigann import Tensor, State


class Dropout(Layer):
    def __init__(self,
                 input_shape: List,
                 rate: float = 0.5):
        super().__init__(input_shape, input_shape)
        self.rate = rate
        self._mask = np.zeros(shape=input_shape)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        if self._state is State.TRAIN:
            self._mask = (np.random.random(size=self.input_shape) < self.rate) / (1 - self.rate)
            np.multiply(in_tensor.x, self._mask, out=out_tensor.x)
        else:
            np.copyto(out_tensor.x, in_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        if self._state is State.TRAIN:
            np.multiply(in_tensor.dx, self._mask, out=out_tensor.dx)
        else:
            np.copyto(out_tensor.x, in_tensor.x)


class LayerNormalization(Layer):
    def __init__(self, input_shape: List):
        super().__init__(input_shape, input_shape)
        self.gamma = Tensor(input_shape[-1:], initializer=ones)
        self.beta = Tensor(input_shape[-1:], initializer=zeros)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.divide(in_tensor.x - in_tensor.x.mean(axis=-1, keepdims=True),
                  in_tensor.x.std(axis=-1, keepdims=True),
                  out=out_tensor.x)
        out_tensor.x *= self.gamma
        out_tensor.x += self.beta

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        raise NotImplementedError()
