from typing import Tuple, List, Callable
from nn.base import Tensor
import nn.layer
import numpy as np


class NeuralNetwork:
    def __init__(self, layers: List[nn.layer.Layer]) -> None:
        self.layers = layers
        self.tensors = [Tensor(layer.output_shape) for layer in self.layers]

    def forward(self, x: Tensor) -> np.array:
        for i in range(len(self.layers)):
            self.layers[i].forward(x, self.tensors[i])
            x = self.tensors[i]
        return x

    def backward(self, x: Tensor) -> None:
        for i in reversed(range(1, len(self.layers))):
            self.layers[i].backward(self.tensors[i], self.tensors[i - 1])
        self.calculate_delta_weights(x)

    def calculate_delta_weights(self, x: Tensor) -> None:
        for i in range(len(self.layers)):
            if not hasattr(self.layers[i], 'calculate_delta_weights'):
                continue
            if i == 0:
                self.layers[i].calculate_delta_weights(self.tensors[i], x)
            else:
                self.layers[i].calculate_delta_weights(self.tensors[i], self.tensors[i - 1])
