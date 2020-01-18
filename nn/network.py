from enum import Enum
from typing import Tuple, List, Callable
from nn.base import Tensor
import nn.layer
import numpy as np


class State(Enum):
    TRAIN = 1
    PREDICT = 2


class NeuralNetwork:
    def __init__(self, layers: List[nn.layer.Layer]) -> None:
        self.layers = layers
        self.tensors = [Tensor(layer.output_shape) for layer in self.layers]
        self.state = State.TRAIN


    def forward(self, x: Tensor) -> np.array:
        for i in range(len(self.layers)):
            self.layers[i].forward(x, self.tensors[i])
            x = self.tensors[i]
        return x

    def backward(self, x: Tensor) -> None:
        for i in reversed(range(len(self.layers))):
            self.layers[i].backward(self.tensors[i], self.tensors[i - 1] if i > 0 else x)
        self.calculate_delta_weights(x)

    def calculate_delta_weights(self, x: Tensor) -> None:
        for i in range(len(self.layers)):
            if not hasattr(self.layers[i], 'calculate_delta_weights'):
                continue
            self.layers[i].calculate_delta_weights(self.tensors[i], self.tensors[i - 1] if i > 0 else x)

    def set_state(self, state: State):
        self.state = state
        for layer in self.layers:
            layer._set_state(state)