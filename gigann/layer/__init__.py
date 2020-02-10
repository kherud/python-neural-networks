from typing import List, Callable, Tuple
from abc import ABC, abstractmethod

from gigann import Tensor, State


class Layer(ABC):
    def __init__(self, input_shape: List, output_shape: List) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._state = State.TRAIN

    @abstractmethod
    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    @abstractmethod
    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    def _set_state(self, state: State):
        self._state = state


class TrainableLayer(Layer, ABC):
    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 weights_initializer: Callable,
                 bias_initializer: Callable) -> None:
        super().__init__(input_shape, output_shape)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    @abstractmethod
    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    @abstractmethod
    def get_weights(self) -> Tuple:
        pass

    @abstractmethod
    def get_bias(self) -> Tuple:
        pass 
