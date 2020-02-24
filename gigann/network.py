from typing import List

from gigann import Tensor, State
from gigann.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.tensors = [Tensor(layer.output_shape) for layer in self.layers]
        self.state = State.TRAIN
        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].output_shape

    def forward(self, x: Tensor) -> Tensor:
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
            try:
                self.layers[i].calculate_delta_weights(self.tensors[i], self.tensors[i - 1] if i > 0 else x)
            except AttributeError:
                pass

    def set_state(self, state: State):
        self.state = state
        for layer in self.layers:
            layer.set_state(state)

    def set_batch_size(self, batch_size: int):
        for layer in self.layers:
            layer.set_batch_size(batch_size)
        self.tensors = [Tensor(layer.output_shape) for layer in self.layers]
