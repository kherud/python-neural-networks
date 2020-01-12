import tqdm
import logging
import numpy as np
from typing import List
from abc import ABC, abstractmethod

from nn.base import Tensor
from nn.loss import Loss
from nn.metrics import Metric
from nn.network import NeuralNetwork


class Optimizer(ABC):
    def __init__(self, loss: Loss):
        self.loss = loss

    def optimize(self, neural_network: NeuralNetwork,
                 x_train: List[Tensor],
                 y_train: List[Tensor],
                 x_test: List[Tensor] = None,
                 y_test: List[Tensor] = None,
                 epochs: int = 1,
                 batch_size: int = 16,
                 metrics: List[Metric] = None):
        for epoch in range(epochs):
            desc = "train {}/{} loss = {{:.3f}}".format(epoch + 1, epochs)
            pbar = tqdm.trange(len(x_train), desc=desc.format(np.inf))
            loss_total = 0
            for i in pbar:
                prediction = neural_network.forward(x_train[i])

                self.loss.forward(prediction, y_train[i])
                loss = self.loss.get_loss()
                loss_total -= loss_total / (i + 1) - loss / (i + 1)
                pbar.set_description(desc.format(loss_total))
                self.loss.backward(prediction, y_train[i])

                neural_network.backward(x_train[i])

                self._optimize_layers(neural_network)

            if x_test is not None and y_test is not None:
                self._evaluate(neural_network, x_test, y_test, metrics)

    def _evaluate(self, neural_network, x_test, y_test, metrics):
            predictions = []
            truths = []
            loss_total = 0
            # pbar = tqdm.trange(len(x_test), desc="test")
            for i in range(len(x_test)):
                prediction = neural_network.forward(x_test[i])
                predictions.extend(np.argmax(prediction.x, axis=1))
                truths.extend(np.argmax(y_test[i].x, axis=1))

                self.loss.forward(prediction, y_test[i])
                loss = self.loss.get_loss()
                loss_total -= loss_total / (i + 1) - loss / (i + 1)
            desc = "test: loss = {:.4f}".format(loss)
            if metrics is not None:
                for metric in metrics:
                    metric.evaluate(np.array(predictions), np.array(truths))
                    desc += "; {}".format(str(metric))
            print(desc)
            # pbar.set_description(desc)


    def _optimize_layers(self, neural_network):
        for layer in neural_network.layers:
            if not hasattr(layer, 'calculate_delta_weights'):
                continue
            self._optimize_parameters(layer.W)
            self._optimize_parameters(layer.b)

    @abstractmethod
    def _optimize_parameters(self, tensor: Tensor):
        pass


class MinibatchGradientDescent(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate

    def _optimize_parameters(self, tensor: Tensor):
        tensor.x -= self.learning_rate * tensor.dx


class Momentum(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate
        self.states = {}
        self.mu = 0.9

    def _optimize_parameters(self, tensor: Tensor):
        if tensor not in self.states:
            self.states[tensor] = {
                "v": np.zeros(shape=tensor.shape),
            }
        state = self.states[tensor]

        state["v"] = self.mu * state["v"] - self.learning_rate * tensor.dx

        tensor.x += state["v"]


class Adagrad(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate
        self.states = {}
        self.eps = 1e-8

    def _optimize_parameters(self, tensor: Tensor):
        if tensor not in self.states:
            self.states[tensor] = {
                "cache": np.zeros(shape=tensor.shape),
            }
        state = self.states[tensor]

        state["cache"] += tensor.dx ** 2

        tensor.x -= self.learning_rate * tensor.dx / (np.sqrt(state["cache"]) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate
        self.states = {}
        self.eps = 1e-8
        self.decay_rate = 0.99

    def _optimize_parameters(self, tensor: Tensor):
        if tensor not in self.states:
            self.states[tensor] = {
                "cache": np.zeros(shape=tensor.shape),
            }
        state = self.states[tensor]

        state["cache"] = self.decay_rate * state["cache"] + (1 - self.decay_rate) * tensor.dx ** 2

        tensor.x -= self.learning_rate * tensor.dx / (np.sqrt(state["cache"]) + self.eps)


class SimpleAdam(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate
        self.states = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _optimize_parameters(self, tensor: Tensor):
        if tensor not in self.states:
            self.states[tensor] = {
                "m": np.zeros(shape=tensor.shape),
                "v": np.zeros(shape=tensor.shape),
            }
        state = self.states[tensor]

        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * tensor.dx
        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (tensor.dx ** 2)

        tensor.x -= self.learning_rate * state["m"] / (np.sqrt(state["v"]) + self.eps)

class Adam(Optimizer):
    def __init__(self, loss: Loss, learning_rate: float = 1e-3):
        super().__init__(loss)
        self.learning_rate = learning_rate
        self.states = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _optimize_parameters(self, tensor: Tensor):
        if tensor not in self.states:
            self.states[tensor] = {
                "m": np.zeros(shape=tensor.shape),
                "v": np.zeros(shape=tensor.shape),
                "t": np.ones(shape=tensor.shape),
                "mt": np.zeros(shape=tensor.shape),
                "vt": np.zeros(shape=tensor.shape),
            }
        state = self.states[tensor]

        np.add(self.beta1 * state["m"], (1 - self.beta1) * tensor.dx, out=state["m"])
        np.divide(state["m"], 1 - self.beta1 ** state["t"], out=state["mt"])

        np.add(self.beta2 * state["v"], (1 - self.beta2) * (tensor.dx ** 2), out=state["v"])
        np.divide(state["v"], 1 - self.beta2 ** state["t"], out=state["vt"])

        tensor.x -= self.learning_rate * state["mt"] / (np.sqrt(state["vt"]) + self.eps)