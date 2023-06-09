import os
import tqdm
import pickle
import numpy as np
from typing import List, Callable, Iterable
from abc import ABC, abstractmethod

from gigann import Tensor, State
from gigann.loss import Loss
from gigann.network import NeuralNetwork


class Optimizer(ABC):
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None):
        self.loss = loss
        self.weight_decay = weight_decay

    def optimize(self, neural_network: NeuralNetwork,
                 x_train: List[Tensor],
                 y_train: List[Tensor],
                 x_test: List[Tensor] = None,
                 y_test: List[Tensor] = None,
                 epochs: int = 1,
                 metrics: Iterable[Callable] = (),
                 save_to: str = None,
                 save_by: str = None):
        best_evaluation = 0
        for epoch in range(epochs):
            neural_network.set_state(State.TRAIN)
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
                evaluation = self._evaluate(neural_network, x_test, y_test, metrics)

                eval_desc = "test " + ", ".join(f"{k} = {v:.3f}" for k, v in evaluation.items())
                pbar.write(eval_desc)

                if save_by is not None:
                    assert save_by in evaluation, "save_by metric not specified"
                if save_by in evaluation and evaluation[save_by] <= best_evaluation:
                    continue
                elif save_by in evaluation:
                    best_evaluation = evaluation[save_by]
                if save_to:
                    path = os.path.join(save_to, f"model-{best_evaluation:.3f}.pkl")
                    with open(path, "wb") as file:
                        pickle.dump(neural_network, file)

    def _evaluate(self,
                  neural_network: NeuralNetwork,
                  x_test: List[Tensor],
                  y_test: List[Tensor],
                  metrics: Iterable[Callable]) -> dict:
        neural_network.set_state(State.PREDICT)
        predictions = []
        truths = []
        loss_total = 0
        for i in range(len(x_test)):
            prediction = neural_network.forward(x_test[i])
            predictions.extend(np.argmax(prediction.x, axis=1))
            truths.extend(np.argmax(y_test[i].x, axis=1))

            self.loss.forward(prediction, y_test[i])
            loss = self.loss.get_loss()
            loss_total -= loss_total / (i + 1) - loss / (i + 1)
        evaluation = {metric.__name__: metric(predictions, truths) for metric in metrics}
        evaluation["loss"] = loss_total
        return evaluation

    def _optimize_layers(self, neural_network):
        for layer in neural_network.layers:
            try:
                for weight in layer.get_weights():
                    if self.weight_decay:
                        weight.dx += self.weight_decay * weight.x
                    self._optimize_parameters(weight)
                for bias in layer.get_bias():
                    self._optimize_parameters(bias)
            except AttributeError:
                pass

    @abstractmethod
    def _optimize_parameters(self, tensor: Tensor):
        pass


class MinibatchGradientDescent(Optimizer):
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
        self.learning_rate = learning_rate

    def _optimize_parameters(self, tensor: Tensor):
        tensor.x -= self.learning_rate * tensor.dx


class Momentum(Optimizer):
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
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
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
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
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
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
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
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
    def __init__(self,
                 loss: Loss,
                 weight_decay: float = None,
                 learning_rate: float = 1e-3):
        super().__init__(loss, weight_decay)
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
