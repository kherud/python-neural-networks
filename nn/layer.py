import numpy as np
from typing import List
from abc import ABC, abstractmethod

from nn.base import Tensor
from nn.initializer import Initializer


class Layer(ABC):
    def __init__(self, input_shape: List, output_shape: List) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    @abstractmethod
    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass


class TrainableLayer(Layer, ABC):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape)
        self.initializer = initializer

    @abstractmethod
    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass


class Dense(TrainableLayer):
    def __init__(self, input_shape: List, output_shape: List, initializer: Initializer) -> None:
        super().__init__(input_shape, output_shape, initializer)
        self.W = Tensor([self.input_shape[-1], self.output_shape[-1]], initializer=initializer)
        self.b = Tensor(self.output_shape[-1], initializer=initializer)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.add(in_tensor.x @ self.W.x, self.b.x, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.dot(in_tensor.dx, self.W.x.T, out=out_tensor.dx)

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.sum(in_tensor.dx, axis=0, out=self.b.dx)
        np.dot(out_tensor.x.T, in_tensor.dx, out=self.W.dx)


class LSTM(TrainableLayer):
    def __init__(self, input_shape: List, output_shape: List, initializer) -> None:
        super().__init__(input_shape, output_shape, initializer)
        # states
        self.h = Tensor(input_shape[:-1] + [output_shape[-1]])
        self.c = Tensor(input_shape[:-1] + [output_shape[-1]])

        self.f = Tensor(input_shape[:-1] + [output_shape[-1]])
        self.i = Tensor(input_shape[:-1] + [output_shape[-1]])
        self.o = Tensor(input_shape[:-1] + [output_shape[-1]])
        self.a = Tensor(input_shape[:-1] + [output_shape[-1]])

        # initial state
        self.h_init = Tensor([1, output_shape[-1]])
        self.c_init = Tensor([1, output_shape[-1]])

        # weights
        self.Uf = Tensor([output_shape[-1], output_shape[-1]])
        self.Wf = Tensor([input_shape[-1], output_shape[-1]])
        self.bf = Tensor(output_shape[-1])
        self.Ui = Tensor([output_shape[-1], output_shape[-1]])
        self.Wi = Tensor([input_shape[-1], output_shape[-1]])
        self.bi = Tensor(output_shape[-1])
        self.Uo = Tensor([output_shape[-1], output_shape[-1]])
        self.Wo = Tensor([input_shape[-1], output_shape[-1]])
        self.bo = Tensor(output_shape[-1])
        self.Ua = Tensor([output_shape[-1], output_shape[-1]])
        self.Wa = Tensor([input_shape[-1], output_shape[-1]])
        self.ba = Tensor(output_shape[-1])

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        for step in range(self.input_shape[0]):
            h_step = self.h_init.x if step == 0 else self.h.x[step-1]
            c_step = self.c_init.x if step == 0 else self.c.x[step-1]

            self._forget_gate_forward(in_tensor.x[step], self.f.x[step], h_step)
            self._input_gate_forward(in_tensor.x[step], self.i.x[step], self.a.x[step], h_step)
            self._output_gate_forward(in_tensor.x[step], self.o.x[step],  h_step)
            self._state_update_forward(step, c_step)

        np.copyto(out_tensor.x, self.h.x[-1])

    def _forget_gate_forward(self, _in: np.array, _out: np.array, h: np.array):
        np.add(h @ self.Uf.x + _in @ self.Wf.x, self.bf.x, out=_out)
        self._sigmoid(_out)

    def _input_gate_forward(self, _in: np.array, _i_out: np.array, _a_out: np.array, h: np.array):
        np.add(h @ self.Ui.x + _in @ self.Wi.x, self.bi.x, out=_i_out)
        self._sigmoid(_i_out)
        np.add(h @ self.Ua.x + _in @ self.Wa.x, self.ba.x, out=_a_out)
        self._tanh(_a_out)

    def _output_gate_forward(self, _in: np.array, _out: np.array, h: np.array):
        np.add(h @ self.Uo.x + _in @ self.Wo.x, self.bo.x, out=_out)
        self._sigmoid(_out)

    def _state_update_forward(self, step, c):
        np.add(self.a.x[step] * self.i.x[step], self.f.x[step] * c, out=self.c.x[step])
        np.multiply(np.tanh(self.c.x[step]), self.o.x[step], out=self.h.x[step])

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        for step in reversed(range(self.input_shape[0])):
            self._state_update_backward(step, c_step)
            self._forget_gate_forward(in_tensor.x[step], self.f.x[step], h_step)
            self._input_gate_forward(in_tensor.x[step], self.i.x[step], self.a.x[step], h_step)
            self._output_gate_forward(in_tensor.x[step], self.o.x[step], h_step)
            self._state_update_forward(step, c_step)

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    def _sigmoid(self, _in):
        np.divide(1, (1 + np.exp(-_in)), out=_in)

    def _tanh(self, _in):
        np.tanh(_in, out=_in)