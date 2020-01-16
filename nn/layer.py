import numpy as np
from typing import List, Callable
from abc import ABC, abstractmethod

from nn.base import Tensor
from nn.initializer import zeros, ones, xavier_normal


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


class Dense(TrainableLayer):
    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 weights_initializer: Callable = xavier_normal,
                 bias_initializer: Callable = zeros) -> None:
        super().__init__(input_shape, output_shape, weights_initializer, bias_initializer)
        self.W = Tensor([self.input_shape[-1], self.output_shape[-1]], initializer=weights_initializer)
        self.b = Tensor([self.output_shape[-1]], initializer=bias_initializer)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.add(in_tensor.x @ self.W.x, self.b.x, out=out_tensor.x)

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.dot(in_tensor.dx, self.W.x.T, out=out_tensor.dx)

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.sum(in_tensor.dx, axis=0, out=self.b.dx)
        np.dot(out_tensor.x.T, in_tensor.dx, out=self.W.dx)


class LSTM(TrainableLayer):
    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 weights_initializer: Callable = xavier_normal,
                 bias_initializer: Callable = zeros) -> None:
        super().__init__(input_shape, output_shape, weights_initializer, bias_initializer)
        # states (batch size, steps, features)
        self.h = Tensor(input_shape[:-1][::-1] + [output_shape[-1]])
        self.c = Tensor(input_shape[:-1][::-1] + [output_shape[-1]])

        # initial state
        self.h_init = Tensor([1, output_shape[-1]], initializer=zeros)
        self.c_init = Tensor([1, output_shape[-1]], initializer=zeros)

        # gates
        self.gates = Tensor(input_shape[:-1][::-1] + [4 * output_shape[-1]])
        self.a = Tensor(input_shape[:-1][::-1] + [output_shape[-1]],
                        x=self.gates.x[:, :, :output_shape[-1]],
                        dx=self.gates.dx[:, :, :output_shape[-1]],
                        reference=True)
        self.i = Tensor(input_shape[:-1][::-1] + [output_shape[-1]],
                        x=self.gates.x[:, :, output_shape[-1]:output_shape[-1] * 2],
                        dx=self.gates.dx[:, :, output_shape[-1]:output_shape[-1] * 2],
                        reference=True)
        self.f = Tensor(input_shape[:-1][::-1] + [output_shape[-1]],
                        x=self.gates.x[:, :, output_shape[-1] * 2:output_shape[-1] * 3],
                        dx=self.gates.dx[:, :, output_shape[-1] * 2:output_shape[-1] * 3],
                        reference=True)
        self.o = Tensor(input_shape[:-1][::-1] + [output_shape[-1]],
                        x=self.gates.x[:, :, output_shape[-1] * 3:output_shape[-1] * 4],
                        dx=self.gates.dx[:, :, output_shape[-1] * 3:output_shape[-1] * 4],
                        reference=True)

        # weights
        self.W = Tensor([input_shape[-1], 4 * output_shape[-1]])
        self.Wa = Tensor([input_shape[-1], output_shape[-1]],
                         x=self.W.x[:, :output_shape[-1]],
                         dx=self.W.dx[:, :output_shape[-1]],
                         initializer=weights_initializer,
                         reference=True)
        self.Wi = Tensor([input_shape[-1], output_shape[-1]],
                         x=self.W.x[:, output_shape[-1]:output_shape[-1] * 2],
                         dx=self.W.dx[:, output_shape[-1]:output_shape[-1] * 2],
                         initializer=weights_initializer,
                         reference=True)
        self.Wf = Tensor([input_shape[-1], output_shape[-1]],
                         x=self.W.x[:, output_shape[-1] * 2:output_shape[-1] * 3],
                         dx=self.W.dx[:, output_shape[-1] * 2:output_shape[-1] * 3],
                         initializer=weights_initializer,
                         reference=True)
        self.Wo = Tensor([input_shape[-1], output_shape[-1]],
                         x=self.W.x[:, output_shape[-1] * 3:output_shape[-1] * 4],
                         dx=self.W.dx[:, output_shape[-1] * 3:output_shape[-1] * 4],
                         initializer=weights_initializer,
                         reference=True)

        self.U = Tensor([output_shape[-1], 4 * output_shape[-1]])
        self.Ua = Tensor([output_shape[-1], output_shape[-1]],
                         x=self.U.x[:, :output_shape[-1]],
                         dx=self.U.dx[:, :output_shape[-1]],
                         initializer=weights_initializer,
                         reference=True)
        self.Ui = Tensor([output_shape[-1], output_shape[-1]],
                         x=self.U.x[:, output_shape[-1]:output_shape[-1] * 2],
                         dx=self.U.dx[:, output_shape[-1]:output_shape[-1] * 2],
                         initializer=weights_initializer,
                         reference=True)
        self.Uf = Tensor([output_shape[-1], output_shape[-1]],
                         x=self.U.x[:, output_shape[-1] * 2:output_shape[-1] * 3],
                         dx=self.U.dx[:, output_shape[-1] * 2:output_shape[-1] * 3],
                         initializer=weights_initializer,
                         reference=True)
        self.Uo = Tensor([output_shape[-1], output_shape[-1]],
                         x=self.U.x[:, output_shape[-1] * 3:output_shape[-1] * 4],
                         dx=self.U.dx[:, output_shape[-1] * 3:output_shape[-1] * 4],
                         initializer=weights_initializer,
                         reference=True)

        self.b = Tensor([4 * output_shape[-1]])
        self.ba = Tensor([output_shape[-1]],
                         x=self.b.x[:output_shape[-1]],
                         dx=self.b.dx[:output_shape[-1]],
                         initializer=bias_initializer,
                         reference=True)
        self.bi = Tensor([output_shape[-1]],
                         x=self.b.x[output_shape[-1]:output_shape[-1] * 2],
                         dx=self.b.dx[output_shape[-1]:output_shape[-1] * 2],
                         initializer=bias_initializer,
                         reference=True)
        self.bf = Tensor([output_shape[-1]],
                         x=self.b.x[output_shape[-1] * 2:output_shape[-1] * 3],
                         dx=self.b.dx[output_shape[-1] * 2:output_shape[-1] * 3],
                         initializer=ones,
                         reference=True)
        self.bo = Tensor([output_shape[-1]],
                         x=self.b.x[output_shape[-1] * 3:output_shape[-1] * 4],
                         dx=self.b.dx[output_shape[-1] * 3:output_shape[-1] * 4],
                         initializer=bias_initializer,
                         reference=True)

        # self.Uf = Tensor([output_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.Wf = Tensor([input_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.bf = Tensor(output_shape[-1], initializer=ones)
        # self.Ui = Tensor([output_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.Wi = Tensor([input_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.bi = Tensor(output_shape[-1], initializer=bias_initializer)
        # self.Uo = Tensor([output_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.Wo = Tensor([input_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.bo = Tensor(output_shape[-1], initializer=bias_initializer)
        # self.Ua = Tensor([output_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.Wa = Tensor([input_shape[-1], output_shape[-1]], initializer=weights_initializer)
        # self.ba = Tensor(output_shape[-1], initializer=bias_initializer)

    def forward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        for step in range(self.input_shape[1]):
            h_prev = np.repeat(self.h_init.x, self.input_shape[0], axis=0) if step == 0 else self.h.x[step - 1]

            self._forget_gate_forward(in_tensor.x.swapaxes(0, 1)[step], self.f.x[step], h_prev)
            self._input_gate_forward(in_tensor.x.swapaxes(0, 1)[step], self.i.x[step], self.a.x[step], h_prev)
            self._output_gate_forward(in_tensor.x.swapaxes(0, 1)[step], self.o.x[step], h_prev)
            self._state_update_forward(step)

        n_steps = self.output_shape[1] if len(self.output_shape) > 2 else 1
        np.copyto(out_tensor.x, self.h.x[-n_steps:].swapaxes(0, 1).reshape(self.output_shape))

    def _forget_gate_forward(self, _in: np.array, _out: np.array, h_prev: np.array):
        np.add(h_prev @ self.Uf.x + _in @ self.Wf.x, self.bf.x, out=_out)
        self._sigmoid(_out)

    def _input_gate_forward(self, _in: np.array, _i_out: np.array, _a_out: np.array, h_prev: np.array):
        np.add(h_prev @ self.Ui.x + _in @ self.Wi.x, self.bi.x, out=_i_out)
        self._sigmoid(_i_out)
        np.add(h_prev @ self.Ua.x + _in @ self.Wa.x, self.ba.x, out=_a_out)
        self._tanh(_a_out)

    def _output_gate_forward(self, _in: np.array, _out: np.array, h_prev: np.array):
        np.add(h_prev @ self.Uo.x + _in @ self.Wo.x, self.bo.x, out=_out)
        self._sigmoid(_out)

    def _state_update_forward(self, step: int):
        c_prev = np.repeat(self.c_init.x, self.input_shape[0], axis=0) if step == 0 else self.c.x[step - 1]
        np.add(self.a.x[step] * self.i.x[step], self.f.x[step] * c_prev, out=self.c.x[step])
        np.multiply(np.tanh(self.c.x[step]), self.o.x[step], out=self.h.x[step])

    def backward(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        for step in reversed(range(self.input_shape[1])):
            dc_step = np.zeros_like(self.c_init.dx) if step == self.input_shape[1] - 1 else self.c.dx[step + 1]
            self._state_update_backward(in_tensor, step, dc_step)
            print(self.h.dx[step])
            print(self.c.dx[step])
            # self._forget_gate_forward(in_tensor.x[step], self.f.x[step], h_step)
            # self._input_gate_forward(in_tensor.x[step], self.i.x[step], self.a.x[step], h_step)
            # self._output_gate_forward(in_tensor.x[step], self.o.x[step], h_step)
            # self._state_update_forward(step, c_step)

    def _state_update_backward(self, in_tensor, step, dc_step):
        dh_prev = np.zeros_like(self.h_init.dx) if step == self.input_shape[1] - 1 else self.h.dx[step + 1]
        f_prev = np.zeros_like(self.f.x[step]) if step == self.input_shape[1] - 1 else self.f.x[step + 1]
        if step < self.input_shape[1] - self.output_shape[1]:
            np.copyto(self.h.dx[step], dh_prev)
        else:
            np.add(in_tensor.dx.swapaxes(0, 1)[step], dh_prev, out=self.h.dx[step])
        np.add(self.h.dx[step] * self.o.x[step] * (1 - np.tanh(self.c.x[step]) ** 2), dc_step * f_prev, out=self.c.dx[step])


    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        pass

    def _sigmoid(self, _in):
        np.divide(1, (1 + np.exp(-_in)), out=_in)

    def _tanh(self, _in):
        np.tanh(_in, out=_in)
