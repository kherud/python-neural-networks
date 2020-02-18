from typing import List, Callable, Tuple

import numpy as np

from gigann import Tensor
from gigann.initializer import zeros, ones, xavier_normal
from gigann.layer import TrainableLayer


class FullyConnected(TrainableLayer):
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

    def get_weights(self) -> Tuple:
        return self.W,

    def get_bias(self) -> Tuple:
        return self.b,


class LSTM(TrainableLayer):
    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 weights_initializer: Callable = xavier_normal,
                 bias_initializer: Callable = zeros,
                 train_initial_state=False) -> None:
        super().__init__(input_shape, output_shape, weights_initializer, bias_initializer)
        self.train_initial_state = train_initial_state

        # states
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
        self.h.dx[-1] *= 0
        for step in reversed(range(self.input_shape[1])):
            dc_step = np.zeros_like(self.c_init.dx) if step == self.input_shape[1] - 1 else self.c.dx[step + 1]
            self._state_update_backward(in_tensor, step, dc_step)
            self._output_gate_backward(step)
            self._input_gate_backward(step)
            self._forget_gate_backward(step)

            if step > 0:
                np.dot(self.gates.dx[step], self.U.x.T, out=self.h.dx[step - 1])
            else:
                self.h_init.dx[:] = np.einsum("ik,kj->j", self.gates.dx[step], self.U.x.T)
            out_tensor.dx[:, step] = (self.gates.dx[step] @ self.W.x.T)

    def _state_update_backward(self, in_tensor: Tensor, step: int, dc_step: np.array):
        f_prev = np.zeros_like(self.f.x[step]) if step == self.input_shape[1] - 1 else self.f.x[step + 1]
        if step == self.input_shape[1] - 1 and len(self.output_shape) == 2:
            self.h.dx[step] += in_tensor.dx
        elif step >= self.input_shape[1] - self.output_shape[1] and len(self.output_shape) == 3:
            self.h.dx[step] += in_tensor.dx.swapaxes(0, 1)[step]
        np.add(np.einsum("ij,ij,ij->ij", self.h.dx[step], self.o.x[step], 1 - np.tanh(self.c.x[step]) ** 2),
               dc_step * f_prev,
               out=self.c.dx[step])

    def _output_gate_backward(self, step: int):
        np.einsum("ij,ij,ij,ij->ij", self.h.dx[step], np.tanh(self.c.x[step]), self.o.x[step], 1 - self.o.x[step], out=self.o.dx[step])
        # np.multiply(self.h.dx[step], np.tanh(self.c.x[step]) * self.o.x[step] * (1 - self.o.x[step]), out=self.o.dx[step])

    def _input_gate_backward(self, step: int):
        np.einsum("ij,ij,ij,ij->ij", self.c.dx[step], self.a.x[step], self.i.x[step], 1 - self.i.x[step], out=self.i.dx[step])
        np.einsum("ij,ij,ij->ij", self.c.dx[step], self.i.x[step], 1 - self.a.x[step] ** 2, out=self.a.dx[step])
        # np.multiply(self.c.dx[step], self.a.x[step] * self.i.x[step] * (1 - self.i.x[step]), out=self.i.dx[step])
        # np.multiply(self.c.dx[step], self.i.x[step] * (1 - self.a.x[step] ** 2), out=self.a.dx[step])

    def _forget_gate_backward(self, step: int):
        c_next = np.repeat(self.c_init.x, self.input_shape[0], axis=0) if step == 0 else self.c.x[step - 1]
        np.einsum("ij,ij,ij,ij->ij", self.c.dx[step], c_next, self.f.x[step], 1 - self.f.x[step], out=self.f.dx[step])
        # np.multiply(self.c.dx[step], c_next * self.f.x[step] * (1 - self.f.x[step]), out=self.f.dx[step])

    def calculate_delta_weights(self, in_tensor: Tensor, out_tensor: Tensor) -> None:
        np.einsum("ijk,jil->lk", self.gates.dx, out_tensor.x, out=self.W.dx)
        np.einsum("ijk,ijl->lk", self.gates.dx[1:], self.h.x[:-1], out=self.U.dx)
        np.einsum("ijk->k", self.gates.dx, out=self.b.dx)

    def get_weights(self) -> Tuple:
        if self.train_initial_state:
            return self.W, self.U, self.h_init, self.c_init
        else:
            return self.W, self.U

    def get_bias(self) -> Tuple:
        return self.b,


    def _sigmoid(self, _in):
        np.divide(1, (1 + np.exp(-_in)), out=_in)

    def _tanh(self, _in):
        np.tanh(_in, out=_in)
