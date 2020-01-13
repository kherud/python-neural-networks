import numpy as np

from nn.base import Tensor


def normal(tensor: Tensor):
    tensor.x = 2 * np.random.normal(size=tensor.shape) / np.prod(tensor.shape)


def xavier(tensor: Tensor):
    tensor.x = np.random.uniform(-1, 1, size=tensor.shape) / np.prod(tensor.shape)


def he(tensor: Tensor):
    raise NotImplementedError()