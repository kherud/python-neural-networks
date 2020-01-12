from abc import ABC, abstractmethod

import numpy as np

from nn.base import Tensor


class Initializer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init(self, tensor: Tensor):
        pass


class Normal(Initializer):
    def __init__(self):
        super().__init__()

    def init(self, tensor: Tensor):
        tensor.x = 2 * np.random.normal(size=tensor.shape) / np.prod(tensor.shape)


class Xavier(Initializer):
    def __init__(self):
        super().__init__()

    def init(self, tensor: Tensor):
        tensor.x = np.random.uniform(-1, 1, size=tensor.shape) / np.prod(tensor.shape)


class He(Initializer):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def init(self, tensor: Tensor):
        pass
