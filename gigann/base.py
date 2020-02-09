import numpy as np
from enum import Enum
from typing import Callable, List


class State(Enum):
    TRAIN = 1
    PREDICT = 2


class Tensor:
    # TODO: refactor
    def __init__(self,
                 shape: List = None,
                 x: np.array = None,
                 dx: np.array = None,
                 initializer: Callable = None,
                 reference=False):
        if shape is None:
            assert x is not None
            shape = x.shape
        self.shape = shape
        if x is not None:
            if reference:
                self.x = x.reshape(shape)
                if initializer is not None:
                    initializer(self)
            else:
                self.x = np.array(x).reshape(shape)
        else:
            self.x = np.empty(shape=shape)
            if initializer is not None:
                initializer(self)
        if dx is not None:
            if reference:
                self.dx = dx.reshape(shape)
            else:
                self.dx = np.array(dx).reshape(shape)
        else:
            self.dx = np.empty(shape=shape)
        # self.x = self.x.astype(np.float32)
        # self.dx = self.x.astype(np.float32)
