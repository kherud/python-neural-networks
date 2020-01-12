import numpy as np


class Tensor:
    def __init__(self,
                 shape=None,
                 x: np.array = None,
                 dx: np.array = None,
                 initializer = None):
        if shape is None:
            assert x is not None
            shape = x.shape
        self.shape = shape
        if x is not None:
            self.x = np.array(x).reshape(shape)
        else:
            self.x = np.empty(shape=shape)
            if initializer is not None:
                initializer.init(self)
        if dx is not None:
            self.dx = np.array(dx).reshape(shape)
        else:
            self.dx = np.empty(shape=shape)
