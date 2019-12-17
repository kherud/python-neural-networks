import numpy as np


class Tensor:
    def __init__(self, shape, x=None, dx=None):
        if x:
            self.x = np.array(x).astype(np.float32)
        else:
            self.x = np.empty(shape=shape).astype(np.float32)
        if dx:
            self.dx = np.array(dx).astype(np.float32)
        else:
            self.dx = np.empty(shape=shape).astype(np.float32)
