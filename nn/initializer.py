import numpy as np

from nn.base import Tensor


def zeros(tensor: Tensor):
    init = np.zeros(shape=tensor.shape)
    np.copyto(tensor.x, init)


def ones(tensor: Tensor):
    init = np.ones(shape=tensor.shape)
    np.copyto(tensor.x, init)


def normal(tensor: Tensor):
    init = 2 * np.random.normal(size=tensor.shape) / np.prod(tensor.shape)
    np.copyto(tensor.x, init)


def xavier_uniform(tensor: Tensor):
    limit = np.sqrt(6 / np.sum(tensor.shape))
    init = np.random.uniform(-limit, limit, size=tensor.shape)
    np.copyto(tensor.x, init)


def xavier_normal(tensor: Tensor):
    std_dev = np.sqrt(2 / np.sum(tensor.shape))
    init = np.random.normal(scale=std_dev, size=tensor.shape)
    np.copyto(tensor.x, init)


def kaiming_uniform(tensor: Tensor):
    limit = np.sqrt(6 / tensor.shape[0])
    init = np.random.uniform(-limit, limit, size=tensor.shape)
    np.copyto(tensor.x, init)


def kaiming_normal(tensor: Tensor):
    std_dev = np.sqrt(2 / tensor.shape[0])
    init = np.random.normal(scale=std_dev, size=tensor.shape)
    np.copyto(tensor.x, init)