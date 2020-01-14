import numpy as np

from nn.base import Tensor


def normal(tensor: Tensor):
    tensor.x = 2 * np.random.normal(size=tensor.shape) / np.prod(tensor.shape)


def xavier_uniform(tensor: Tensor):
    limit = np.sqrt(6 / np.sum(tensor.shape))
    tensor.x = np.random.uniform(-limit, limit, size=tensor.shape)


def xavier_normal(tensor: Tensor):
    std_dev = np.sqrt(2 / np.sum(tensor.shape))
    tensor.x = np.random.normal(scale=std_dev)


def kaiming_uniform(tensor: Tensor):
    limit = np.sqrt(6 / tensor.shape[0])
    tensor.x = np.random.uniform(-limit, limit, size=tensor.shape)


def kaiming_normal(tensor: Tensor):
    std_dev = np.sqrt(2 / tensor.shape[0])
    tensor.x = np.random.normal(scale=std_dev, size=tensor.shape)
