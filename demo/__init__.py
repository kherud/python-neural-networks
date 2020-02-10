import os
from typing import List, Tuple

import numpy as np
from urllib.request import urlretrieve
from gigann import Tensor


def load_mnist(path="datasets/mnist.npz"):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        urlretrieve(origin_folder + 'mnist.npz', path)
    with np.load(path, allow_pickle=True) as files:
        x_train = files["x_train"]
        x_test = files["x_test"]

        one_hot = np.eye(10)
        y_train = one_hot[files["y_train"]]
        y_test = one_hot[files["y_test"]]

    return (x_train, y_train), (x_test, y_test)


def make_tensors(data: np.array, batch_size=32) -> List[Tensor]:
    return [Tensor(x=data[i - batch_size:i]) for i in range(batch_size, len(data), batch_size)]
