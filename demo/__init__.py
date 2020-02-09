import os
from typing import List, Tuple

import numpy as np
from urllib.request import urlretrieve
from gigann import Tensor


def load_mnist(path="datasets/mnist.npz", one_hot=True):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        urlretrieve(origin_folder + 'mnist.npz', path)
    with np.load(path, allow_pickle=True) as files:
        x_train = files["x_train"]
        x_test = files["x_test"]
        if one_hot:
            one_hot_mask = np.eye(10)
            y_train = one_hot_mask[files["y_train"]]
            y_test = one_hot_mask[files["y_test"]]
        else:
            y_train = files["y_train"]
            y_test = files["y_test"]

    return (x_train, y_train), (x_test, y_test)


def make_tensors(data, batch_size=32) -> List[Tensor]:
    return [Tensor(x=data[i:i + batch_size]) for i in range(len(data) // batch_size)]

if __name__ == "__main__":
    _ = load_mnist()
    print(_)