import numpy as np
import tensorflow as tf

from nn.activation import Sigmoid, Softmax, ReLU, Dropout
from nn.base import Tensor
from nn.initializer import Normal, Xavier
from nn.layer import Dense
from nn.loss import CrossEntropy
from nn.metrics import F1Score, Accuracy
from nn.network import NeuralNetwork
from nn.optimizer import MinibatchGradientDescent, Adam, SimpleAdam, Momentum, Adagrad, RMSProp

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

_x_train, _x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0

one_hot = np.eye(10)
_y_train, _y_test = one_hot[y_train], one_hot[y_test]

batch_size = 32

x_train, y_train, x_test, y_test = [], [], [], []
for i in range(len(_x_train) // batch_size):
    index = batch_size * i
    x_train.append(Tensor(x=_x_train[index:index + batch_size]))
    y_train.append(Tensor(x=_y_train[index:index + batch_size]))
for i in range(len(_x_test) // batch_size):
    index = batch_size * i
    x_test.append(Tensor(x=_x_test[index:index + batch_size]))
    y_test.append(Tensor(x=_y_test[index:index + batch_size]))

loss = CrossEntropy([batch_size, 10])
# optimizer = MinibatchGradientDescent(loss)
optimizer = RMSProp(loss, learning_rate=1e-3)
initializer = Xavier()
metrics = [Accuracy(), F1Score()]

neural_network = NeuralNetwork([
    Dense([batch_size, 784], [batch_size, 256], initializer),
    Sigmoid([batch_size, 256]),
    Dense([batch_size, 256], [batch_size, 64], initializer),
    Sigmoid([batch_size, 64]),
    Dense([batch_size, 64], [batch_size, 10], initializer),
    Softmax([batch_size, 10])
])

print("batch_size:", batch_size)
optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=25,
                   batch_size=batch_size,
                   metrics=metrics)