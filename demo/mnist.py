import numpy as np
import tensorflow as tf

from gigann.activation import Sigmoid, Softmax, ReLU
from gigann.base import Tensor
from gigann.initializer import normal, kaiming_normal
from gigann.layer import Dense
from gigann.loss import CrossEntropy
from gigann.metrics import f1_score_mean, accuracy
from gigann.network import NeuralNetwork
from gigann.optimizer import MinibatchGradientDescent, Adam, SimpleAdam, Momentum, Adagrad, RMSProp
from gigann.regularization import Dropout

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
optimizer = RMSProp(loss, weight_decay=1e-8, learning_rate=1e-3)
# optimizer = SimpleAdam(loss, weight_decay=1e-8, learning_rate=1e-3)
metrics = [accuracy, f1_score_mean]
# xavier for tanh sigmoid, kaiming else

neural_network = NeuralNetwork([
    Dense([batch_size, 784], [batch_size, 200], kaiming_normal),
    Dropout([batch_size, 200]),
    ReLU([batch_size, 200]),
    Dense([batch_size, 200], [batch_size, 64], kaiming_normal),
    Dropout([batch_size, 64]),
    ReLU([batch_size, 64]),
    Dense([batch_size, 64], [batch_size, 10], kaiming_normal),
    Softmax([batch_size, 10])
])

print("batch_size:", batch_size)
optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=25,
                   batch_size=batch_size,
                   metrics=metrics)
