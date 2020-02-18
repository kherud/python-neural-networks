from demo import load_mnist, make_tensors
from gigann.initializer import normal, kaiming_normal
from gigann.loss import CrossEntropy
from gigann.metrics import f1_score_mean, accuracy
from gigann.network import NeuralNetwork
from gigann.optimizer import MinibatchGradientDescent, Adam, SimpleAdam, Momentum, Adagrad, RMSProp
from gigann.layer.activation import Sigmoid, Softmax, ReLU
from gigann.layer.regularization import Dropout
from gigann.layer.trainable import FullyConnected

batch_size = 32

(x_train, y_train), (x_test, y_test) = load_mnist()

x_train = make_tensors(x_train.reshape(-1, 784) / 255., batch_size=batch_size)
x_test = make_tensors(x_test.reshape(-1, 784) / 255., batch_size=batch_size)
y_train = make_tensors(y_train, batch_size=batch_size)
y_test = make_tensors(y_test, batch_size=batch_size)


loss = CrossEntropy([batch_size, 10])
optimizer = RMSProp(loss, weight_decay=1e-8, learning_rate=1e-3)
metrics = [accuracy, f1_score_mean]

neural_network = NeuralNetwork([
    FullyConnected([batch_size, 784], [batch_size, 200], kaiming_normal),
    Dropout([batch_size, 200]),
    ReLU([batch_size, 200]),
    FullyConnected([batch_size, 200], [batch_size, 64], kaiming_normal),
    Dropout([batch_size, 64]),
    ReLU([batch_size, 64]),
    FullyConnected([batch_size, 64], [batch_size, 10], kaiming_normal),
    Softmax([batch_size, 10])
])

optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=25,
                   metrics=metrics)
