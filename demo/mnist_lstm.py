from demo import load_mnist, make_tensors
from gigann.loss import CrossEntropy
from gigann.metrics import f1_score_mean, accuracy
from gigann.network import NeuralNetwork
from gigann.optimizer import MinibatchGradientDescent, Adam, SimpleAdam, Momentum, Adagrad, RMSProp
from gigann.layer.activation import Sigmoid, Softmax, ReLU
from gigann.layer.trainable import Dense, LSTM


batch_size = 32

(x_train, y_train), (x_test, y_test) = load_mnist()

x_train = make_tensors(x_train.reshape(-1, 14, 56) / 255., batch_size=batch_size)
x_test = make_tensors(x_test.reshape(-1, 14, 56) / 255., batch_size=batch_size)
y_train = make_tensors(y_train, batch_size=batch_size)
y_test = make_tensors(y_test, batch_size=batch_size)


loss = CrossEntropy([batch_size, 10])
optimizer = Adam(loss, weight_decay=1e-8, learning_rate=1e-2)  # weight_decay=1e-8,
metrics = [accuracy, f1_score_mean]

neural_network = NeuralNetwork([
    LSTM([batch_size, 14, 56], [batch_size, 14, 64]),
    LSTM([batch_size, 14, 64], [batch_size, 32]),
    Dense([batch_size, 32], [batch_size, 10]),
    Softmax([batch_size, 10])
])

print("batch_size:", batch_size)
optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=25,
                   metrics=metrics)
