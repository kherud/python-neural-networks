import numpy as np
from gigann import make_tensors
from gigann.loss import CrossEntropy
from gigann.metrics import f1_score_mean, accuracy, confusion_matrix
from gigann.network import NeuralNetwork
from gigann.optimizer import MinibatchGradientDescent, Adam, SimpleAdam, Momentum, Adagrad, RMSProp
from gigann.layer.activation import Sigmoid, Softmax, ReLU
from gigann.layer.trainable import FullyConnected, LSTM

from helper import load_mnist

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
    FullyConnected([batch_size, 32], [batch_size, 10]),
    Softmax([batch_size, 10])
])

print("batch_size:", batch_size)
optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=5,
                   metrics=metrics)

# prediction phase
predictions, truths = [], []
for x, y in zip(x_test, y_test):
    out_tensor = neural_network.forward(x)
    predictions.extend(np.argmax(out_tensor.x, axis=-1))
    truths.extend(np.argmax(y.x, axis=-1))

evaluation = confusion_matrix(predictions, truths)
print(evaluation)
