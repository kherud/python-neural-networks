# 

## Losses

There are two losses, call them by providing an input shape

- Categorical Cross Entropy
- Mean Squared Error

```python3
from gigann.loss import *

loss = CrossEntropy([batch_size, n_labels])
loss = MeanSquaredError([batch_size, n_labels])
```

## Metrics

All optimizers use loss as default metric, you can provide these additional metrics.

- Accuracy
- F1Score

```python3
from gigann.metrics import f1_score_mean, accuracy

metrics = [f1_score_mean, accuracy]
```



## Optimizers

The abstract `Optimizer` class implements basic functionality for training a neural network, i.e. training loop, evaluation, and parameter update.

Define your optimizer by calling a subclass. Of course you need your previously defined loss function to optimize.

- Minibatch Gradient Descent
- Momentum Based Gradient Descent
- Adagrad
- RMSProp
- Simple Adam
- Adam

```python3
from gigann.optimizer import *

optimizer = MinibatchGradientDescent(loss, weight_decay=None, learning_rate=1e-3)
optimizer = Momentum(loss, weight_decay=None, learning_rate=1e-3)
optimizer = Adagrad(loss, weight_decay=None, learning_rate=1e-3)
optimizer = RMSProp(loss, weight_decay=None, learning_rate=1e-3)
optimizer = SimpleAdam(loss, weight_decay=None, learning_rate=1e-3)
optimizer = Adam(loss, weight_decay=None, learning_rate=1e-3)
```

You can then use your optimizer to train your neural network. If no test data is provided the neural network will not be evaluated.

```python3
optimizer.optimize(neural_network: NeuralNetwork,
                   x_train: List[Tensor],
                   y_train: List[Tensor],
                   x_test: List[Tensor] = None,
                   y_test: List[Tensor] = None,
                   epochs: int = 1,
                   metrics: Iterable[Callable] = (),  # You defined these above
                   save_to: str = None,  # Directory to save your trained neural network
                   save_by: str = None)  # Only save models that improves this metric
```