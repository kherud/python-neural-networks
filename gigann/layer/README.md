# Layers

## Activations

There are three activation layers, call them by providing an input shape

- Sigmoid
- ReLU
- Softmax

```python3
from gigann.activations import *

activation = Sigmoid([batch_size, n_neurons])
activation = ReLU([batch_size, n_neurons])
activation = Softmax([batch_size, n_neurons])
```

## Regularization

There is only Dropout. Don't forget to call `neural_network.set_state(State.PREDICT)` when evaluating your network.

```python3
dropout = Dropout([batch_size, n_neurons])
```

## Trainable Layers

- Fully  Connected
- Long Short Term Memory (LSTM)

```python3

```