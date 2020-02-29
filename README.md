# GigaNN
teh best neural network in town

## Installation
```
python setup.py install
```

## Usage
Complete examples can be found in the demo/ folder
```python
neural_network = NeuralNetwork([
    LSTM([batch_size, 14, 56], [batch_size, 14, 64]),
    LSTM([batch_size, 14, 64], [batch_size, 32]),
    Dropout([batch_size, 32]),
    FullyConnected([batch_size, 32], [batch_size, 10]),
    Softmax([batch_size, 10])
])

loss = CrossEntropy([batch_size, 10])
optimizer = Adam(loss, weight_decay=1e-8, learning_rate=1e-2)

optimizer.optimize(neural_network,
                   x_train, y_train,
                   x_test, y_test,
                   epochs=25,
                   metrics=[accuracy, f1_score_mean])
```

## Reference

### Tensor
```python
import numpy as np
from gigann import Tensor

Tensor(shape: List = None,  # if not explicitly defined, shape is inferred from x
       x: np.array = None,  # Tensor values are set to x
       dx: np.array = None,  # dx is the gradient with respect to x
       initializer: Callable = None,  # Tensor is initialized by this function
       reference=False)  # If set, the tensor references to x instead of copying it
```
Either
- `shape` and optionally `initializer` (if no initializer is specified the memory is not initialized),
- or `x` and optionally `dx`, 
- or `reference=True` and `x` and optionally `initializer`
are specified. In the latter two cases, `x` is reshaped to `shape`.

A helper method is provided to convert data into a list of `Tensor`s where each `Tensor` is a minibatch:
```python
from gigann import Tensor, make_tensor

make_tensors(data: np.array, batch_size=32) -> List[Tensor]
```

#### Initializers
Used to initialize values of (weight) tensors. Tensors are modified in place.
```python
from gigann.initializer import *

zeros(tensor: Tensor) -> None
ones(tensor: Tensor) -> None
normal(tensor: Tensor) -> None
xavier_uniform(tensor: Tensor) -> None
xavier_normal(tensor: Tensor) -> None
kaiming_uniform(tensor: Tensor) -> None
kaiming_normal(tensor: Tensor) -> None
```

### Neural Network
Describes a complete neural network. All layers, described in the subsections, inherit from `Layer`.

```python
from gigann.network import NeuralNetwork

NeuralNetwork(layers: List[Layer])
```

It has the following methods:

```python
from gigann import State

forward(x: Tensor) -> Tensor  # Does a forward/prediction pass on a data set x
set_state(state: State) -> None  # Switches between training and prediction mode (for Dropout)
set_batch_size(batch_size: int) -> None  # Sets the optimizer batch size
                                         # i.e. 1 for prediction, >=1 for training
```

#### Trainable Layers
The core of a neural network.

Fully connected layers have a number of neurons with weights and a bias.

LSTMs are used for modeling time series by using 2D data samples, where every sample contains a number of time steps. LSTM layers have a hidden internal state which propagates information between these time steps. By default, only shared weights and biases are trained.  When the optional `train_initial_state` parameter is set, the initial internal state is also trained, otherwise it is empty.

```python
from gigann.network import NeuralNetwork

FullyConnected(input_shape: List[int],
               output_shape: List[int],
               weights_initializer: Callable[[Tensor], None] = xavier_normal,
               bias_initializer: Callable[[Tensor], None] = zeros)

LSTM(input_shape: List[int],
     output_shape: List[int],
     weights_initializer: Callable[[Tensor], None] = xavier_normal,
     bias_initializer: Callable[[Tensor], None] = zeros,
     train_initial_state=False)
```

#### Activations
Activation layers are used to introduce non-linearity. As per definition, the output shape is identical to the input shape.

```python
from gigann.layer.activation import *

Sigmoid(shape: List[int])
ReLU(shape: List[int])
Softmax(shape: List[int])
```

#### Regularization
Dropout regularization is used to improve generalization of the previous layer during training. Don't forget to call `neural_network.set_state(State.PREDICT)` when evaluating your network to disable Dropout layers.

```python
from gigann.layer.regularization import *

Dropout(shape: List[int],
        rate: float = 0.5)
```

### Losses
Loss functions are used to calculate the error of predictions.

```python
from gigann.loss import *

CrossEntropy(input_shape: List[int])
MeanSquaredError(input_shape: List[int])
```

### Optimizers

The abstract `Optimizer` class implements basic functionality for training a neural network, i.e. training loop, evaluation, and parameter update.

You need a previously defined loss function to optimize. Weight decay penalizes high weights to improve generalization.

```python
from gigann.optimizer import *
from gigann.loss import Loss

MinibatchGradientDescent(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
Momentum(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
Adagrad(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
RMSProp(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
SimpleAdam(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
Adam(loss: Loss, weight_decay: float = None, learning_rate: float = 1e-3)
```

You can then use your optimizer with the following method to train your neural network. If no test data is provided the neural network will not be evaluated.

```python
optimize(neural_network: NeuralNetwork,
         x_train: List[Tensor],
         y_train: List[Tensor],
         x_test: List[Tensor] = None,
         y_test: List[Tensor] = None,
         epochs: int = 1,  # Number of passes over your dataset/all minibatches.
         metrics: Iterable[Callable] = (),  # See subsection below
         save_to: str = None,  # Directory to save your trained neural network
         save_by: str = None)  # Only save models that improve this metric
```
#### Metrics

All optimizers use loss as default metric, you can provide additional metrics from this module.

```python
from gigann.metrics import f1_score_mean, accuracy

metrics = [f1_score_mean, accuracy]
```