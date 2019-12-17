import unittest
import nn.loss
import nn.layer
import nn.activation
import numpy as np
from nn.base import Tensor
from nn.network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn1 = NeuralNetwork([
            nn.layer.Dense([3], [3], None),
            nn.activation.Sigmoid(3),
            nn.layer.Dense([3], [2], None),
            nn.activation.Softmax([2])
        ])

        self.nn1.layers[0].W = Tensor([3, 3], x=[[-.5057, .3356, -.3485], [.3987, .1673, -.4597], [-.8943, .8321, -.1121]])
        self.nn1.layers[0].b = Tensor([3], x=[0, 0, 0])
        self.nn1.layers[2].W = Tensor([3, 2], x=[[.4047, -.8192], [.3662, .9563], [-.1274, -.7252]])
        self.nn1.layers[2].b = Tensor([2], x=[0, 0])

        self.x1 = Tensor(3, x=[0.4183, 0.5209, 0.0291])
        self.y1 = Tensor(2, x=[0.7095, 0.0942])

        self.loss_function = nn.loss.CrossEntropy()

    def test_nn1_forward(self):
        expected = [
            np.array([-0.0469, 0.2406, 0.0561]),
            np.array([0.4883, 0.5599, 0.5140]),
            np.array([-0.0728, 0.0229]),
            np.array([0.4761, 0.5239])
        ]

        self.nn1.forward(self.x1)

        for index, (tensor, expected_tensor) in enumerate(zip(self.nn1.tensors, expected)):
            np.testing.assert_array_almost_equal(tensor.x, expected_tensor, decimal=4)

    def test_nn1_backward(self):
        expected_deltas = [
            np.array([0.0451, 0.0557, -0.0891]),
            np.array([-0.3268, 0.3268]),
        ]

        expected_weight_deltas = [
            np.array([[.0188, .0235, .0013], [.0233, .0290, .0016], [-.0373, -.0464, -.0026]]),
            np.array([[-.1596, -.1830, -.1680], [.1596, .1830, .1680]])
        ]

        self.nn1.forward(self.x1)
        self.loss_function.backward(self.nn1.tensors[-1], self.y1)
        self.nn1.backward(self.x1)

        np.testing.assert_array_almost_equal(expected_weight_deltas[0], self.nn1.layers[0].W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_weight_deltas[1], self.nn1.layers[2].W.dx, decimal=4)

        np.testing.assert_array_almost_equal(expected_deltas[0], self.nn1.tensors[0].dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_deltas[1], self.nn1.tensors[2].dx, decimal=4)


if __name__ == '__main__':
    unittest.main()
