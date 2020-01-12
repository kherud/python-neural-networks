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
            nn.layer.Dense([1, 3], [1, 3], None),
            nn.activation.Sigmoid([1, 3]),
            nn.layer.Dense([1, 3], [1, 2], None),
            nn.activation.Softmax([1, 2])
        ])

        self.nn2 = NeuralNetwork([
            nn.layer.Dense([16, 3], [16, 3], None),
            nn.activation.Sigmoid([16, 3]),
            nn.layer.Dense([16, 3], [16, 2], None),
            nn.activation.Softmax([16, 2])
        ])

        self.nn1.layers[0].W = self.nn2.layers[0].W = Tensor([3, 3], x=[-0.5057, 0.3987, -0.8943, 0.3356, 0.1673, 0.8321, -0.3485, -0.4597, -0.1121])
        self.nn1.layers[0].b = self.nn2.layers[0].b = Tensor([3], x=[0., 0., 0.])
        self.nn1.layers[2].W = self.nn2.layers[2].W = Tensor([3, 2], x=[0.4047, 0.9563, -0.8192, -0.1274, 0.3662, -0.7252])
        self.nn1.layers[2].b = self.nn2.layers[2].b = Tensor([2], x=[0., 0.])

        self.loss_function = nn.loss.CrossEntropy()

    def test_nn1_forward(self):
        expected = [
            np.array([-0.0469, 0.2406, 0.0561]).reshape(1, 3),
            np.array([0.4883, 0.5599, 0.5140]).reshape(1, 3),
            np.array([-0.0728, 0.0229]).reshape(1, 2),
            np.array([0.4761, 0.5239]).reshape(1, 2)
        ]

        in_tensor = Tensor([1, 3], x=[0.4183, 0.5209, 0.0291])

        self.nn1.forward(in_tensor)

        for index, (tensor, expected_tensor) in enumerate(zip(self.nn1.tensors, expected)):
            np.testing.assert_array_almost_equal(tensor.x, expected_tensor, decimal=4)

    def test_nn1_backward(self):
        expected_deltas = [
            np.array([0.0451, 0.0557, -0.0891]).reshape(1, 3),
            np.array([-0.3268, 0.3268]).reshape(1, 2),
        ]

        expected_weight_deltas = [
            np.array([0.0188, 0.0233, -0.0373, 0.0235, 0.0290, -0.0464, 0.0013, 0.0016, -0.0026]).reshape(3, 3),
            np.array([-0.1596, 0.1596, -0.1830, 0.1830, -0.1680, 0.1680]).reshape(3, 2)
        ]

        expected_bias_deltas = [
            np.array([0.0451, 0.0557, -0.0891]).reshape(3),
            np.array([-0.3268, 0.3268]).reshape(2)
        ]

        in_tensor = Tensor([1, 3], x=[0.4183, 0.5209, 0.0291])
        out_tensor = Tensor([1, 2], x=[0.7095, 0.0942])

        self.nn1.forward(in_tensor)
        self.loss_function.backward(self.nn1.tensors[-1], out_tensor)
        self.nn1.backward(in_tensor)

        np.testing.assert_array_almost_equal(expected_weight_deltas[0], self.nn1.layers[0].W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_weight_deltas[1], self.nn1.layers[2].W.dx, decimal=4)

        np.testing.assert_array_almost_equal(expected_bias_deltas[0], self.nn1.layers[0].b.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_bias_deltas[1], self.nn1.layers[2].b.dx, decimal=4)

        np.testing.assert_array_almost_equal(expected_deltas[0], self.nn1.tensors[0].dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_deltas[1], self.nn1.tensors[2].dx, decimal=4)


    def test_nn2_forward(self):
        expected = [
            np.array([-0.0469, 0.2406, 0.0561] * 16).reshape(16, 3),
            np.array([0.4883, 0.5599, 0.5140] * 16).reshape(16, 3),
            np.array([-0.0728, 0.0229] * 16).reshape(16, 2),
            np.array([0.4761, 0.5239] * 16).reshape(16, 2)
        ]

        in_tensor = Tensor([16, 3], x=[0.4183, 0.5209, 0.0291] * 16)

        self.nn2.forward(in_tensor)

        for index, (tensor, expected_tensor) in enumerate(zip(self.nn2.tensors, expected)):
            np.testing.assert_array_almost_equal(tensor.x, expected_tensor, decimal=4)

    def test_nn2_backward(self):
        expected_deltas = [
            np.array([0.0451, 0.0557, -0.0891] * 16).reshape(16, 3),
            np.array([-0.3268, 0.3268] * 16).reshape(16, 2),
        ]

        expected_weight_deltas = [
            np.array([0.0188, 0.0233, -0.0373, 0.0235, 0.0290, -0.0464, 0.0013, 0.0016, -0.0026]).reshape(3, 3) * 16,
            np.array([-0.1596, 0.1596, -0.1830, 0.1830, -0.1680, 0.1680]).reshape(3, 2) * 16
        ]

        expected_bias_deltas = [
            np.array([0.0451, 0.0557, -0.0891]).reshape(3) * 16,
            np.array([-0.3268, 0.3268]).reshape(2) * 16
        ]

        in_tensor = Tensor([16, 3], x=[0.4183, 0.5209, 0.0291] * 16)
        out_tensor = Tensor([16, 2], x=[0.7095, 0.0942] * 16)

        self.nn2.forward(in_tensor)
        self.loss_function.backward(self.nn2.tensors[-1], out_tensor)
        self.nn2.backward(in_tensor)

        # lower precision (decimals) due to multiplication imprecision build up
        np.testing.assert_array_almost_equal(expected_weight_deltas[0], self.nn2.layers[0].W.dx, decimal=2)
        np.testing.assert_array_almost_equal(expected_weight_deltas[1], self.nn2.layers[2].W.dx, decimal=2)

        np.testing.assert_array_almost_equal(expected_bias_deltas[0], self.nn2.layers[0].b.dx, decimal=2)
        np.testing.assert_array_almost_equal(expected_bias_deltas[1], self.nn2.layers[2].b.dx, decimal=2)

        np.testing.assert_array_almost_equal(expected_deltas[0], self.nn2.tensors[0].dx, decimal=2)
        np.testing.assert_array_almost_equal(expected_deltas[1], self.nn2.tensors[2].dx, decimal=2)


if __name__ == '__main__':
    unittest.main()
