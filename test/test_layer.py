import unittest
import nn.layer
import numpy as np
from nn.base import Tensor


class TestDense(unittest.TestCase):
    def setUp(self):
        self.layer1 = nn.layer.Dense([1, 3], [1, 3], None)
        self.layer2 = nn.layer.Dense([1, 3], [1, 2], None)

        self.layer1.W = Tensor([3, 3], x=[-0.5057, 0.3987, -0.8943, 0.3356, 0.1673, 0.8321, -0.3485, -0.4597, -0.1121])
        self.layer1.b = Tensor(3, x=[0., 0., 0.])
        self.layer2.W = Tensor([3, 2], x=[0.4047, 0.9563, -0.8192, -0.1274, 0.3662, -0.7252])
        self.layer2.b = Tensor(2, x=[0., 0.])

    def test_forward1(self):
        expected = np.array([-.0469, .2406, .0561]).reshape(1, 3)

        in_tensor = Tensor([1, 3], x=[0.4183, 0.5209, 0.0291])
        out_tensor = Tensor([1, 3])

        self.layer1.forward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.x, decimal=4)

    def test_backward1(self):
        expected = np.array([0.0791, -0.0497, -0.0313]).reshape(1, 3)

        in_tensor = Tensor([1, 3], dx=[0.0451, 0.0557, -0.0891])
        out_tensor = Tensor([1, 3])

        self.layer1.backward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.dx, decimal=4)

    def test_calculate_delta_weights1(self):
        expected_weight_deltas = np.array([0.0188, 0.0233, -0.0373, 0.0235, 0.0290, -0.0464, 0.0013, 0.0016, -0.0026]).reshape(3, 3)
        expected_bias_deltas = np.array([0.0451, 0.0557, -0.0891]).reshape(3)

        in_tensor = Tensor([1, 3], dx=[0.0451, 0.0557, -0.0891])
        out_tensor = Tensor([1, 3], x=[0.4183, 0.5209, 0.0291])

        self.layer1.calculate_delta_weights(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected_weight_deltas, self.layer1.W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_bias_deltas, self.layer1.b.dx, decimal=4)

    def test_forward2(self):
        expected = np.array([-0.0728, 0.0229]).reshape(1, 2)

        in_tensor = Tensor([1, 3], x=[0.4883, 0.5599, 0.5140])
        out_tensor = Tensor([1, 2])

        self.layer2.forward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.x, decimal=4)

    def test_backward2(self):
        expected = np.array([0.1803, 0.2261, -0.3567]).reshape(1, 3)

        in_tensor = Tensor([1, 2], dx=[-0.3268, 0.3268])
        out_tensor = Tensor([1, 3])

        self.layer2.backward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.dx, decimal=4)

    def test_calculate_delta_weights2(self):
        expected_weight_deltas = np.array([-0.1596, 0.1596, -0.1830, 0.1830, -0.1680, 0.1680]).reshape(3, 2)
        expected_bias_deltas = np.array([-0.3268, 0.3268]).reshape(2)

        in_tensor = Tensor([1, 2], dx=[-0.3268, 0.3268])
        out_tensor = Tensor([1, 3], x=[0.4883, 0.5599, 0.5140])

        self.layer2.calculate_delta_weights(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected_weight_deltas, self.layer2.W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_bias_deltas, self.layer2.b.dx, decimal=4)