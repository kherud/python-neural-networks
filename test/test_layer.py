import unittest
import nn.layer
import numpy as np
from nn.base import Tensor


class TestDense(unittest.TestCase):
    def setUp(self):
        self.layer1 = nn.layer.Dense(3, 3, None)
        self.layer2 = nn.layer.Dense(3, 2, None)

        self.layer1.W = Tensor([3, 3], x=[[-.5057, .3356, -.3485], [.3987, .1673, -.4597], [-.8943, .8321, -.1121]])
        self.layer1.b = Tensor(3, x=[0, 0, 0])
        self.layer2.W = Tensor([3, 2], x=[[.4047, -.8192, .3662], [.9563, -.1274, -.7252]])
        self.layer2.b = Tensor(2, x=[0, 0])

        self.x1 = Tensor(3, x=[0.4183, 0.5209, 0.0291], dx=[.0451, .0557, -.0891])
        self.y1 = Tensor(3, x=[0.4183, 0.5209, 0.0291])
        self.x2 = Tensor(3, x=[.4883, 0.5599, 0.5140], dx=[-0.3268, 0.3268])
        self.y2 = Tensor(2)

    def test_forward1(self):
        expected = np.array([-.0469, .2406, .0561])

        self.layer1.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward1(self):
        expected = np.array([0.0791, -0.0497, -0.0313])

        self.layer1.backward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)

    def test_calculate_delta_weights1(self):
        expected_weight_deltas = np.array([[.0188, .0235, .0013], [.0233, .0290, .0016], [-.0373, -.0464, -.0026]])
        expected_bias_deltas = np.array([.0451, .0557, -.0891])

        self.layer1.calculate_delta_weights(self.x1, self.y1)