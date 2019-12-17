import unittest
import nn.activation
import numpy as np
from nn.base import Tensor


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.layer = nn.activation.Sigmoid(3)
        self.x1 = Tensor(3, x=[-0.0469, 0.2406, 0.0561])
        self.x2 = Tensor(3, x=[0.4883, 0.5599, 0.5140], dx=[0.1803, 0.2261, -0.3567])
        self.y1 = Tensor(3)

    def test_forward(self):
        expected = np.array([0.4883, 0.5599, 0.5140])

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([0.0451, 0.0557, -0.0891])

        self.layer.backward(self.x2, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.layer = nn.activation.Softmax([2])
        self.x1 = Tensor(2, x=[-0.0728, 0.0229])
        self.x2 = Tensor(2, x=[0.4761, 0.5239], dx=[-1.4901, -0.1798])
        self.y1 = Tensor(2)

    def test_forward(self):
        expected = np.array([0.4761, 0.5239])

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([-0.3268, 0.3268])

        self.layer.backward(self.x2, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.layer = nn.activation.ReLU(3)
        self.x1 = Tensor(6, x=[-0.0469, 0.2406, 0.0561, -0.0469, 0.2406, 0.0561], dx=[1., 2., 3., 4., 5., 6.])
        self.y1 = Tensor(6)

    def test_forward(self):
        expected = np.array([0, 0.2406, 0.0561, 0, 0.2406, 0.0561])

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([0, 2., 3., 0, 5., 6.])

        self.layer.backward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)


if __name__ == '__main__':
    unittest.main()
