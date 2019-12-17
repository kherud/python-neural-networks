import unittest
import nn.loss
import numpy as np
from nn.base import Tensor


class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.loss_function = nn.loss.CrossEntropy()
        self.x1 = Tensor(2, [0.4761, 0.5239])
        self.y1 = Tensor(2, [0.7095, 0.0942])

    def test_forward(self):
        expected = np.array([0.5874])

        self.loss_function.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.loss_function.loss.x, decimal=4)

    def test_backward(self):
        expected = np.array([-1.4902, -0.1798])

        self.loss_function.backward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.x1.dx, decimal=4)


if __name__ == '__main__':
    unittest.main()
