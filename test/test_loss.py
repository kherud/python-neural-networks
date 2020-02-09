import unittest

import numpy as np

import gigann.loss
from gigann import Tensor


class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.loss1 = gigann.loss.CrossEntropy([1])
        self.loss2 = gigann.loss.CrossEntropy([16])
        self.x1 = Tensor([1, 2], [0.4761, 0.5239])
        self.y1 = Tensor([1, 2], [0.7095, 0.0942])
        self.x2 = Tensor([16, 2], [0.4761, 0.5239] * 16)
        self.y2 = Tensor([16, 2], [0.7095, 0.0942] * 16)

    def test_forward1(self):
        expected = 0.5874

        self.loss1.forward(self.x1, self.y1)

        np.testing.assert_almost_equal(expected, self.loss1.get_loss(), decimal=4)

    def test_backward1(self):
        expected = np.array([-1.4902, -0.1798]).reshape(1, 2)

        self.loss1.backward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.x1.dx, decimal=4)

    def test_forward2(self):
        expected = 0.5874 * 16

        self.loss2.forward(self.x2, self.y2)

        np.testing.assert_array_almost_equal(expected, self.loss2.get_loss(), decimal=2)

    def test_backward2(self):
        expected = np.array([-1.4902, -0.1798] * 16).reshape(16, 2)

        self.loss2.backward(self.x2, self.y2)

        np.testing.assert_array_almost_equal(expected, self.x2.dx, decimal=2)


if __name__ == '__main__':
    unittest.main()
