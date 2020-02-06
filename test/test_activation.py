import unittest
import gigann.activation
import numpy as np
from gigann.base import Tensor


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.layer = gigann.activation.Sigmoid([1, 3])
        self.x1 = Tensor([1, 3], x=[-0.0469, 0.2406, 0.0561])
        self.x2 = Tensor([1, 3], x=[0.4883, 0.5599, 0.5140], dx=[0.1803, 0.2261, -0.3567])
        self.y1 = Tensor([1, 3])

    def test_forward(self):
        expected = np.array([0.4883, 0.5599, 0.5140]).reshape(1, 3)

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([0.0451, 0.0557, -0.0891]).reshape(1, 3)

        self.layer.backward(self.x2, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.layer = gigann.activation.Softmax([1, 2])
        self.layer2 = gigann.activation.Softmax([2, 2])
        self.x1 = Tensor([1, 2], x=[-0.0728, 0.0229])
        self.x2 = Tensor([1, 2], x=[0.4761, 0.5239], dx=[-1.4901, -0.1798])
        self.y1 = Tensor([1, 2])

    def test_forward(self):
        expected = np.array([0.4761, 0.5239]).reshape(1, 2)

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([-0.3268, 0.3268]).reshape(1, 2)

        self.layer.backward(self.x2, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)

    def test_forward2(self):
        expected = np.array([0.58125141, 0.41874859, 0.55727108, 0.44272892]).reshape(2, 2)

        in_tensor = Tensor([2, 2], x=[0.87421676, 0.5463041, 0.44702225, 0.21692812])
        out_tensor = Tensor([2, 2])

        self.layer2.forward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.x, decimal=4)

    def test_backward2(self):
        expected = np.array([0.12990534, -0.12990534, -0.05538925, 0.05538925]).reshape(2, 2)

        in_tensor = Tensor([2, 2], x=[0.58125141, 0.41874859, 0.55727108, 0.44272892], dx=[0.85576569, 0.3220504, 0.71366714, 0.93816957])
        out_tensor = Tensor([2, 2])

        self.layer2.backward(in_tensor, out_tensor)

        np.testing.assert_array_almost_equal(expected, out_tensor.dx, decimal=4)




class TestReLU(unittest.TestCase):
    def setUp(self):
        self.layer = gigann.activation.ReLU([1, 6])
        self.x1 = Tensor([1, 6], x=[-0.0469, 0.2406, 0.0561, -0.0469, 0.2406, 0.0561], dx=[1., 2., 3., 4., 5., 6.])
        self.y1 = Tensor([1, 6])

    def test_forward(self):
        expected = np.array([0, 0.2406, 0.0561, 0, 0.2406, 0.0561]).reshape(1, 6)

        self.layer.forward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.x, decimal=4)

    def test_backward(self):
        expected = np.array([0, 2., 3., 0, 5., 6.]).reshape(1, 6)

        self.layer.backward(self.x1, self.y1)

        np.testing.assert_array_almost_equal(expected, self.y1.dx, decimal=4)


class TestDropout(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
