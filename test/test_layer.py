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


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.layer1 = nn.layer.LSTM([3, 1, 2], [1, 3], None)

        self.layer1.h_init = Tensor([1, 3], x=[1.3812, 0.5108, 1.7547])
        self.layer1.c_init = Tensor([1, 3], x=[0.5468, 1.3003, -0.8365])

        # weights
        self.layer1.Uf = Tensor([3, 3], x=[-0.3701, 0.3375, -0.4102, 0.38, 0.1149, 0.3829, -0.5364, -0.006, 0.5701])
        self.layer1.Wf = Tensor([2, 3], x=[0.5963, -0.5465, -0.6261, -0.4732, -0.2196, 0.3384])
        self.layer1.bf = Tensor(3, x=[1., 1., 1.])
        self.layer1.Ui = Tensor([3, 3], x=[-0.3474, 0.1011, -0.1903, -0.2617, 0.1884, -0.1364, 0.1827, -0.2626, 0.0687])
        self.layer1.Wi = Tensor([2, 3], x=[-0.225, 0.3247, 0.1201, 0.028, -0.3777, -0.0642])
        self.layer1.bi = Tensor(3, x=[0., 0., 0.])
        self.layer1.Uo = Tensor([3, 3], x=[-0.315, -0.1448, -0.2816, -0.3676, 0.059, -0.4758, -0.103, -0.0204, -0.0876])
        self.layer1.Wo = Tensor([2, 3], x=[0.4518, 0.1388, -0.5324, 0.5744, 0.6232, 0.3959])
        self.layer1.bo = Tensor(3, x=[0., 0., 0.])
        self.layer1.Uc = Tensor([3, 3], x=[0.3936, -0.0901, 0.2262, -0.0697, 0.3737, -0.2524, 0.048, 0.3168, 0.3984])
        self.layer1.Wc = Tensor([2, 3], x=[0.3499, -0.0971, -0.6283, -0.2142, 0.5399, -0.3755])
        self.layer1.bc = Tensor(3, x=[0., 0., 0.])

        self.layer2 = nn.layer.LSTM([2, 2, 2], [2, 1], None)

        self.layer2.h_init = Tensor([2, 1], x=[0., 0.])
        self.layer2.c_init = Tensor([2, 1], x=[0., 0.])

        # weights
        self.layer2.Uf = Tensor([1, 1], x=[0.1])
        self.layer2.Wf = Tensor([2, 1], x=[0.7, 0.45])
        self.layer2.bf = Tensor(1, x=[0.15])
        self.layer2.Ui = Tensor([1, 1], x=[0.8])
        self.layer2.Wi = Tensor([2, 1], x=[0.95, 0.8])
        self.layer2.bi = Tensor(1, x=[0.65])
        self.layer2.Uo = Tensor([1, 1], x=[0.25])
        self.layer2.Wo = Tensor([2, 1], x=[0.6, 0.4])
        self.layer2.bo = Tensor(1, x=[0.1])
        self.layer2.Uc = Tensor([1, 1], x=[0.15])
        self.layer2.Wc = Tensor([2, 1], x=[0.45, 0.25])
        self.layer2.bc = Tensor(1, x=[0.2])


    def test_forward1(self):
        in_tensor = Tensor([3, 1, 2], x=[0.4893, 0.9738, 0.3544, 0.0961, 0.0487, 0.9644])
        out_tensor = Tensor([1, 3])

        self.layer1.forward(in_tensor, out_tensor)

        print(out_tensor.x)

    def test_forward2(self):
        in_tensor = Tensor([2, 2, 2], x=[1., 2., 0.5, 3., 1., 2., 0.5, 3.])
        out_tensor = Tensor([2, 1])

        self.layer2.forward(in_tensor, out_tensor)


