import unittest
import numpy as np
import gigann.layer
from gigann.base import Tensor


class TestDense(unittest.TestCase):
    def setUp(self):
        self.layer1 = gigann.layer.Dense([1, 3], [1, 3])
        self.layer2 = gigann.layer.Dense([1, 3], [1, 2])

        self.layer1.W = Tensor([3, 3], x=[-0.5057, 0.3987, -0.8943, 0.3356, 0.1673, 0.8321, -0.3485, -0.4597, -0.1121])
        self.layer1.b = Tensor([3], x=[0., 0., 0.])
        self.layer2.W = Tensor([3, 2], x=[0.4047, 0.9563, -0.8192, -0.1274, 0.3662, -0.7252])
        self.layer2.b = Tensor([2], x=[0., 0.])

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
        self.layer1 = gigann.layer.LSTM([1, 3, 2], [1, 3])

        self.layer1.h_init = Tensor([1, 3], x=[1.3812, 0.5108, 1.7547])
        self.layer1.c_init = Tensor([1, 3], x=[0.5468, 1.3003, -0.8365])

        self.layer1.W.x[:, :3] = np.array([0.3499, -0.0971, -0.6283, -0.2142, 0.5399, -0.3755]).reshape(-1, 3)
        self.layer1.W.x[:, 3:6] = np.array([-0.225, 0.3247, 0.1201, 0.028, -0.3777, -0.0642]).reshape(-1, 3)
        self.layer1.W.x[:, 6:9] = np.array([0.5963, -0.5465, -0.6261, -0.4732, -0.2196, 0.3384]).reshape(-1, 3)
        self.layer1.W.x[:, 9:12] = np.array([0.4518, 0.1388, -0.5324, 0.5744, 0.6232, 0.3959]).reshape(-1, 3)
        self.layer1.U.x[:, :3] = np.array([0.3936, -0.0901, 0.2262, -0.0697, 0.3737, -0.2524, 0.048, 0.3168, 0.3984]).reshape(-1, 3)
        self.layer1.U.x[:, 3:6] = np.array([-0.3474, 0.1011, -0.1903, -0.2617, 0.1884, -0.1364, 0.1827, -0.2626, 0.0687]).reshape(-1, 3)
        self.layer1.U.x[:, 6:9] = np.array([-0.3701, 0.3375, -0.4102, 0.38, 0.1149, 0.3829, -0.5364, -0.006, 0.5701]).reshape(-1, 3)
        self.layer1.U.x[:, 9:12] = np.array([-0.315, -0.1448, -0.2816, -0.3676, 0.059, -0.4758, -0.103, -0.0204, -0.0876]).reshape(-1, 3)
        self.layer1.b.x *= 0

        self.layer2 = gigann.layer.LSTM([1, 2, 2], [1, 2, 1])

        self.layer2.h_init = Tensor([1, 1], x=[0.])
        self.layer2.c_init = Tensor([1, 1], x=[0.])

        self.layer2.W.x[:, :1] = np.array([0.45, 0.25]).reshape(-1, 1)
        self.layer2.W.x[:, 1:2] = np.array([0.95, 0.8]).reshape(-1, 1)
        self.layer2.W.x[:, 2:3] = np.array([0.7, 0.45]).reshape(-1, 1)
        self.layer2.W.x[:, 3:4] = np.array([0.6, 0.4]).reshape(-1, 1)
        self.layer2.U.x[:, :1] = np.array([0.15]).reshape(-1, 1)
        self.layer2.U.x[:, 1:2] = np.array([0.8]).reshape(-1, 1)
        self.layer2.U.x[:, 2:3] = np.array([0.1]).reshape(-1, 1)
        self.layer2.U.x[:, 3:4] = np.array([0.25]).reshape(-1, 1)
        self.layer2.b.x[:1] = np.array([0.2]).reshape(1)
        self.layer2.b.x[1:2] = np.array([0.65]).reshape(1)
        self.layer2.b.x[2:3] = np.array([0.15]).reshape(1)
        self.layer2.b.x[3:4] = np.array([0.1]).reshape(1)

        self.layer3 = gigann.layer.LSTM([1, 3, 2], [1, 3])

        self.layer3.h_init = Tensor([1, 3], x=[0., 0., 0.])
        self.layer3.c_init = Tensor([1, 3], x=[0., 0., 0.])

        self.layer3.W.x[:, :3] = 0.1 * np.arange(6).reshape((3, 2)).T
        self.layer3.W.x[:, 3:6] = 0.1*np.arange(-3, 3).reshape((3, 2)).T
        self.layer3.W.x[:, 6:9] = 0.1*np.arange(-6, 0).reshape((3, 2)).T
        self.layer3.W.x[:, 9:12] = 0.1*np.arange(-4, 2).reshape((3, 2)).T
        self.layer3.U.x[:, :3] = 0.1 * np.arange(9).reshape((3, 3)).T
        self.layer3.U.x[:, 3:6] = 0.1*np.arange(-3, 6).reshape((3, 3)).T
        self.layer3.U.x[:, 6:9] = 0.1*np.arange(-9, 0).reshape((3, 3)).T
        self.layer3.U.x[:, 9:12] = 0.1*np.arange(-6, 3).reshape((3, 3)).T
        self.layer3.b.x[:3] = np.array([1., 1., 1.]).reshape(3)
        self.layer3.b.x[3:6] = np.array([1., 1., 1.]).reshape(3)
        self.layer3.b.x[6:9] = np.array([1., 1., 1.]).reshape(3)
        self.layer3.b.x[9:12] = np.array([1., 1., 1.]).reshape(3)


        self.layer4 = gigann.layer.LSTM([16, 2, 2], [16, 2, 1])

        self.layer4.h_init = Tensor([1, 1], x=[0.])
        self.layer4.c_init = Tensor([1, 1], x=[0.])

        self.layer4.W.x[:, :1] = np.array([0.45, 0.25]).reshape(-1, 1)
        self.layer4.W.x[:, 1:2] = np.array([0.95, 0.8]).reshape(-1, 1)
        self.layer4.W.x[:, 2:3] = np.array([0.7, 0.45]).reshape(-1, 1)
        self.layer4.W.x[:, 3:4] = np.array([0.6, 0.4]).reshape(-1, 1)
        self.layer4.U.x[:, :1] = np.array([0.15]).reshape(-1, 1)
        self.layer4.U.x[:, 1:2] = np.array([0.8]).reshape(-1, 1)
        self.layer4.U.x[:, 2:3] = np.array([0.1]).reshape(-1, 1)
        self.layer4.U.x[:, 3:4] = np.array([0.25]).reshape(-1, 1)
        self.layer4.b.x[:1] = np.array([0.2]).reshape(1)
        self.layer4.b.x[1:2] = np.array([0.65]).reshape(1)
        self.layer4.b.x[2:3] = np.array([0.15]).reshape(1)
        self.layer4.b.x[3:4] = np.array([0.1]).reshape(1)
    
    @unittest.SkipTest
    def test_forward1(self):
        in_tensor = Tensor([1, 3, 2], x=[0.4893, 0.9738, 0.3544, 0.0961, 0.0487, 0.9644])
        out_tensor = Tensor([1, 3])

        self.layer1.forward(in_tensor, out_tensor)

    def test_layer2(self):
        in_tensor = Tensor([1, 2, 2], x=[1., 2., 0.5, 3.])
        out_tensor = Tensor([1, 2, 1], dx=[0.03631, -0.47803])

        expected_gates_x = np.array([0.8177, 0.9608, 0.8519, 0.8175, 0.8498, 0.9811, 0.8703, 0.8499]).reshape((2, 1, 4))
        expected_gates_dx = np.array([-0.0170, -0.0016, -0., 0.0017, -0.0193, -0.0011, -0.0063, -0.0553]).reshape((2, 1, 4))
        expected_h = np.array([0.5363, 0.7719]).reshape((2, 1, 1))
        expected_dh = np.array([0.0180, -0.4780]).reshape((2, 1, 1))
        expected_c = np.array([0.7857, 1.5176]).reshape((2, 1, 1))
        expected_dc = np.array([-0.0534, -0.0711]).reshape((2, 1, 1))
        expected_dW = np.array([-0.0267, -0.0022,  -0.0031, -0.0259, -0.0922, -0.0066, -0.0189, -0.1626]).reshape((2, 4))
        expected_dU = np.array([-0.0103, -0.0005, -0.0033, -0.0297]).reshape((1, 4))
        expected_db = np.array([-0.0364, -0.0027, -0.0063, -0.0536])
        expected_dx = np.array([-0.0081, -0.0048, -0.0474, -0.0307]).reshape((1, 2, 2))

        self.layer2.forward(in_tensor, out_tensor)

        self.layer2.backward(out_tensor, in_tensor)

        self.layer2.calculate_delta_weights(out_tensor, in_tensor)

        np.testing.assert_array_almost_equal(expected_gates_x, self.layer2.gates.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_gates_dx, self.layer2.gates.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_h, self.layer2.h.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_dh, self.layer2.h.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_c, self.layer2.c.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_dc, self.layer2.c.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dW, self.layer2.W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dU, self.layer2.U.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_db, self.layer2.b.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dx, in_tensor.dx, decimal=4)

    def test_layer4(self):
        in_tensor = Tensor([16, 2, 2], x=[1., 2., 0.5, 3.] * 16)
        out_tensor = Tensor([16, 2, 1], dx=[0.03631, -0.47803] * 16)

        expected_gates_x = np.array([0.8177, 0.9608, 0.8519, 0.8175, 0.8498, 0.9811, 0.8703, 0.8499] * 16).reshape((16, 2, 4)).swapaxes(0, 1)
        expected_gates_dx = np.array([-0.0170, -0.0016, -0., 0.0017, -0.0193, -0.0011, -0.0063, -0.0553] * 16).reshape((16, 2, 4)).swapaxes(0, 1)
        expected_h = np.array([0.5363, 0.7719] * 16).reshape((16, 2, 1)).swapaxes(0, 1)
        expected_dh = np.array([0.0180, -0.4780] * 16).reshape((16, 2, 1)).swapaxes(0, 1)
        expected_c = np.array([0.7857, 1.5176] * 16).reshape((16, 2, 1)).swapaxes(0, 1)
        expected_dc = np.array([-0.0534, -0.0711] * 16).reshape((16, 2, 1)).swapaxes(0, 1)
        expected_dW = np.array([-0.4274, -0.0352, -0.0504, -0.4148, -1.4752, -0.1062, -0.3027, -2.6017]).reshape((2, 4))
        expected_dU = np.array([-0.1663, -0.0095, -0.0541, -0.4752]).reshape((1, 4))
        expected_db = np.array([-0.5825, -0.0441, -0.1009, -0.8578])
        expected_dx = np.array([-0.0081, -0.0048, -0.0474, -0.0307] * 16).reshape((16, 2, 2))

        self.layer4.forward(in_tensor, out_tensor)

        self.layer4.backward(out_tensor, in_tensor)

        self.layer4.calculate_delta_weights(out_tensor, in_tensor)

        np.testing.assert_array_almost_equal(expected_gates_x, self.layer4.gates.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_gates_dx, self.layer4.gates.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_h, self.layer4.h.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_dh, self.layer4.h.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_c, self.layer4.c.x, decimal=4)
        np.testing.assert_array_almost_equal(expected_dc, self.layer4.c.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dW, self.layer4.W.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dU, self.layer4.U.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_db, self.layer4.b.dx, decimal=4)
        np.testing.assert_array_almost_equal(expected_dx, in_tensor.dx, decimal=4)

    @unittest.SkipTest
    def test_forward3(self):
        in_tensor = Tensor([1, 3, 2], x=[-1.5, -1., -0.5, 0., 0.5, 1.])
        out_tensor = Tensor([1, 3, 3], dx=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        self.layer3.forward(in_tensor, out_tensor)

        self.layer3.backward(out_tensor, in_tensor)

        self.layer3.calculate_delta_weights(out_tensor, in_tensor)
