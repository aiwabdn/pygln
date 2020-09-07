import unittest

from pygln import baselines, utils


class TestBaselines(unittest.TestCase):

    def test_mlp(self):
        X_train, y_train, X_test, y_test = utils.get_mnist()

        model = baselines.MLP(layer_sizes=[4, 4], input_size=X_train.shape[1], num_classes=10)

        output = model.predict(X_train[:1])
        self.assertEqual(output.dtype, y_test.dtype)
        self.assertEqual(output.shape, (1,))

        output = model.predict(X_train[:10], target=y_train[:10])
        self.assertEqual(output.dtype, y_train.dtype)
        self.assertEqual(output.shape, (10,))
