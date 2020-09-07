import numpy as np
import unittest

from pygln import GLN, utils


class TestBackends(unittest.TestCase):

    def _test_binary(self, backend):
        X_train, y_train, X_test, y_test = utils.get_mnist()
        y_train = (y_train == 0)
        y_test = (y_test == 0)

        model = GLN(backend=backend, layer_sizes=[4, 4], input_size=X_train.shape[1])

        output = model.predict(X_train[:1])
        self.assertEqual(output.dtype, y_test.dtype)
        self.assertEqual(output.shape, (1,))

        output = model.predict(X_train[:10], target=y_train[:10])
        self.assertEqual(output.dtype, y_test.dtype)
        self.assertEqual(output.shape, (10,))

        output = model.predict(X_train[:4], target=y_train[:4], return_probs=True)
        self.assertTrue(np.issubdtype(output.dtype, np.floating))
        self.assertEqual(output.shape, (4,))

    def _test_classification(self, backend):
        X_train, y_train, X_test, y_test = utils.get_mnist()

        model = GLN(
            backend=backend, layer_sizes=[4, 4], input_size=X_train.shape[1], num_classes=10
        )

        output = model.predict(X_train[:1])
        self.assertTrue(np.issubdtype(output.dtype, np.integer))
        self.assertEqual(output.shape, (1,))

        output = model.predict(X_train[:10], target=y_train[:10])
        self.assertTrue(np.issubdtype(output.dtype, np.integer))
        self.assertEqual(output.shape, (10,))

        output = model.predict(X_train[:4], target=y_train[:4], return_probs=True)
        self.assertTrue(np.issubdtype(output.dtype, np.floating))
        self.assertEqual(output.shape, (4, 10))

    def test_jax(self):
        self._test_binary(backend='jax')
        self._test_classification(backend='jax')

    def test_numpy(self):
        self._test_binary(backend='numpy')
        self._test_classification(backend='numpy')

    def test_pytorch(self):
        self._test_binary(backend='pytorch')
        self._test_classification(backend='pytorch')

    def test_tf(self):
        self._test_binary(backend='tf')
        self._test_classification(backend='tf')
