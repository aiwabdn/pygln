import unittest


class TestReadme(unittest.TestCase):

    def test_readme(self):
        from pygln.utils import get_mnist

        X_train, y_train, X_test, y_test = get_mnist()


        y_train_3 = (y_train == 3)
        y_test_3 = (y_test == 3)


        from pygln.numpy import GLN

        model_3 = GLN(layer_sizes=[4, 4, 1], input_size=X_train.shape[1],
                      learning_rate=1e-4)

        # swapped

        from pygln import GLN

        model_3 = GLN(backend='numpy', layer_sizes=[4, 4, 1],
                      input_size=X_train.shape[1], learning_rate=1e-4)


        for n in range(X_train.shape[0]):
            pred = model_3.predict(input=X_train[n:n+1], target=y_train_3[n:n+1])


        preds = []
        batch_size = 100
        for n in range(X_test.shape[0] // batch_size):
            batch = X_test[n * batch_size: (n + 1) * batch_size]
            pred = model_3.predict(batch)
            preds.append(pred)


        import numpy as np
        from sklearn.metrics import accuracy_score

        accuracy_score(y_test_3, np.concatenate(preds, axis=0))


        model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=X_train.shape[1],
                    num_classes=10, learning_rate=1e-4)

        for n in range(X_train.shape[0]):
            model.predict(input=X_train[n:n+1], target=y_train[n:n+1])

        preds = []
        for n in range(X_test.shape[0]):
            preds.append(model.predict(X_test[n]))

        accuracy_score(y_test, np.vstack(preds))


        from pygln.utils import evaluate_mnist

        model_3 = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=784,
                      learning_rate=1e-4)

        print(evaluate_mnist(model_3, mnist_class=3, batch_size=4))


        model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=784,
                    num_classes=10, learning_rate=1e-4)

        print(evaluate_mnist(model, batch_size=4))
