import numpy as np
import os
import time

from pygln import baselines, utils


# Experiment config
num_runs = 10
num_epochs = 1
batch_size = 1
eval_batch_size = 100


# MNIST dataset
train_images, train_labels, test_images, test_labels = utils.get_mnist()


# Record results
os.makedirs('data/results', exist_ok=True)
with open('data/results/baseline_learning_rate.csv', 'w') as file:
    file.write(',' + ','.join(map(str, range(num_runs))) + ',avg-runtime\n')

    # Learning rate
    for learning_rate in (1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5, 1e-5):
        file.write(str(learning_rate))

        # Multiple runs
        start = time.time()
        for run in range(num_runs):

            # Model
            model = baselines.MLP(
                layer_sizes=[64], input_size=train_images.shape[1], num_classes=10,
                learning_rate=learning_rate
            )

            # Training
            for n in range((num_epochs * train_images.shape[0]) // batch_size):
                indices = np.arange(n * batch_size, (n + 1) * batch_size)
                indices = indices % train_images.shape[0]
                model.predict(train_images[indices], train_labels[indices])

            # Evaluation
            num_correct = 0
            for n in range(test_images.shape[0] // eval_batch_size):
                indices = np.arange(n * eval_batch_size, (n + 1) * eval_batch_size)
                prediction = model.predict(test_images[indices])
                num_correct += np.count_nonzero(prediction == test_labels[indices])

            # Record accuracy
            accuracy = num_correct / test_images.shape[0]
            file.write(f',{accuracy}')
            print(learning_rate, run, accuracy)

        # Average runtime
        runtime = (time.time() - start) / num_runs
        print(learning_rate, runtime)
        file.write(f',{runtime}\n')
