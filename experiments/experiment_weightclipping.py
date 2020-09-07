import numpy as np
import os
import time

from pygln import GLN, utils


# Experiment config
num_runs = 10
num_epochs = 1
batch_size = 1
eval_batch_size = 100


# MNIST dataset
train_images, train_labels, test_images, test_labels = utils.get_mnist()


# Record results
os.makedirs('data/results', exist_ok=True)
with open('data/results/weight_clipping.csv', 'w') as file:
    file.write(',' + ','.join(map(str, range(num_runs))) + ',avg-runtime\n')

    # Weight clipping
    for weight_clipping in (1e-1, 2.5e-1, 5e-1, 1e0, 2.5e0, 5e0, 1e1, 2.5e1, 5e1, 1e2):
        file.write(str(weight_clipping))

        # Multiple runs
        start = time.time()
        for run in range(num_runs):

            # Model
            model = GLN(
                backend='pytorch', layer_sizes=[16, 16], input_size=train_images.shape[1],
                num_classes=10, weight_clipping=weight_clipping
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
            print(weight_clipping, run, accuracy)

        # Average runtime
        runtime = (time.time() - start) / num_runs
        print(weight_clipping, runtime)
        file.write(f',{runtime}\n')
