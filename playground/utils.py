import numpy as np


def batch_generator(X, y, batch_size):

    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    for idx in range(0, X.shape[0], batch_size):
        yield X[idx: idx + batch_size], y[idx: idx + batch_size]
