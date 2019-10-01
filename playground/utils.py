import numpy as np


def batch_generator(X, y, batch_size):
    """Shuffles data and yields mini-batches.

    Args:
        X (np.ndarray): feature matrix, shape (n_samples, dim)
        y (array_like): integer class labels (0 to n_classes-1), shape (n_samples,)
        batch_size (int)

    Returns:
        X_batch (np.ndarray): mini-batch features, shape (batch_size, dim)
        y_batch (array_like): mini-batch targets, shape (batch_size,)

    """
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    for idx in range(0, X.shape[0], batch_size):
        yield X[idx: idx + batch_size], y[idx: idx + batch_size]
