import numpy as np


class NLLLoss:
    """Negative log-likelihood loss.

    """

    @staticmethod
    def forward(target, pred):
        """Calculate mean loss value on mini-batch.

        Args:
            target (array_like): integer class labels (0 to n_classes-1), shape (batch_size,)
            pred (np.ndarray): outputs of LogSoftmax, shape (batch_size, n_classes)

        Returns:
            np.float64

        """
        return np.mean(-pred[range(pred.shape[0]), target])

    @staticmethod
    def backward(target, pred):
        """Calculate loss gradient wrt LogSoftmax outputs.

        Returns:
            np.ndarray, shape (batch_size, n_classes)

        """
        return -np.eye(pred.shape[1])[target] / pred.shape[0]
