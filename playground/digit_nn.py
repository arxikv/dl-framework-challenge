import logging

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from playground.core import Sequential
from playground.layers import Dense
from playground.activations import ReLU, LogSoftmax
from playground.optimizers import SGD, Adam
from playground.losses import NLLLoss
from playground.utils import batch_generator


class DigitNN:
    """Very simple feed-forward net for digit recognition.

    Args:
        input_size (int): dimensionality of (flattened) inputs, e.g. 64=8*8
        n_classes (int): 10
        hidden_size (int): dimensionality of hidden layer outputs

    Attributes:
        net (Sequential): network layers ending with LogSoftmax activation
        optimizer (Optimizer): gradient-based optimizer used for training

    """

    def __init__(self, input_size, n_classes, hidden_size=50):

        self.net = Sequential()
        self.net.add(Dense(input_size, hidden_size))
        self.net.add(ReLU())
        self.net.add(Dense(hidden_size, n_classes))
        self.net.add(LogSoftmax())

        self.optimizer = SGD(self.net.params(), learning_rate=.01)

    def fit(self, X, y, n_epochs=10, batch_size=32):
        """Train the network for n_epochs.

        Args:
            X (np.ndarray): feature matrix, shape (n_samples, dim)
            y (array_like): integer class labels (0 to n_classes-1), shape (n_samples,)
            n_epochs (int): number of epochs to run
            batch_size (int)

        Returns:
            List[np.float64], mean NLL loss value for each epoch

        """
        loss_epochs = []
        for epoch in range(n_epochs):
            loss_batches = []
            for X_batch, y_batch in batch_generator(X, y, batch_size):

                pred = self.net.forward(X_batch)
                loss_batch = NLLLoss.forward(y_batch, pred)
                loss_batches.append(loss_batch)
                self.net.backward(NLLLoss.backward(y_batch, pred))
                self.optimizer.step()

            loss_epoch = np.mean(loss_batches)
            logging.info(f'Epoch {epoch + 1}/{n_epochs}: loss {loss_epoch}')
            loss_epochs.append(loss_epoch)
        return loss_epochs

    def evaluate(self, X_test, y_test):
        """Calculate, print and return test accuracy.

        """
        pred = self.net.forward(X_test)
        accuracy = np.mean(y_test == np.argmax(pred, axis=1))
        logging.info(f'Test accuracy: {accuracy}')
        return accuracy


if __name__ == '__main__':

    np.random.seed(42)

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)  # scale features to [0, 1] range
    X_test = scaler.transform(X_test)

    clf = DigitNN(input_size=X.shape[1], n_classes=10)

    logging.basicConfig(level='INFO')
    clf.fit(X_train, y_train, n_epochs=100)
    clf.evaluate(X_test, y_test)
