import h5py
import logging

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from playground.digit_nn import DigitNN
from playground.optimizers import SGD, Adam


def prepare_data():

    np.random.seed(42)
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)  # scale features to [0, 1] range
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def test_sgd():

    X_train, X_test, y_train, y_test = prepare_data()
    clf = DigitNN(input_size=X_train.shape[1], n_classes=10)
    clf.optimizer = SGD(clf.net.params(), learning_rate=.01)

    loss_epochs = clf.fit(X_train, y_train, n_epochs=100)
    accuracy = clf.evaluate(X_test, y_test)

    h5f = h5py.File('sgd_history.h5', 'r')
    target_loss = h5f['loss'][:]
    target_acc = h5f['accuracy']

    assert np.allclose(loss_epochs, target_loss)
    assert np.allclose(accuracy, target_acc)
    h5f.close()


def test_sgd_momentum():

    X_train, X_test, y_train, y_test = prepare_data()
    clf = DigitNN(input_size=X_train.shape[1], n_classes=10)
    clf.optimizer = SGD(clf.net.params(), learning_rate=.01, momentum=.9)

    loss_epochs = clf.fit(X_train, y_train, n_epochs=50)
    accuracy = clf.evaluate(X_test, y_test)

    h5f = h5py.File('sgd_momentum_history.h5', 'r')
    target_loss = h5f['loss'][:]
    target_acc = h5f['accuracy']

    assert np.allclose(loss_epochs, target_loss)
    assert np.allclose(accuracy, target_acc)
    h5f.close()


def test_sgd_nesterov():

    X_train, X_test, y_train, y_test = prepare_data()
    clf = DigitNN(input_size=X_train.shape[1], n_classes=10)
    clf.optimizer = SGD(clf.net.params(), learning_rate=.01, momentum=.9, nesterov=True)

    loss_epochs = clf.fit(X_train, y_train, n_epochs=50)
    accuracy = clf.evaluate(X_test, y_test)

    h5f = h5py.File('sgd_nesterov_history.h5', 'r')
    target_loss = h5f['loss'][:]
    target_acc = h5f['accuracy']

    assert np.allclose(loss_epochs, target_loss)
    assert np.allclose(accuracy, target_acc)
    h5f.close()
