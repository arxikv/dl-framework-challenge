import numpy as np


class NLLLoss:

    @staticmethod
    def forward(target, pred):

        return np.mean(-pred[range(pred.shape[0]), target])

    @staticmethod
    def backward(target, pred):

        return -np.eye(pred.shape[1])[target] / pred.shape[0]
