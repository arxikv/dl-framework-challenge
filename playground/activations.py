import numpy as np
from scipy.special import logsumexp


class Activation:
    pass


class ReLU(Activation):

    def __init__(self):

        self.inputs = None

    def forward(self, inputs):

        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_wrt_outputs):

        return (self.inputs > 0) * grad_wrt_outputs


class LogSoftmax(Activation):

    def __init__(self):

        self.outputs = None

    def forward(self, inputs):

        self.outputs = inputs - logsumexp(inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, grad_wrt_outputs):

        grad_sum = np.sum(grad_wrt_outputs, axis=1, keepdims=True)
        return grad_wrt_outputs - np.exp(self.outputs) * grad_sum
