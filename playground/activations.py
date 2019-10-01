import numpy as np
from scipy.special import logsumexp

from playground.layers import Layer


class ReLU(Layer):
    """ReLU activation: ReLU(x) = max(0, x).

    Note:
        forward and backward preserve shape.

    """

    def __init__(self):

        super().__init__()
        self.inputs = None

    def forward(self, inputs):

        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_wrt_outputs):

        return (self.inputs > 0) * grad_wrt_outputs

    def params(self):
        """ReLU has no trainable parameters.

        """
        return []


class LogSoftmax(Layer):
    """Log softmax activation.

    logsoftmax_i(x) = log (exp(x_i) / sum_j exp(x_j))

    Note:
        forward and backward preserve shape.

    """

    def __init__(self):

        super().__init__()
        self.outputs = None

    def forward(self, inputs):

        # Log of ratio == difference of logs
        self.outputs = inputs - logsumexp(inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, grad_wrt_outputs):

        grad_sum = np.sum(grad_wrt_outputs, axis=1, keepdims=True)
        return grad_wrt_outputs - np.exp(self.outputs) * grad_sum

    def params(self):
        """LogSoftmax has no trainable parameters.

        """
        return []
