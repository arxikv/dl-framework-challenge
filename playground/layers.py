import numpy as np


class ParamsLayer:

    def params(self):
        pass


class Dense(ParamsLayer):

    def __init__(self, input_size, output_size):

        self.inputs = None

        he_std = np.sqrt(2 / input_size)
        self.weights = np.random.normal(scale=he_std, size=(input_size, output_size))
        self.bias = np.zeros(output_size)

        self.grad_wrt_weights = np.zeros_like(self.weights)
        self.grad_wrt_bias = np.zeros_like(self.bias)

    def forward(self, inputs):

        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_wrt_outputs):

        np.dot(np.transpose(self.inputs), grad_wrt_outputs, out=self.grad_wrt_weights)
        np.sum(grad_wrt_outputs, axis=0, out=self.grad_wrt_bias)
        return np.dot(grad_wrt_outputs, np.transpose(self.weights))

    def params(self):

        params_ = [
            {'value': self.weights, 'grad': self.grad_wrt_weights},
            {'value': self.bias, 'grad': self.grad_wrt_bias}
        ]
        return params_


class Conv2D(ParamsLayer):

    def __init__(self, filters, kernel_size):

        raise NotImplementedError('Implement me!')

    def forward(self, inputs):

        raise NotImplementedError('Implement me!')

    def backward(self, grad_wrt_outputs):

        raise NotImplementedError('Implement me!')

