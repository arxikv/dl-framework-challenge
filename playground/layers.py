import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_wrt_outputs):
        pass

    @abstractmethod
    def params(self):
        pass


class Dense(Layer):
    """Dense layer with He normal weight init.

    Args:
        input_size (int): dimensionality I of input space
        output_size (int): dimensionality O of output space

    Attributes:
        weights (np.ndarray): weight matrix, shape (I, O)
        bias (np.ndarray): bias vector, shape (O,)
        grad_wrt_weights (np.ndarray): loss gradient wrt weights, shape (I, O)
        grad_wrt_bias (np.ndarray): loss gradient wrt bias, shape (O,)

    """

    def __init__(self, input_size, output_size):

        super().__init__()

        self.inputs = None
        he_std = np.sqrt(2 / input_size)
        self.weights = np.random.normal(scale=he_std, size=(input_size, output_size))
        self.bias = np.zeros(output_size)
        self.grad_wrt_weights = np.zeros_like(self.weights)
        self.grad_wrt_bias = np.zeros_like(self.bias)

    def forward(self, inputs):
        """Calculate outputs using current weights and bias.

        Args:
            inputs (np.ndarray): mini-batch features, shape (batch_size, I)

        Returns:
            np.ndarray, mini-batch outputs, shape (batch_size, O)

        """
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad_wrt_outputs):
        """Calculate loss gradient wrt inputs.

        Args:
            grad_wrt_outputs (np.ndarray): grad wrt mini-batch outputs, shape (batch_size, O)

        Returns:
            np.ndarray, grad wrt mini-batch inputs, shape (batch_size, I)

        """
        np.dot(np.transpose(self.inputs), grad_wrt_outputs, out=self.grad_wrt_weights)
        np.sum(grad_wrt_outputs, axis=0, out=self.grad_wrt_bias)
        return np.dot(grad_wrt_outputs, np.transpose(self.weights))

    def params(self):
        """Get list of layer parameters and their gradients.

        """

        params_ = [
            {'value': self.weights, 'grad': self.grad_wrt_weights},
            {'value': self.bias, 'grad': self.grad_wrt_bias}
        ]
        return params_


class Conv2D(Layer):
    """Simple convolutional layer.

    Args:
        input_shape (tuple): height H, width W and input channels C_in, e.g. (128, 128, 3)
        output_channels (int): C_out
        kernel_size (tuple): kernel height kH and width kW, e.g. (5, 5)

    Attributes:
        weights (np.ndarray): weight tensor, shape (kH, kW, C_in, C_out)

    """

    def __init__(self, input_shape, output_channels, kernel_size):

        raise NotImplementedError('Implement me!')

    def forward(self, inputs):
        """
        Args:
            inputs (np.ndarray): mini-batch features, shape (batch_size, H, W, C_in)

        Returns:
            np.ndarray, mini-batch outputs, shape (batch_size, H-kH+1, W-kW+1, C_out)

        """
        raise NotImplementedError('Implement me!')

    def backward(self, grad_wrt_outputs):

        raise NotImplementedError('Implement me!')

    def params(self):

        raise NotImplementedError('Implement me!')


class MaxPooling2D(Layer):
    """Max pooling layer.

    Args:
        input_shape (tuple): height, width and number of channels, e.g. (124, 124, 10)
        pool_size (tuple): pool height and width, e.g. (2, 2)

    """
    def __init__(self, input_shape, pool_size):

        raise NotImplementedError('Implement me!')

    def forward(self, inputs):

        raise NotImplementedError('Implement me!')

    def backward(self, grad_wrt_outputs):

        raise NotImplementedError('Implement me!')

    def params(self):
        """Max pooling has no trainable parameters.

        """
        return []
