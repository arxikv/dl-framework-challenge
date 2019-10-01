class Sequential:
    """Linear stack of layers.

    Attributes:
        layers (list)

    """

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, inputs):

        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, grad_wrt_outputs):

        current = grad_wrt_outputs
        for layer in reversed(self.layers):
            current = layer.backward(current)
        return current

    def params(self):
        """Get trainable parameters of all layers.

        """
        params_ = []
        for layer in self.layers:
            params_.extend(layer.params())
        return params_
