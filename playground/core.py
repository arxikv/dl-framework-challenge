from playground.layers import ParamsLayer


class Sequential:

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

        params_ = []
        for layer in self.layers:
            if isinstance(layer, ParamsLayer):
                params_.extend(layer.params())
        return params_
