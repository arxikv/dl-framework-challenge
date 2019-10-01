class Optimizer:
    pass


class SGD(Optimizer):

    def __init__(self, params, learning_rate=1e-3, momentum=0.0):

        self.params = params
        self.lr = learning_rate
        self.momentum = momentum

        raise NotImplementedError('Implement me!')

    def step(self):

        raise NotImplementedError('Implement me!')


class Adam(Optimizer):

    def __init__(self, params, learning_rate=1e-3, beta_1=.9, beta_2=.999):

        raise NotImplementedError('Implement me!')
