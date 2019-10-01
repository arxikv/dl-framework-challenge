import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    """Stochastic gradient descent with momentum.

    """

    def __init__(self, params, learning_rate=1e-3, momentum=0.0, nesterov=False):

        super().__init__()
        self.params = params
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        raise NotImplementedError('Implement me!')

    def step(self):

        raise NotImplementedError('Implement me!')


class Adam(Optimizer):

    def __init__(self, params, learning_rate=1e-3, beta_1=.9, beta_2=.999):

        raise NotImplementedError('Implement me!')

    def step(self):

        raise NotImplementedError('Implement me!')
