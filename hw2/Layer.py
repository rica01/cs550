from random import shuffle
from typing import Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt
from ANNConfig import ANNConfig


class Layer:

    def __init__(self, nInput, nNeurons, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.uniform(-1.0, 1.0, (nInput, nNeurons))
        self.activation = activation
        self.bias = bias if bias is not None else np.random.uniform(-1.0, 1.0, nNeurons)
        self.previousIteration = None
        self.error = None
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.previousIteration = self.activationFunction(r)
        return self.previousIteration

    def activationFunction(self, r):
        if self.activation == 'None':
            return r

        elif self.activation == 'tanh':
            return np.tanh(r)

        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        elif self.activation == 'relu':
            return np.max(0, r)

    def activationFunctionDerivative(self, r):
        if self.activation == 'None':
            return r

        elif self.activation == 'tanh':
            return 1 - r ** 2

        elif self.activation == 'sigmoid':
            return r * (1 - r)

        elif self.activation == 'relu':
            if r >= 0:
                return 1
            else:
                return 0
