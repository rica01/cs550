from random import shuffle
from typing import Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt


from ANNConfig import ANNConfig
from Layer import Layer


class ANN:

    def __init__(self, seed=None):
        self.Layers = []
        self.learningRate=0.01
        self.epochs=1000
        self.normalizeInput=True
        self.threshold = 10.0
        if seed is not None:
            np.random.seed(seed)

    def loadConfig(self, config):
        self.normalizeInput = config.normalizeInput
        self.learningRate = config.learningRate
        self.epochs = config.epochs
        self.threshold = config.threshold
        self.momentum = config.momentum
        self.trainingType=config.trainingType
        for l in range(config.numLayers):
            inputs = config.layers[l]["inputs"]
            neurons = config.layers[l]["neurons"]
            activationFunction = config.layers[l]["activationFunction"]
            self.addLayer(Layer(inputs, neurons, activationFunction))


    def addLayer(self, layer):
        self.Layers.append(layer)

    def feedForward(self, X):
        for layer in self.Layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        return self.feedForward(X)

    def train(self, X,y):
        if self.trainingType == "b":
            return self.trainBatch(X,y)
        elif self.trainingType == "s":
            return self.trainStochastic(X,y)


    def backPropagationBatch(self, X, y, learningRate):
        output=[]
        for x in X:
            output.append( self.feedForward(x))

        # output = np.array([np.mean(output)])
        for i in reversed(range(len(self.Layers))):
            layer = self.Layers[i]

            if layer == self.Layers[-1]:
                layer.error = np.array([np.mean(y - output)])

                if layer.activation != 'None':
                    layer.delta = layer.error * layer.activationFunctionDerivative(output)
                else:
                    layer.delta = layer.error
            else:
                nextLayer = self.Layers[i + 1]
                layer.error = np.dot(nextLayer.weights, nextLayer.delta)
                layer.delta = layer.error * layer.activationFunctionDerivative(layer.previousIteration)

        for i in range(len(self.Layers)):
            layer = self.Layers[i]
            input = np.atleast_2d(np.array([np.mean(X)]) if i == 0 else self.Layers[i - 1].previousIteration)  * (1- self.momentum)
            layer.weights += layer.delta * input.T * learningRate


    def backPropagationStochastic(self, X, y, learningRate):
        output = self.feedForward(X)

        for i in reversed(range(len(self.Layers))):
            layer = self.Layers[i]

            if layer == self.Layers[-1]:
                layer.error = y - output

                if layer.activation != 'None':
                    layer.delta = layer.error * layer.activationFunctionDerivative(output)
                else:
                    layer.delta = layer.error
            else:
                nextLayer = self.Layers[i + 1]
                layer.error = np.dot(nextLayer.weights, nextLayer.delta)
                layer.delta = layer.error * layer.activationFunctionDerivative(layer.previousIteration)

        for i in range(len(self.Layers)):
            layer = self.Layers[i]
            input = np.atleast_2d(X if i == 0 else self.Layers[i - 1].previousIteration)  * (1- self.momentum)
            layer.weights += layer.delta * input.T * learningRate

    def trainStochastic(self, X, y):
        mses = []
        msesAt=[]
        i=0
        for i in range(self.epochs):
            for j in range(len(X)):
                self.backPropagationStochastic(X[j], y[j], self.learningRate)
            mse = np.mean(np.square(y - self.feedForward(X)))
            mses.append(mse)
            msesAt.append(i)
            if (mse < self.threshold):
                break
        return np.array(mses),np.array(msesAt), i

    def trainBatch(self, X, y):
        mses = []
        msesAt=[]
        i=0
        for i in range(self.epochs):
            self.backPropagationBatch(X, y, self.learningRate)
            mse = np.mean(np.square(y - self.feedForward(X)))
            mses.append(mse)
            msesAt.append(i)
            if (mse < self.threshold):
                break
        return np.array(mses),np.array(msesAt), i




    def getAccuracy(y_pred, y_true):
        return (y_pred == y_true).mean()


    def normalizeVector(self, V):
        max_v = np.abs(V).max()
        V /= max_v
        return V, max_v

