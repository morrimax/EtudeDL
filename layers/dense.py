import numpy as np

class dense:

    def __init__(self, inputSize, outputSize):
        self.w = np.random.rand(outputSize, inputSize) * 2 - 1
        self.b = np.random.rand(outputSize, 1) * 2 - 1

    def inference(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def backPropagate(self, gradient, learningRate):
        self.w -= learningRate * np.dot(gradient, self.x.T)
        self.b -= learningRate * np.sum(gradient, axis=1)
        return np.dot(self.w.T, gradient)