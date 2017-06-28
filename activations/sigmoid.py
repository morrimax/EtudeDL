import numpy as np

class sigmoid:

    def inference(self, x):
        return 1 / (1 + np.exp(-x))

    def backPropagate(self, gradient, learningRate):
        return np.multiply(gradient, 1 - gradient)