import numpy as np

class relu:

    def inference(self, x):
        self.activations = x.clip(0)
        return self.activations

    def backPropagate(self, gradient, learningRate):
        return np.multiply(gradient, self.activations.astype(bool))