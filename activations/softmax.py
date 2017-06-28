import numpy as np

class softmax:

    def __init__(self):
        self.activation = np.ndarray([])

    def inference(self, x):
        # self.activation = np.divide(np.exp(x), np.sum(np.exp(x), axis=0))
        # return self.activation
        return np.divide(np.exp(x), np.sum(np.exp(x), axis=0))

    def backPropagate(self, gradient, learningRate):
        # (numLabels, batchSize) = self.activation.shape
        # result = np.zeros(gradient.shape)
        # for i in range(batchSize):
        #     deltaSi = self.activation[:,i] * np.identity(numLabels)
        #     SiSj = np.outer(self.activation[:,i], self.activation[:,i])
        #     result[:,i] = np.squeeze(np.dot(deltaSi - SiSj, gradient[:,i]))
        # return result
        return gradient