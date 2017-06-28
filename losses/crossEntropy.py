from losses.loss import Loss
import numpy as np

class crossEntropy(Loss):

    def evaluate(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions

        entropy = np.multiply(labels,np.log2(predictions))
        return -np.mean(np.sum(entropy, axis=0))

    def backPropagate(self):
        return (self.predictions - self.labels) / float(self.labels.shape[1])