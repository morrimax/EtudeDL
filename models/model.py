import losses
import numpy as np

from metrics import accuracy

class model:

    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        if isinstance(layer, losses.Loss):
          self.loss = layer
        else:
            self.layers.append(layer)

    def train(
        self, 
        data,
        batchSize=128,
        epochs=20,
        learningRate=1e-3):

        trainX, trainY, testX, testY = data

        numDatum = trainX.shape[1]

        print (numDatum)

        for epoch in range(epochs):
            for batch in range(numDatum // batchSize):

                batchIndex = np.random.randint(numDatum, size=(batchSize))
                batchData = trainX[:,batchIndex]
                batchLabels = trainY[:,batchIndex]

                activations = self.inference(batchData)

                predictions = activations.pop()
                batchLoss = self.loss.evaluate(batchLabels, predictions)

                gradients = [self.loss.backPropagate()]
                for layer in reversed(self.layers):
                    gradient = layer.backPropagate(gradients[-1], learningRate)
                    gradients.append(gradient)

            activations = self.inference(testX)

            print ('Epoch', epoch,
                   '\n\tLoss', batchLoss,
                   '\n\tAccuracy', accuracy(batchLabels, predictions),
                   '\n\tValidation Loss', self.loss.evaluate(testY, activations[-1]),
                   '\n\tValidation Accuracy', accuracy(testY, activations[-1]))

    def evaluate(self, data, labels):
        return accuracy(labels, predict(data))

    def predict(self, data):
        return inference(data)[-1]

    def inference(self, X):
        activations = [X]
        for layer in self.layers:
            activations.append(layer.inference(activations[-1]))
        return activations